import base64

import logging
import colorlog
import os
import sys
import gradio as gr

from pathlib import Path
from PIL import Image
from llama_index.core import SimpleDirectoryReader

from huggingface_hub import whoami
from typing import List, TypedDict, Annotated, Optional
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
from langchain.docstore.document import Document

from dotenv import load_dotenv


load_dotenv()
vision_llm = ChatOpenAI(model="gpt-4o")

log_colors = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(reset)s %(message)s",
    log_colors=log_colors,
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(handler)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logger.debug(f"running on {sys.platform}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load data from directory
input_dir = Path("C:/Users/lavml/Documents/Datasets/cooking_fast/")
logger.info(f"huggingface user fullname: {whoami()['fullname']}")
logger.info(f"\n[loading data from {input_dir}]")

documents_dataset = SimpleDirectoryReader(input_dir=input_dir).load_data(show_progress=True)  # encoding="latin-1

docs = [
    Document(
        page_content="\n".join([
            f"text: {doc.text_resource.text}",
        ]),
        metadata={"id": doc.id_, "file_name": doc.metadata["file_name"]}
    )
    for doc in documents_dataset
]

bm25_retriever = BM25Retriever.from_documents(docs)


def retrieve_text_from_cookbook(query: str) -> str:
    """
    Retrieves information from a knowledge cookbook using a bm25_retriever.
    """
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."


cookbook_search_retriever_tool = Tool(
    name="guest_info_retriever",
    func=retrieve_text_from_cookbook,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

#---------------------------------------------------------------------------
# AnyMessage: class for messages
# add_messages: operator that add the latest message rather than overwriting it with the latest state.
# This is a new concept in LangGraph, where you can add operators in your state to define the way they should interact together.

class AgentState(TypedDict):
    # The document provided
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]


# tool
def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    The files often contain meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:
        # Read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call the vision-capable model
        response = vision_llm.invoke(message)

        # Append extracted text
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""


# Equip the Chef with tools
tools = [
    extract_text,
    retrieve_text_from_cookbook
]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


def assistant(state: AgentState):
    # System message
    textual_description_of_tool="""
    extract_text(img_path: str) -> str:
        Extract text from an image file using a multimodal model.
    
        Args:
            img_path: A local image file path (strings).
    
        Returns:
            A single string containing the concatenated text extracted from each image.
        
    retrieve_text_from_cookbook(query: str) -> str:
        Retrieves information from a knowledge cookbook using a bm25_retriever.
        
        Args:
            query: The key query to search in the cookbook to retrieve information from.
            
        Returns:
            A string with the page_content which is the matching content to the reference query.
    """
    image=state["input_file"]
    sys_msg = SystemMessage(content=f"You are an helpful cooking assistant called ChefNet. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }


# The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
chefnet = builder.compile()  # react_graph

# Show the Chef's thought process
image_data = chefnet.get_graph(xray=True).draw_mermaid_png()

with open("chefnet_thought_process.png", "wb") as f:
    f.write(image_data)


def respond(user_message, image_path, history):
    """
    Process the input text and optional image.

    Args:
        user_message (str): The text message from the user.
        image_path (str or None): File path to the uploaded image.
        history (list): List of tuples holding conversation history.

    Returns:
        tuple: (new_history, cleared_text, cleared_image)
    """
    # Start building the bot's response based on text input.
    metadata = f"Got your message: '{user_message}'"

    # If an image file path was provided, process it.
    if image_path:
        if os.path.exists(image_path):
            try:
                # Open the image and extract its size and format info.
                img = Image.open(image_path)
                metadata += f" and the image has size {img.size} (format: {img.format})."
            except Exception as e:
                metadata += f" but there was an error opening the image: {e}"
        else:
            metadata += " but the provided image path is invalid."

    messages = [HumanMessage(content=user_message)]
    response = chefnet.invoke({"messages": messages, "input_file": image_path})

    # Show the messages
    for m in response['messages']:
        m.pretty_print()

    final_answer = response['messages'][-1].content
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": f"metadata: {metadata}"},
        {"role": "assistant", "content": final_answer}
    ]
    return history, "", None


if __name__ == "__main__":
    # Build the Gradio Blocks layout.
    with gr.Blocks() as demo:
        # Chatbot component to display conversation history.
        chatbot_history = gr.Chatbot(label="Chat History", type="messages")

        # Create a row for user inputs.
        with gr.Row():
            txt_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Message"
            )
            image_input = gr.Image(
                type="filepath",  # image is received as a file path (a str)
                label="Upload an Image (optional)"
            )

        # A button to submit the message.
        submit = gr.Button("Send")

        # State is used to hold the conversation history.
        state = gr.State([])

        # Wire up the event: when the button is clicked, process the inputs.
        submit.click(
            fn=respond,
            inputs=[txt_input, image_input, state],
            outputs=[chatbot_history, txt_input, image_input],
            queue=True
        )

    # Launch the app.
    demo.launch()


# todo: separate tools and retriever (add SentenceTransformers, or vectorDB as Itty)
# https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG/tree/main
# todo: [opc] ideally have the 3 rag options, to compare them, and add traceability/observability

# build on top of that if you need to, kick off LangGraph course