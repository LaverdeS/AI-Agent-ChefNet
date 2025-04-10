import base64

from typing import List, TypedDict, Annotated, Optional
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from dotenv import load_dotenv


load_dotenv()
vision_llm = ChatOpenAI(model="gpt-4o")


# AnyMessage: class for messages
# add_messages: operator that add the latest message rather than overwriting it with the latest state.
# This is a new concept in LangGraph, where you can add operators in your state to define the way they should interact together.

class AgentState(TypedDict):
    # The document provided
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]


def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Master Wayne often leaves notes with his training regimen or meal plans.
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


# dummy
def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b


# Equip the butler with tools
tools = [
    divide,
    extract_text
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
            
    divide(a: int, b: int) -> float:
        Divide a and b
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
react_graph = builder.compile()

# Show the butler's thought process
image_data = react_graph.get_graph(xray=True).draw_mermaid_png()

with open("chefnet_thought_process.png", "wb") as f:
    f.write(image_data)


if __name__ == "__main__":
    messages = [HumanMessage(content="Divide 6790 by 5")]
    messages = react_graph.invoke({"messages": messages, "input_file": None})

    # Show the messages
    for m in messages['messages']:
        m.pretty_print()

    messages = [HumanMessage(
        content="According to the information found in the provided image. What's the list of ingredients I should buy to prepare the dish?")]
    messages = react_graph.invoke({"messages": messages, "input_file": "data/image_1.jpg"})

    # Show the messages
    for m in messages['messages']:
        m.pretty_print()