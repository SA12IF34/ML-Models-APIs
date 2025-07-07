import os
import sqlite3

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

os.environ['DEEPSEEK_API_KEY'] = 'sk-f8c9ff52eea44cd8a00447c0878a3cde'
os.environ['TAVILY_API_KEY'] = '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'

tavily_search = TavilySearchResults(max_results=3)
@tool
def web_search(input_: str) -> dict | str:
    """
    Search the web for desired thing
    """

    results = tavily_search.invoke(input_)
    return results


tools = [web_search]

class State(TypedDict):

    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatDeepSeek(
    model='deepseek-chat',
    temperature=1.5,
    max_retries=3
)
llm = llm.bind_tools(tools)



def chatbot(state: State):

    message = llm.invoke(state['messages'])
    assert len(message.tool_calls) <= 1
    return {'messages': [message]}

tool_node = ToolNode(tools=tools)

# memory = MemorySaver()

graph_builder.add_node('tools', tool_node)
graph_builder.add_node('chatbot', chatbot)

graph_builder.add_edge(START, 'chatbot')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)
graph_builder.add_edge('tools', 'chatbot')


conn = sqlite3.connect('checkpoint.sqlite3', check_same_thread=False)
memory = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=memory)
config = {'configurable': {'thread_id': '1'}}

