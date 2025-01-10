import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

os.environ['GOOGLE_API_KEY'] = 'AIzaSyDTNv_WRwsdPr4LjoIht25VKsnRrws_nXU'
os.environ['TAVILY_API_KEY'] = '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'

tavily_search = TavilySearchResults(max_results=3)
@tool
def web_search(input_: str) -> dict | str:
    '''
    Search the web for desired thing.
    '''
    results = tavily_search.invoke(input_)
    return results

tools = [web_search]

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', max_output_tokens=None, timeout=None)

prompt = ChatPromptTemplate(
    [
        ('system', '''
                    You are a helpful assistant. If the user asks for a web search or a thing you do not know, use web_search(<user-query>) and extract urls from it's output then format your response as follows:
                    (search-results:
                    ``<url-1>,<url-2>,...``)
                    Do not use emojis in your responses.
                    Speech-Based Conversations: The user interacts with you through voice input. 
                    Ensure responses are clear, concise, and appropriate for spoken language.
                    Avoid overly long or complex sentences. Use conversational tone and natural phrasing.
                   '''),
        ('placeholder', '{chat_history}'),
        ('human', '{input}'),
        ('placeholder', '{agent_scratchpad}')
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
