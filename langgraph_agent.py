from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    results = tavily.search(query, max_results=2)
    output = ""
    for i, result in enumerate(results['results']):
        output += f"Source {i+1}: {result['title']}\n"
        output += f"URL: {result['url']}\n"
        output += f"Content: {result['content'][:300]}\n\n"
    return output

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Create a ReAct agent - this is the LangGraph magic!
agent = create_react_agent(
    model=llm,
    tools=[search_web],
    prompt="You are PaperTrail, an AI research assistant. Search the web for current information before answering."
)

print("PaperTrail agent created!")
print("Testing agent now...\n")

# Run the agent
response = agent.invoke({
    "messages": [HumanMessage(content="What are the latest AI agent developments in 2026?")]
})

# Print final response
print("Final research report:")
print(response['messages'][-1].content)