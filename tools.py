from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    results = tavily.search(query, max_results=3)
    output = ""
    for i, result in enumerate(results['results']):
        output += f"Source {i+1}: {result['title']}\n"
        output += f"URL: {result['url']}\n"
        output += f"Content: {result['content']}\n\n"
    return output

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llm_with_tools = llm.bind_tools([search_web])

# Step 1 - initial call
messages = [
    SystemMessage(content="You are PaperTrail, an AI research assistant. Always use the search_web tool to find current information before answering."),
    HumanMessage(content="What are the latest developments in LangGraph in 2026?")
]

response = llm_with_tools.invoke(messages)

# Step 2 - execute tool if called
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"Tool called: {tool_call['name']}")
    print(f"Query: {tool_call['args']['query']}")
    
    search_result = search_web.invoke(tool_call['args']['query'])
    print("\nSearch done! Feeding results to LLM...")
    
    messages.append(response)
    messages.append(ToolMessage(
        content=search_result,
        tool_call_id=tool_call['id']
    ))
    
    # Step 3 - final response
    final = llm.invoke(messages)
    print("\nFinal response:")
    print(final.content)
else:
    print("LLM response:", response.content)



# ============================================
# PAPERTRAIL - DAY 2 NOTES
# ============================================

# 1. What is a Tool in AI
#    - Gives LLM ability to interact with real world
#    - LLM decides WHEN to call a tool autonomously
#    - LLM decides WHAT query to use
#    - Tool returns results → LLM reads and responds

# 2. @tool decorator
#    - Converts any Python function into a LangChain tool
#    - Docstring tells LLM what the tool does
#    - LLM reads docstring to decide when to use it

# 3. bind_tools()
#    - Attaches tools to the LLM
#    - llm_with_tools = llm.bind_tools([search_web])
#    - LLM can now call any bound tool autonomously

# 4. Tool call loop
#    - Step 1: LLM decides to call tool
#    - Step 2: We execute the tool and get results
#    - Step 3: Feed results back to LLM as ToolMessage
#    - Step 4: LLM reads results and gives final answer

# 5. Tavily
#    - Real time web search API
#    - Free tier available
#    - Returns title, URL, content for each result

print("Day 2 complete! PaperTrail can search the web!")    