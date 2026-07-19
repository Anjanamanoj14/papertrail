from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from tavily import TavilyClient
from datetime import datetime

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    results = tavily.search(query, max_results=2)
    output = ""
    for result in results['results']:
        output += f"Title: {result['title']}\n"
        output += f"Content: {result['content'][:200]}\n\n"
    return output

@tool
def save_report(filename: str, content: str) -> str:
    """Save a research report to a markdown file."""
    with open(f"{filename}.md", "w") as f:
        f.write(f"# Research Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(content)
    return f"Report saved as {filename}.md successfully!"

tools = {
    "search_web": search_web,
    "save_report": save_report
}

llm_with_tools = llm.bind_tools(list(tools.values()))

# Memory - this list grows as conversation continues
memory = [
    SystemMessage(content="You are PaperTrail, an AI research assistant. Use search_web to find information and save_report to save findings. Be concise.")
]

def run_agent(user_message: str):
    """Run agent with memory of previous conversations."""
    print(f"\nUser: {user_message}")
    print("-" * 40)

    # Add user message to memory
    memory.append(HumanMessage(content=user_message))

    # Run agent loop
    for i in range(3):
        response = llm_with_tools.invoke(memory)
        memory.append(response)

        if not response.tool_calls:
            print(f"PaperTrail: {response.content}")
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            print(f"Using tool: {tool_name}")

            result = tools[tool_name].invoke(tool_call['args'])
            print(f"Result: {result[:100]}...")

            memory.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            ))

# Test memory with 2 questions
run_agent("Search for what LangGraph is")
run_agent("Now save what you just found in a file called 'langgraph_report'")