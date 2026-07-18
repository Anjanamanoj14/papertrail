from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

@tool
def get_current_date() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def save_report(filename: str, content: str) -> str:
    """Save a research report to a markdown file."""
    with open(f"{filename}.md", "w") as f:
        f.write(f"# Research Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(content)
    return f"Report saved as {filename}.md successfully!"

# Tool registry
tools = {
    "get_current_date": get_current_date,
    "save_report": save_report
}

llm_with_tools = llm.bind_tools(list(tools.values()))

# Manual agent loop - more reliable than create_react_agent
messages = [
    SystemMessage(content="You are PaperTrail. Use tools to complete tasks."),
    HumanMessage(content="Get the current date, then save a report called 'mcp_intro' with content about what MCP protocol is.")
]

print("PaperTrail running...\n")

# Run up to 3 iterations
for i in range(3):
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("Final response:")
        print(response.content)
        break

    for tool_call in response.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        print(f"Tool called: {tool_name}")
        print(f"Args: {tool_args}\n")

        result = tools[tool_name].invoke(tool_args)
        print(f"Result: {result}\n")

        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call['id']
        ))

# ============================================
# PAPERTRAIL - DAY 4 NOTES
# ============================================

# 1. What is MCP (Model Context Protocol)
#    - Anthropic's open standard for AI-tool connections
#    - Like USB — one protocol, connects to everything
#    - Every major company building MCP servers in 2026
#    - Real MCP servers run separately, agent connects via protocol

# 2. Multiple tools in one agent
#    - bind_tools([tool1, tool2, tool3])
#    - Agent decides which tool to call and when
#    - Can call multiple tools in sequence autonomously

# 3. Manual agent loop vs create_react_agent
#    - create_react_agent = automatic but less control
#    - Manual loop = more reliable for small free models
#    - Both use same tool call pattern from Day 2

# 4. File saving tool
#    - Agent can write files to disk autonomously
#    - save_report creates .md files with timestamps
#    - This is the foundation of PaperTrail's report feature

# 5. Tool registry pattern
#    - tools = {"name": function}
#    - Lets us call any tool by name dynamically
#    - Cleaner than if/else for multiple tools

print("Day 4 complete! Multi-tool agent working!")        