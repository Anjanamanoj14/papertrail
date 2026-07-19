from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from datetime import datetime

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def search(query: str) -> list:
    """Search and return results."""
    results = tavily.search(query, max_results=2)
    return results['results']

def summarize(text: str, topic: str) -> str:
    """Use LLM to summarize search results."""
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Summarize the given text clearly and concisely in 3-4 sentences. Use only the information provided."),
        HumanMessage(content=f"Topic: {topic}\n\nText to summarize:\n{text}")
    ])
    return response.content

def research(topic: str):
    """Full research pipeline."""
    print(f"\n🔍 Researching: {topic}")
    print("=" * 50)

    # Step 1 - Search basics
    print("🌐 Searching basics...")
    basic_results = search(f"{topic} overview")
    basic_text = "\n".join([r['content'][:300] for r in basic_results])
    basic_summary = summarize(basic_text, topic)
    print("✅ Basic search done!")

    # Step 2 - Search latest news
    print("🌐 Searching latest developments...")
    news_results = search(f"{topic} latest 2026")
    news_text = "\n".join([r['content'][:300] for r in news_results])
    news_summary = summarize(news_text, topic)
    print("✅ News search done!")

    # Step 3 - Build report
    filename = topic.replace(" ", "_").lower()
    sources = basic_results + news_results

    report = f"""# PaperTrail Research Report
**Topic:** {topic}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Overview
{basic_summary}

## Latest Developments
{news_summary}

## Sources
"""
    for i, source in enumerate(sources):
        report += f"{i+1}. [{source['title']}]({source['url']})\n"

    # Step 4 - Save report
    with open(f"{filename}_report.md", "w") as f:
        f.write(report)

    print(f"\n💾 Report saved as {filename}_report.md!")
    print("\n--- REPORT PREVIEW ---")
    print(report)

# Test
research("LangGraph AI framework")