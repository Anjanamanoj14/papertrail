from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("Groq key loaded:", "✅" if groq_key else "❌")
print("Tavily key loaded:", "✅" if tavily_key else "❌")

from langchain_groq import ChatGroq

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# Your first LangChain call
response = llm.invoke("What is artificial intelligence in 2 sentences?")

print("\nFirst LangChain call:")
print(response.content)

from langchain_core.prompts import ChatPromptTemplate

# A prompt template - reusable structure for LLM calls
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are PaperTrail, an expert research assistant. Be concise and factual."),
    ("user", "Research this topic briefly: {topic}")
])

# Chain = prompt → llm (pipe them together)
chain = prompt | llm

# Test the chain
response = chain.invoke({"topic": "Large Language Models"})

print("\nPaperTrail chain response:")
print(response.content)

# ============================================
# PAPERTRAIL - DAY 1 NOTES
# ============================================

# 1. LangChain
#    - Framework for building LLM applications
#    - llm.invoke("prompt") = single LLM call
#    - chain = prompt | llm = pipe steps together

# 2. ChatPromptTemplate
#    - Reusable prompt structure
#    - system message = tells LLM who it is
#    - user message = actual input with {placeholders}

# 3. Groq
#    - Free LLM API using Llama 3.1
#    - Fast, free, perfect for development
#    - model = "llama-3.1-8b-instant"

# 4. .env file
#    - Stores API keys safely
#    - Never commit this to GitHub!
#    - load_dotenv() loads keys into os.getenv()

print("Day 1 complete! First LangChain chain working!")