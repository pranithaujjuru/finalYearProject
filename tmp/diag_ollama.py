import asyncio
from langchain_community.chat_models import ChatOllama

async def main():
    print("Checking Ollama connection...")
    try:
        model = ChatOllama(model="llama3.2")
        response = await model.ainvoke("Say hello")
        print(f"Response: {response.content}")
        print("✅ Ollama is working via LangChain")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
