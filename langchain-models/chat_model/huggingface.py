import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="lmstudio-community/Qwen3-4B-Thinking-2507-GGUF",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm , api_key = HUGGINGFACEHUB_API_TOKEN)

response = model.invoke("Hello, how are you today?")
print("chat model response:", response.content)
