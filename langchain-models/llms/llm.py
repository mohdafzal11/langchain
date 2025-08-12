import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

llm = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
response = llm.invoke("Hello, how are you today?")

print("llm response" , response)
