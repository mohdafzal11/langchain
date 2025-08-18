from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

prompt = PromptTemplate(
    input_variables=["input"],  
    template="Write 5 funny jokes about {input}.")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY)

parser = StrOutputParser()

print("Input the topic for jokes:")
topic = input()

chain = prompt | model | parser

result = chain.invoke({"input": topic})

print(result)
