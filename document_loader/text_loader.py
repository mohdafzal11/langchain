from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.text import TextLoader


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY)

prompt = PromptTemplate(
    input_variables=["poem"],
    template="Write a summary of the following poem:\n\n{poem}\n\nSummary:")

parser = StrOutputParser()

document = TextLoader (file_path="cricket.txt" , encoding="utf-8")

poem = document.load()

chain = prompt | model | parser
result = chain.invoke({"poem": poem[0].page_content})

print("Summary of the poem:")
print(result)
