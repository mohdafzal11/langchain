from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model  = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)

topic_prompt = PromptTemplate(
    input_variables=["topic"],  
    template="Write in detailed paragraphs about {topic}."
)

summary_prompt = PromptTemplate(
    input_variables=["text"],  
    template="Write in summary in two lines about {text}."
)

output_parser = StrOutputParser()

chain  = topic_prompt | model | output_parser | summary_prompt | model | output_parser

print("Enter a topic to write about:")
topic = input()

result = chain.invoke({"topic": topic})

print("Generated Summary:")
print(result)
