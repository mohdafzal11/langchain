from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   
    api_key=GEMINI_API_KEY)


topic_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on the topic: {topic}.",
)

summary_prompt = PromptTemplate(
    input_variables=["report"],
    template="Write a 5 points summary of the report: {report}.",)


parser = StrOutputParser()

print("Input the topic for the report:")
topic = input()
print("Loading the topic...")

result = topic_prompt | model | parser | summary_prompt | model | parser
final_result = result.invoke({"topic": topic})
print(final_result)

