from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from langchain_core.runnables import RunnableBranch , RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   
    api_key=GEMINI_API_KEY)

class SentimentAnalysisOutput(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the text."
    )
    
sentiment_parser = PydanticOutputParser(pydantic_object=SentimentAnalysisOutput)

sentiment_prompt = PromptTemplate( 
    input_variables=["feedback"],
    partial_variables= {"format_instructions":sentiment_parser.get_format_instructions()},
    template="What is the sentiment of the following feedbacl? {feedback} '\n\nFormat instructions: {format_instructions} "
)

postive_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""The sentiment of the following feedback is positive: {feedback} . Write a positive response to this feedback.""",
)

negative_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""The sentiment of the following feedback is negative: {feedback} . Write a negative response to this feedback.""",
)

parser = StrOutputParser()

simple_chain = sentiment_prompt | model | sentiment_parser

feedback = "I love the new features of this product, they are amazing!"


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", postive_prompt | model | parser),
    (lambda x: x.sentiment == "negative", negative_prompt | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
    )

final_chain = simple_chain | branch_chain

result = final_chain.invoke({"feedback": feedback})

print("From the following feedback:")
print(result)


