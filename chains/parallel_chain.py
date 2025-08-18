from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY)

notes_prompt = PromptTemplate(
    input_variables=["text"],
    template="Write a detailed set of notes on the following text: {text}.",
)

quiz_prompt = PromptTemplate(
    input_variables=["text"],
    template="Create a quiz with 5 questions based on the following text: {text}.",)

final_prompt = PromptTemplate(
    input_variables=["notes", "quiz"],
    template="Merge the notes and quiz into a single document with the following format:\n Notes:\n{notes}\n\nQuiz:\n{quiz}",)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
        "notes": notes_prompt | model | parser,
        "quiz": quiz_prompt | model | parser,})

chain = final_prompt | model | parser

final_chain = parallel_chain | chain

text = """
The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""


result = final_chain.invoke({"text":text})
print("Final Result:")
print(result)