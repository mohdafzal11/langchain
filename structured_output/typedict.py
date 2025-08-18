from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import TypedDict , Literal , Optional , Annotated


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)


class Review(TypedDict):
    name : Annotated[Optional[str], "Name of the reviewer"]
    summary: Annotated[str, "Summary of the review under 20 words"]
    rating:Annotated[Optional[float], "Rating of the product from 1 to 5"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Sentiment of the review"]
    pros : Annotated[Optional[list[str]], "Pros of the review"]
    cons : Annotated[Optional[list[str]], "Cons of the review"]
    
structured_output = model.with_structured_output(Review)


result = structured_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful             
Review by Mohd Afzal
""")
print(result)

