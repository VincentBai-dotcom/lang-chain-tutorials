import os
import getpass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

def load_env():
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

def main():
    print("Hello from classification!")
    load_env()
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
        Classification
    )
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )
    
    inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
    prompt = tagging_prompt.invoke({"input": inp})
    response = llm.invoke(prompt)
    print(response.model_dump())


if __name__ == "__main__":
    main()
