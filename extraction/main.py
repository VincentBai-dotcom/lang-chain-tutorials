from typing import List, Optional
import getpass
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import tool_example_to_messages




class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

def load_env():
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

def main():
    print("Hello from extraction!")
    load_env()

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    examples = [
        (
            "The ocean is vast and blue. It's more than 20,000 feet deep.",
            Data(people=[]),
        ),
        (
            "Fiona traveled far from France to Spain.",
            Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
        ),
    ]
    messages = []
    for txt, tool_call in examples:
        if tool_call.people:
            # This final message is optional for some providers
            ai_response = "Detected people."
        else:
            ai_response = "Detected no people."
        messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))
    for message in messages:
        message.pretty_print()
    message_no_extraction = {
        "role": "user",
        "content": "The solar system is large, but earth has only 1 moon.",
    }
    structured_llm = llm.with_structured_output(schema=Data)
    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
    print(structured_llm.invoke(messages + [message_no_extraction]))



if __name__ == "__main__":
    main()
