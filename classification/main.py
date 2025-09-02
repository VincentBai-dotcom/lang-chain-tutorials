import os
import getpass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field



def load_env():
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

def main():
    print("Hello from classification!")
    load_env()
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    print(model.invoke("Hello, world!"))


if __name__ == "__main__":
    main()
