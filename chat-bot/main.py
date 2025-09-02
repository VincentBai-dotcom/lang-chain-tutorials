import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


def main():
    """Main function to run the chat bot with Google Gemini model."""
    print("Hello from chat-bot!")
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter API key for Google Gemini: "
        )
    system_template = "Translate the following from English into {language}"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    response = model.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    main()
