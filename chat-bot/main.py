import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


def main():
    """Main function to run the chat bot with Google Gemini model."""
    print("Hello from chat-bot!")
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter API key for Google Gemini: "
        )

    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!"),
    ]
    print(model.invoke(messages))


if __name__ == "__main__":
    main()
