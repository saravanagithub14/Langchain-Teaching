"""A beginner-friendly LangChain chatbot with in-memory conversation history."""

import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"
SESSION_ID = "cli-demo"


store: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the conversation history for a session, creating it if needed."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chatbot() -> RunnableWithMessageHistory:
    """Build a prompt, model, and memory-enabled LangChain runnable."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly tutor for beginners learning LangChain. "
                "Explain ideas simply and use short examples.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{messages}"),
        ]
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
    chain = prompt | model

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="history",
    )


def main() -> None:
    """Run the interactive chatbot in a terminal."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env or set it in your terminal."
        )

    chatbot = build_chatbot()
    config = {"configurable": {"session_id": SESSION_ID}}

    print("LangChain teaching chatbot")
    print("Type /reset to clear memory or /exit to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not question:
            continue
        if question.lower() in {"/exit", "/quit"}:
            print("Goodbye!")
            return
        if question.lower() == "/reset":
            store.pop(SESSION_ID, None)
            print("Memory cleared.")
            continue

        response = chatbot.invoke({"messages": question}, config=config)
        print(f"\nBot: {response.content}")


if __name__ == "__main__":
    main()
