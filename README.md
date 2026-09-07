# LangChain Teaching

Beginner examples for building a chatbot with LangChain and in-memory conversation history.

## Setup

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The dependencies are listed in `requirements.txt` and are already installed in `.venv`.

Create a `.env` file in this folder and add your model provider key:

```text
OPENAI_API_KEY=your-api-key
```

## Run the chatbot

```powershell
python chatbot.py
```

The chatbot remembers messages while the process is running. Use `/reset` to clear its memory or `/exit` to quit. The notebook version is in `langchain_chatbot_memory.ipynb`.