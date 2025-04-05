# Rocket Mortgage Chatbot

This is a chatbot that uses LangGraph to handle the state and flow of a Rocket Mortgage application.

(rough instructions)

## Prerequisites

- Python 3.10+
- Poetry 1.8.4+ <2.0.0
- Arcade API key
- OpenAI API key


## Setup

1. Clone the repository

```bash
git clone https://github.com/BlakeItOut/app-chat.git
```

2. Install dependencies

```bash
cd backend
make install
```

3. Create a `.env` file in the root directory and add your API keys

```bash
touch .env
```

4. Ensure tools loaded

```bash
arcade show --local
```


4. Add your API keys to the `.env` file

```bash
ARCADE_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
LANGCHAIN_API_KEY=your_api_key
LANGSMITH_TRACING=true
```

5. Deploy tools

in top level directory

```bash
cd ..

arcade deploy
```

6. In seperate terminal, run langgraph

```bash
langgraph dev
```

