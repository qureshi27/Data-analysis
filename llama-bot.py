# pip install -U langchain langchain-experimental langchain-openai pandas
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# 1) Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2) Build CSV agent
agent = create_csv_agent(
    llm,
    "batch_details.csv",               # change to your CSV path
    agent_type="openai-tools",   # recommended agent type
    allow_dangerous_code=True,   # <-- REQUIRED or you'll get that error
    verbose=True
)

# 3) Simple chat history setup
chat_history = [
    SystemMessage(content=(
        "You are a helpful data analyst. "
        "when user says about comparison use WIP_BATCH_NO as batch id for comparison."
        "Answer based on the CSV using Python and pandas if needed."
    ))
]

def ask(question: str):
    chat_history.append(HumanMessage(content=question))

    result = agent.invoke({
        "input": question,
        "chat_history": chat_history
    })

    answer = result.get("output", "")
    print("AI:", answer)

    chat_history.append(AIMessage(content=answer))

# 4) Example queries
ask("compare the WIP_BATCH_NO 160600484  and 200500288. and give me its insights. which one is more costly?")
