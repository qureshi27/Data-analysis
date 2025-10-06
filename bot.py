# # --- requirements ---
# # pip install pandas langchain langchain-core langchain-experimental langchain-openai
# import os
# from pathlib import Path
# import pandas as pd
#
# from langchain_openai import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_core.messages import SystemMessage, HumanMessage
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#
# # ---------------------------
# # 1) Point to your two Excel files
# # ---------------------------
# FILE_1 = "formula.xlsx"
# FILE_2 = "batch_details.xlsx"
#
# # Optional: if you only want specific sheets, set e.g. SHEETS = {"first.xlsx": ["Sheet1"], "second.xlsx": ["Q2"]}
# SHEETS = None  # or a dict as described above
#
# # ---------------------------
# # 2) Load all (or selected) sheets as distinct DataFrames
# #    Each df gets a descriptive .name like "first.xlsx:Sheet1" to help the agent disambiguate
# # ---------------------------
# def load_excel_as_dfs(path: str, wanted_sheets=None):
#     xls = pd.ExcelFile(path)
#     sheet_names = wanted_sheets if wanted_sheets else xls.sheet_names
#     dfs = []
#     for sheet in sheet_names:
#         df = pd.read_excel(path, sheet_name=sheet)
#         # give the DataFrame a name so the agent can reference it
#         df.name = f"{Path(path).name}:{sheet}"
#         dfs.append(df)
#     return dfs
#
# dfs = []
# dfs.extend(load_excel_as_dfs(FILE_1, (SHEETS or {}).get(Path(FILE_1).name)))
# dfs.extend(load_excel_as_dfs(FILE_2, (SHEETS or {}).get(Path(FILE_2).name)))
#
# if not dfs:
#     raise RuntimeError("No DataFrames were loaded. Check file paths / sheet names.")
#
# # ---------------------------
# # 3) LLM and Agent
# #    - Use any chat model supported by LangChain. Here we show OpenAI's chat wrapper.
# #    - allow_dangerous_code=True enables Python execution via a REPL tool. Use only in a sandbox.
# # ---------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
# agent = create_pandas_dataframe_agent(
#     llm=llm,
#     df=dfs,                         # <-- multiple DataFrames supported
#     agent_type="tool-calling",      # modern, tool-calling style
#     verbose=True,
#     include_df_in_prompt=True,      # small head shown to model for quick context
#     number_of_head_rows=5,
#     allow_dangerous_code=True       # IMPORTANT: only enable in secure/sandboxed environments
# )
#
# # ---------------------------
# # 4) Messages (System + User)
# #    Note: the Pandas agent takes a single text "input"; we compose that from HumanMessage.
# #    The SystemMessage is shown here so you can reuse it in a custom prompt/chain if needed.
# # ---------------------------
# system_message = SystemMessage(
#     content=(
#         "You are a data analyst. Use pandas to join and analyze the provided DataFrames.\n"
#         "Each DataFrame is named like '<filename>:<sheet>'. Be explicit about which tables/columns you use.\n"
#         "Prefer vectorized operations; avoid slow loops. Return concise, well-formatted answers."
#     )
# )
#
# # Example user message (replace with your actual question)
# user_message = HumanMessage(
#     content=(
#         "based on the dataframe can you do the deep analysis of batch having id 160600484 and 200500288? and do compare their cost as well. please also mention does they have same formula or not?"
#     )
# )
#
# # ---------------------------
# # 5) Invoke the agent
# # ---------------------------
# # Simple pattern: pass just the user's text to the agent. If you want the system message to guide behavior,
# # you can prepend it to the input or incorporate it via custom prompts. For most quick analyses, this is enough.
# result = agent.invoke({"input": user_message.content})
# print(result["output"])












































# # --- requirements ---
# # pip install pandas langchain langchain-core langchain-experimental langchain-openai
# import os
# from pathlib import Path
# import pandas as pd
#
# from langchain_openai import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_core.messages import SystemMessage, HumanMessage
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#
# # ---------------------------
# # 1) Point to your two Excel files
# # ---------------------------
# FILE_1 = "formula.xlsx"
# FILE_2 = "batch_details.xlsx"
#
# # Optional: if you only want specific sheets, set e.g. SHEETS = {"first.xlsx": ["Sheet1"], "second.xlsx": ["Q2"]}
# SHEETS = None  # or a dict as described above
#
# # ---------------------------
# # 2) Load all (or selected) sheets as distinct DataFrames
# #    Each df gets a descriptive .name like "first.xlsx:Sheet1" to help the agent disambiguate
# # ---------------------------
# def load_excel_as_dfs(path: str, wanted_sheets=None):
#     xls = pd.ExcelFile(path)
#     sheet_names = wanted_sheets if wanted_sheets else xls.sheet_names
#     dfs = []
#     for sheet in sheet_names:
#         df = pd.read_excel(path, sheet_name=sheet)
#         # give the DataFrame a name so the agent can reference it
#         df.name = f"{Path(path).name}:{sheet}"
#         dfs.append(df)
#     return dfs
#
# dfs = []
# dfs.extend(load_excel_as_dfs(FILE_1, (SHEETS or {}).get(Path(FILE_1).name)))
# dfs.extend(load_excel_as_dfs(FILE_2, (SHEETS or {}).get(Path(FILE_2).name)))
#
# if not dfs:
#     raise RuntimeError("No DataFrames were loaded. Check file paths / sheet names.")
#
# # ---------------------------
# # 3) LLM and Agent
# #    - Use any chat model supported by LangChain. Here we show OpenAI's chat wrapper.
# #    - allow_dangerous_code=True enables Python execution via a REPL tool. Use only in a sandbox.
# # ---------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
# agent = create_pandas_dataframe_agent(
#     llm=llm,
#     df=dfs,                         # <-- multiple DataFrames supported
#     agent_type="tool-calling",      # modern, tool-calling style
#     verbose=True,
#     include_df_in_prompt=True,      # small head shown to model for quick context
#     number_of_head_rows=5,
#     allow_dangerous_code=True       # IMPORTANT: only enable in secure/sandboxed environments
# )
#
# # ---------------------------
# # 4) Messages (System + User)
# #    Note: the Pandas agent takes a single text "input"; we compose that from HumanMessage.
# #    The SystemMessage is shown here so you can reuse it in a custom prompt/chain if needed.
# # ---------------------------
# system_message = SystemMessage(
#     content=(
#         "You are a data analyst. Use pandas to join and analyze the provided DataFrames.\n"
#         "Each DataFrame is named like '<filename>:<sheet>'. Be explicit about which tables/columns you use.\n"
#         "Prefer vectorized operations; avoid slow loops. Return concise, well-formatted answers."
#     )
# )
#
# # Example user message (replace with your actual question)
# user_message = HumanMessage(
#     content=(
#         "can you please tell me based on  batch_details.xl sheet tell me about the analysis of  Batch 160600484 and Batch 200500288 "
#     )
# )
#
# # ---------------------------
# # 5) Invoke the agent
# # ---------------------------
# # Simple pattern: pass just the user's text to the agent. If you want the system message to guide behavior,
# # you can prepend it to the input or incorporate it via custom prompts. For most quick analyses, this is enough.
# result = agent.invoke({"input": user_message.content})
# print(result["output"])











###############################################################################################

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
#from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_experimental.agents import create_csv_agent

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# ---------------------------
# # Excel Configuration
# # ---------------------------
# FILE_1 = "formula.xlsx"
# FILE_2 = "batch_details.xlsx"
# SHEETS = None  # Optional: specify sheets
#
#
# # ---------------------------
# # Load Excel Sheets as Named DataFrames
# # ---------------------------
# def load_excel_as_dfs(path: str, wanted_sheets=None):
#     xls = pd.ExcelFile(path)
#     sheet_names = wanted_sheets if wanted_sheets else xls.sheet_names
#     dfs = []
#     for sheet in sheet_names:
#         df = pd.read_excel(path, sheet_name=sheet)
#         df.name = f"{Path(path).name}:{sheet}"
#         dfs.append(df)
#     return dfs
#
#
# # ---------------------------
# # Load All DataFrames Once (Globally)
# # ---------------------------
# dfs = []
# #dfs.extend(load_excel_as_dfs(FILE_1, (SHEETS or {}).get(Path(FILE_1).name)))
# dfs.extend(load_excel_as_dfs(FILE_2, (SHEETS or {}).get(Path(FILE_2).name)))
#
# if not dfs:
#     raise RuntimeError("No DataFrames loaded from Excel files")
#
# # ---------------------------
# # Initialize LLM + Agent (Once)
# # ---------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
# agent = create_pandas_dataframe_agent(
#     llm=llm,
#     df=dfs,
#     agent_type="tool-calling",
#     verbose=True,
#     include_df_in_prompt=True,
#     number_of_head_rows=5,
#     allow_dangerous_code=True
# )
#
#
# # ---------------------------
# # Analysis Function (Use in FastAPI Route)
# # ---------------------------
# def analyze_excel_query(query: str) -> str:
#     """
#     Analyze the Excel files using a natural language query.
#
#     Parameters:
#         query (str): The user's question about the Excel data.
#
#     Returns:
#         str: The agent's analysis output.
#     """
#     try:
#         # Optional: Inject system message if needed — shown here but not required for agent.invoke
#         system_message = SystemMessage(
#             content=(
#                 "You are a data analyst. Use pandas to join and analyze the provided DataFrames.\n"
#                 "Each DataFrame is named like '<filename>:<sheet>'. Be explicit about which tables/columns you use.\n"
#                 "Prefer vectorized operations; avoid slow loops. Return concise, well-formatted answers."
#             )
#         )
#
#         human_message = HumanMessage(content=query)
#         result = agent.invoke({"input": human_message.content})
#         return result["output"]
#
#     except Exception as e:
#         return f"Error: {str(e)}"


def chat_with_csv(query: str) -> str:
    """
    Takes a query string, runs it against the CSV agent, and returns the answer.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    # 1) Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 2) Create CSV Agent
    agent = create_csv_agent(
        llm,
        "batch_details.csv",             # <-- change to your CSV path
        agent_type="openai-tools",
        allow_dangerous_code=True,
        verbose=True,
        pandas_kwargs={"low_memory": False}
    )

    # 3) Create chat history for context
    chat_history = [
        SystemMessage(content=(
"""
You are an expert analyst for manufacturing batch data, especially focused on WIP (Work-In-Progress) cost and performance analysis.

You will receive a dataset containing multiple rows per batch. Each batch is identified by a WIP_BATCH_ID.

Each batch contains:

A summary row where TRANSACTION_TYPE_NAME = 'WIP Completion' — this row reflects the total WIP value and average WIP_RATE for the entire batch.

Multiple other rows for ingredients, issues, returns, etc.

Your primary analysis should always:

Focus only on rows where TRANSACTION_TYPE_NAME = 'WIP Completion'.

Treat WIP_RATE in that row as the true average for the batch.

Identify pairs of batches with the largest difference in WIP_RATE.

Provide a clear breakdown explaining why the difference exists — considering:

Cost of ingredients (from WIP_RATE in other rows)

ROUTING_ID differences

Dates (to infer cost inflation or process changes)

Any unusual values like extreme WIP_RATEs or WIP_VALUEs

Always return results in a well-structured format, bullet points, and clear reasoning.
Whatever the user ask do deep analysis of the input and than proceed accordingly.

"""
        )),
        HumanMessage(content=query),
    ]

    # 4) Run the agent
    result = agent.invoke({
        "input": query,
        "chat_history": chat_history
    })
    print(result)
    return result.get("output", "")


chat_with_csv("give me two batch that have different formula and very high difference of WIP_RATE. along this give me its analysis why there is high difference in WIP_RATE?")