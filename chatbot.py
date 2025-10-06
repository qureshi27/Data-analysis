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
#############################################################################################################################################

# def chat_with_csv(query: str) -> str:
#     """
#     Takes a query string, runs it against the CSV agent, and returns the answer.
#     """
#     load_dotenv()
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError("OPENAI_API_KEY not found in .env")
#
#     # 1) Initialize LLM
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
#     # 2) Create CSV Agent
#     agent = create_csv_agent(
#         llm,
#         "batch_details.csv",             # <-- change to your CSV path
#         agent_type="openai-tools",
#         allow_dangerous_code=True,
#         verbose=True,
#         pandas_kwargs={"low_memory": False}
#     )
#
#     # 3) Create chat history for context
#     chat_history = [
#         SystemMessage(content=(
#             # "You are a helpful BERGER PAINT data analyst."
#             # "Your job is to provide the insights of data. "
#             # "When the user asks for comparison, use WIP_BATCH_NO as the batch id for comparison."
#             # "Please Note for the comparison of cost. Please make sure that each batch have multiple rows. but the average of cost analysis you must compare based on WIP_RATE which is in  that row  that contain the 'TRANSACTION_TYPE_NAME' as WIP Completion "
#             # "after that you will give detailed analysis based on their ingredients that how cost changed etc or any other factors. "
#             # "Answer based on the CSV using Python and pandas if needed."
#
#             "You are a highly skilled BERGER PAINTS production and costing data analyst. "
#             "Your task is to analyze manufacturing batch data in detail and provide accurate insights. "
#             "Each record in the CSV represents either an ingredient (WIP Issue) or a finished product (WIP Completion) "
#             "within a Work In Process (WIP) batch. "
#             "Use the following logic carefully: "
#             "\n\n"
#             "1️⃣ Use 'WIP_BATCH_NO' as the Batch ID for comparison or identification.\n"
#             "2️⃣ Each batch usually contains multiple rows — one 'WIP Completion' record (the product) and many 'WIP Issue' records (ingredients).\n"
#             "3️⃣ The batch cost comparison must be based **only** on the 'WIP_RATE' from the row where 'TRANSACTION_TYPE_NAME' = 'WIP Completion'. "
#             "That value represents the **unit cost per kg** (or per unit) for that batch.\n"
#             "4️⃣ For deeper cost analysis, examine differences in 'WIP_VALUE', 'WIP_QTY', and ingredient-level rates "
#             "to explain why costs differ between batches (e.g., changes in raw material cost, usage qty, or routing).\n"
#             "5️⃣ If two batches have different 'FORMULA_ID', explain that cost differences may also arise from different formulations or ingredient compositions.\n"
#             "6️⃣ Always summarize clearly which batch is higher/lower cost, give a concise numeric comparison, "
#             "and then explain **why** (e.g., ingredient rate variance, batch size, or scrap factor).\n"
#             "7️⃣ Output your final analysis in a clean, structured markdown format with sections: "
#             "'Overview', 'Cost Comparison', and 'Reason for Cost Difference'.\n"
#             "\n"
#             "Perform all calculations using Python and pandas if necessary, and reason entirely from the CSV data provided."
#         )),
#         HumanMessage(content=query),
#     ]
#
#
#
#
#
#     # 4) Run the agent
#     result = agent.invoke({
#         "input": query,
#         "chat_history": chat_history
#     })
#     print(result)
#     return result.get("output", "")
#
#
# chat_with_csv("give me cost analysis of batch 32876896 and batch 32897946? give me detailed analysis of it?")
############################################################################################################################################################################

import httpx
from pathlib import Path
from phi.agent import Agent
from phi.tools.csv_tools import CsvTools
import os
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
csv_path ="batch_details.csv"



async def query_bot(query: str):

    # Initialize the agent
    agent = Agent(
        tools=[CsvTools(csvs=[csv_path])],
        markdown=False,
        show_tool_calls=False,
        instructions=[
        "you are a berger paint data analysis agent."
        "you will perform detailed analysis based on user query."
        "Please note you are provided with a csv file that contain multiple columns."
        "there are multiple formulas and each formula have multiple batch"
        "for the comparison of different batch use the batch id and that row of batch that have 'TRANSACTION_TYPE_NAME' column value is 'WIP Completion' because it shows the average of rest of the rows of the batch"
        "for the cost analysis you must focus on the 'WIP_RATE' column but row must be that where 'TRANSACTION_TYPE_NAME' column value is 'WIP Completion'"
        "after that you can check other factors differences e,g  ingredient rate variance, batch size, or scrap factor"
        "when you see the difference of 'WIP_RATE' between two batch than must check their ingredients qty and rate for deep analysis."
        "Please provide a complete analysis based on user query. "

        ],
    )

    # Run the query
    response = agent.run(query)
    #print(response.content)
    return response.content

# query_bot("give me cost analysis of batch 33739225 and batch 35385194? give me detailed analysis of it?")