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
csv_path =r"D:\work\Data-analysis\batch_details.csv"



def query_bot(query: str):

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
    print(response.content)
    return response.content

query_bot("give me cost analysis of batch 33739225 and batch 35385194? give me detailed analysis of it?")