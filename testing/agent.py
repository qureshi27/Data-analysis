from click import prompt
from openai import OpenAI
import os
import time
import platform
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


import os
import time
import base64

def csv_agent(user_prompt: str) -> dict:
    # Fixed CSV file path
    csv_path = r"D:\work\Data-analysis\batch_details.csv"

    if not os.path.exists(csv_path):
        return {"error": f"File not found at path: {csv_path}"}

    # Step 1: Upload CSV
    file = client.files.create(
        file=open(csv_path, "rb"),
        purpose="assistants"
    )

    # Step 2: Create Assistant
    assistant = client.beta.assistants.create(
        name="Data Analyst",
        description="""
 "You are a BERGER PAINTS data analyst. Analyze manufacturing batch data. "
        "Each batch has multiple rows. Use rows with 'WIP Completion' in 'TRANSACTION_TYPE_NAME' "
        "to extract cost from 'WIP_RATE'. For comparisons, use only these rows. "
        "When analyzing cost differences, check ingredient rates, quantities, and formulations. "
        "Structure results as: Overview, Cost Comparison, Reason for Cost Difference."
        
        """,
        model="gpt-4o",
        tools=[{"type": "code_interpreter"}],
        tool_resources={"code_interpreter": {"file_ids": [file.id]}}
    )

    # Step 3: Create Thread with prompt
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": user_prompt,
            "attachments": [
                {
                    "file_id": file.id,
                    "tools": [{"type": "code_interpreter"}]
                }
            ]
        }]
    )

    # Step 4: Start the assistant run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Step 5: Poll until run is complete
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status in ["failed", "cancelled"]:
            return {"error": f"Run failed: {run_status.status}"}
        time.sleep(1)

    # Step 6: Retrieve and format assistant response
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")

    response = {"analysis": ""}

    for message in messages.data:
        if message.role == "assistant":
            for content in message.content:
                if content.type == "text":
                    response["analysis"] += content.text.value + "\n\n"

    print("here is response",response)
    return response



csv_agent("give me cost analysis of batch 32876896 and batch 32897946? give me detailed analysis of it?")