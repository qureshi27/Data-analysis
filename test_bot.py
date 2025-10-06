from openai import OpenAI
import os
from dotenv import load_dotenv
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

CSV_PATH = r"batch_details.csv"  # <-- your CSV path

import time
# 1) Upload your CSV for assistant use
file = client.files.create(
    file=open(CSV_PATH, "rb"),
    purpose="assistants"
)

# 2) Create an Assistant with Code Interpreter and attach the file
assistant = client.beta.assistants.create(
    name="WIP Batch Analyst",
    instructions=(
"""
You are an expert analyst for manufacturing batch data, especially focused on WIP (Work-In-Progress) cost and performance analysis.

You will receive a dataset containing multiple rows per batch. Each batch is identified by a WIP_BATCH_ID.

Each batch contains:

A summary row where TRANSACTION_TYPE_NAME = 'WIP Completion' — this row reflects the total WIP value and average WIP_RATE for the entire batch.

Multiple other rows for ingredients, issues, returns, etc.

Your primary analysis should always:

Focus only on rows where TRANSACTION_TYPE_NAME = 'WIP Completion'.

Treat WIP_RATE in that row as the true average for the batch.

When comparing batches, compare as per user query.

Identify pairs of batches with the largest difference in WIP_RATE.

Provide a clear breakdown explaining why the difference exists — considering:

Cost of ingredients (from WIP_RATE in other rows)

ROUTING_ID differences

Dates (to infer cost inflation or process changes)

Any unusual values like extreme WIP_RATEs or WIP_VALUEs

Always return results in a well-structured format with tables, bullet points, and clear reasoning.
"""
    ),
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}],
    tool_resources={
        "code_interpreter": {
            "file_ids": [file.id]   # ✅ attach CSV here
        }
    }
)

# 3) Create a thread for conversation
thread = client.beta.threads.create()

# 4) Add a user message
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Give me two batches with different FORMULA_IDs and the highest difference in WIP_RATE. Also explain why."
)

# 5) Run the assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# 6) Wait until the run is completed
while True:
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    if run_status.status == "completed":
        break
    elif run_status.status == "failed":
        raise Exception("Run failed")
    time.sleep(1)

# 7) Get and print the assistant’s final messages
messages = client.beta.threads.messages.list(thread_id=thread.id)

for msg in messages.data[::-1]:  # reverse to get assistant reply last
    if msg.role == "assistant":
        for content in msg.content:
            if content.type == "text":
                print(content.text.value)
