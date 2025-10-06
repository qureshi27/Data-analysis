from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
from datetime import timedelta
import os
import pandas as pd
import numpy as np
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
import io
import base64
import contextlib
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import math
from fastapi.encoders import jsonable_encoder

# main.py

app = FastAPI(
    title="Manufacturing Analytics API",
    version="3.0",
    servers=[{"url": "/"}],   # <= use relative base
)

# If your function is served under /api (common on Vercel):
# app = FastAPI(title="...", version="3.0", servers=[{"url": "/api"}])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod; set specific origins if using cookies
    allow_credentials=False,  # "*" + credentials is invalid; set to False unless needed
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)
# Load and preprocess data
df = pd.read_excel("batch_details.xlsx")
df["WIP_ACT_START_DATE"] = pd.to_datetime(df["WIP_ACT_START_DATE"])
df["WIP_CMPLT_DATE"] = pd.to_datetime(df["WIP_CMPLT_DATE"])

batch_processing = (
    df.groupby("WIP_BATCH_ID")
      .agg({"WIP_ACT_START_DATE": "min", "WIP_CMPLT_DATE": "max"})
      .reset_index()
)
batch_processing["processing_days"] = (
    (batch_processing["WIP_CMPLT_DATE"] - batch_processing["WIP_ACT_START_DATE"]).dt.days
)

# API endpoint (NO inputs, fixed for your chart)
@app.get("/processing-days-histogram")
def get_histogram():
    # Fixed bins (30 like your matplotlib code)
    counts, bin_edges = np.histogram(batch_processing["processing_days"], bins=30)

    return JSONResponse(content={
        "raw_processing_days": batch_processing["processing_days"].tolist(),  # all values
        "counts": counts.tolist(),          # histogram counts (y-axis)
        "bin_edges": bin_edges.tolist(),    # histogram bin edges (x-axis)
        "threshold": 2 ,
         "ai_insights":"""
         # What this chart shows
- A **histogram** of batch-level processing times (`processing_days`).
- The **x-axis** shows how many days each batch took to complete.
- The **y-axis** shows the number of batches falling in each time range.
- The **red dashed line at 2 days** marks the **delay threshold**.

# Insights you can derive
1. **Most batches are fast**
   - The bulk of batches finish within **0–2 days**, clustering left of the threshold line.
   - This indicates the process is generally efficient for a majority of runs.

2. **Significant delayed tail**
   - There is a **long tail to the right**, with some batches taking much longer (5–10+ days, even beyond 20 days).
   - These extended outliers suggest **specific bottlenecks or exceptional cases** that require deeper investigation.

3. **Delay threshold exceedances**
   - A notable number of batches cross the **2-day delay threshold**, visible as bars to the right of the red line.
   - These represent the **share of batches at risk** for customer service or operational performance metrics.

4. **Operational variability**
   - The spread of the distribution shows that while most processes are tightly controlled, there’s **variability across certain batches**.
   - Identifying root causes (equipment issues, material shortages, product type differences) can reduce this variability.

         """
    })

# API endpoint for delayed vs on-time share
@app.get("/delay-share")
def get_delay_share():
    threshold_days = 2  # fixed threshold for delay
    batch_processing["is_delayed"] = batch_processing["processing_days"] > threshold_days

    delay_counts = batch_processing["is_delayed"].value_counts(normalize=True) * 100

    return JSONResponse(content={
        "categories": ["On Time", "Delayed"],
        "percentages": [
            delay_counts.get(False, 0),  # On Time %
            delay_counts.get(True, 0)    # Delayed %
        ],
        "threshold_days": threshold_days,
        "ai_insights": """
        # What this chart shows
- A **bar chart** comparing the percentage of **on-time vs delayed batches**.
- About **74% of batches finish within the 2-day threshold** (on time).
- Around **26% of batches exceed the 2-day threshold** (delayed).

# Insights you can derive
1. **Overall performance**
   - The majority of batches are completed on time, showing that the process is generally reliable.
   - However, with **1 in 4 batches delayed**, delays are not rare and could impact production flow and delivery schedules.

2. **Room for improvement**
   - Reducing the delayed portion even by a few percentage points could yield major improvements in throughput, capacity utilization, and customer satisfaction.

3. **Business impact**
   - If delayed batches involve high-value products or critical customer orders, the **real-world impact is larger than the percentage suggests**.
   - Understanding which formulas or lines contribute most to delays will help prioritize improvement efforts.

# Suggested next steps
- **Break down delay rates by line, formula, or product family** to identify where the 26% delays originate.
- **Quantify financial impact** by linking delayed batches to WIP value and lost opportunity.
- **Investigate recurring causes** (material shortages, equipment downtime, planning issues) for delayed batches.
- **Set improvement targets**, e.g., reduce delays from 26% → 15% over the next quarter.

        """
    })
# API endpoint for monthly average processing days
@app.get("/monthly-average-delay")
def get_monthly_average_delay():
    # Extract month from start date
    batch_processing["month"] = batch_processing["WIP_ACT_START_DATE"].dt.to_period("M")

    # Monthly average processing days
    monthly_delay = (
        batch_processing.groupby("month")["processing_days"]
        .mean()
        .reset_index()
    )

    # Convert Period to Timestamp (string for JSON)
    monthly_delay["month"] = monthly_delay["month"].dt.to_timestamp()

    return JSONResponse(content={
        "months": monthly_delay["month"].dt.strftime("%Y-%m").tolist(),  # e.g., "2024-01"
        "avg_processing_days": monthly_delay["processing_days"].tolist(), # y-axis values
        "threshold": 2 , # delay threshold
        "ai_insights": """

        # What this chart shows
- A **time-series line chart** of the **average batch processing days per month**.
- A **red dashed line marks the 2-day threshold** (delay benchmark).
- Early months mostly stayed **below or near the threshold**.
- In later months, the **average processing time increases sharply**, with several months exceeding **5–10 days on average**.

# Insights you can derive
1. **Early stability, later deterioration**
   - Initially, batch processing was controlled and consistently **under 2 days** on average.
   - Over time, processing days **spiked significantly**, showing a **deterioration in performance**.

2. **Clear upward trend**
   - From the mid-point of the timeline, averages began creeping upward, suggesting **systematic delays** (e.g., demand surge, capacity bottlenecks, resource shortages).
   - The peaks reaching **10–15+ days** highlight **severe operational inefficiencies** in certain months.

3. **Threshold breaches are frequent in later periods**
   - In the first half, breaches of the 2-day delay threshold were rare.
   - In the second half, **delays became the norm rather than the exception**.

# Suggested next steps
- **Root cause analysis by time period**: Identify what changed during the months when delays started trending upward (e.g., seasonal demand, machine breakdowns, supplier issues).
- **Correlate with production volumes**: Check whether spikes coincide with high WIP loads or new product launches.
- **Operational interventions**:
  - Add capacity or shifts during peak months.
  - Rebalance workloads across lines.
  - Improve preventive maintenance to avoid bottlenecks.
- **Set monitoring alerts**: Flag when average monthly processing days exceed **2–3 days**, so corrective actions can be taken early.
        """
    })


# API endpoint for average processing days by line
@app.get("/line-average-delay")
def get_line_average_delay():
    # Calculate processing_days if not already in df
    df["processing_days"] = (df["WIP_CMPLT_DATE"] - df["WIP_ACT_START_DATE"]).dt.days

    # Group by line to compute average processing days
    delay_by_line = df.groupby("LINE_NO")["processing_days"].mean().reset_index()

    return JSONResponse(content={
        "lines": delay_by_line["LINE_NO"].astype(str).tolist(),       # x-axis labels
        "avg_processing_days": delay_by_line["processing_days"].tolist(),  # y-axis values
        "threshold": 2,
        "ai_insights": """
        # What this chart shows
- A **bar chart of average processing days by production line**.
- A **red dashed line at 2 days** marks the threshold for delays.
- Most lines hover around **~2 days or below**, staying close to or under the benchmark.
- However, a few lines (notably **Line 24 and Line 25**) have **very high averages (4–5+ days)**, standing out as clear bottlenecks.

# Insights you can derive
1. **Overall performance is stable for most lines**
   - Lines 1–22 are **well within control**, averaging near or below the 2-day threshold.
   - These lines show **balanced efficiency** with minimal variation.

2. **Critical bottlenecks**
   - **Line 24 and Line 25** are major outliers with averages **double or more** the acceptable limit.
   - These lines are the **biggest contributors to system-wide delays**.

3. **Best performing lines**
   - Lines 21–23 average **well below 2 days**, even under 1 day in some cases.
   - These can serve as **benchmarks for best practices** that may be replicated elsewhere.

# Suggested next steps
- **Deep-dive into Line 24 & 25**:
  - Check for capacity constraints, equipment issues, or staffing shortages.
  - Investigate if product mix or complexity on these lines is higher.

- **Benchmark against top performers (Lines 21–23)**:
  - Analyze what operational strategies, scheduling, or resourcing helps them stay efficient.

- **Balance workloads**:
  - If possible, redistribute high-load batches from Lines 24–25 to underutilized lines.

- **Continuous monitoring**:
  - Regularly track average processing days per line to quickly detect new bottlenecks.
        """
    })


# API endpoint for monthly average processing days by line
@app.get("/line-monthly-average-delay")
def get_line_monthly_average_delay():
    # Batch-level processing days per line
    batch_processing = (
        df.groupby(["WIP_BATCH_ID", "LINE_NO"])
          .agg({"WIP_ACT_START_DATE": "min", "WIP_CMPLT_DATE": "max"})
          .reset_index()
    )
    batch_processing["processing_days"] = (
        (batch_processing["WIP_CMPLT_DATE"] - batch_processing["WIP_ACT_START_DATE"]).dt.days
    )
    batch_processing["month"] = batch_processing["WIP_ACT_START_DATE"].dt.to_period("M")

    # Group by line & month
    avg_delay = (
        batch_processing.groupby(["month", "LINE_NO"])["processing_days"]
        .mean()
        .reset_index()
    )

    # Convert month Period -> Timestamp -> String
    avg_delay["month"] = avg_delay["month"].dt.to_timestamp()
    avg_delay["month"] = avg_delay["month"].dt.strftime("%Y-%m")

    # Pivot to create structure: line_no -> list of avg values aligned with months
    pivoted = avg_delay.pivot(index="month", columns="LINE_NO", values="processing_days").fillna(0)

    return JSONResponse(content={
        "months": pivoted.index.tolist(),
        "lines": {str(col): pivoted[col].tolist() for col in pivoted.columns},
        "threshold": 2,
        "ai_insights": """

        # What this chart shows
- A **monthly trend of average processing days per line** across all 26 lines.
- The **red dashed line at 2 days** is the delay threshold.
- Each colored line represents one production line, with fluctuations in average processing days over time.

# Key observations
1. **Overall variability across months**
   - Most lines operate **close to or below 2 days** in many months, but several spikes occur periodically.
   - Indicates **occasional bottlenecks** rather than consistent systemic delays.

2. **Severe outliers**
   - Some months show extreme peaks:
     - One line (possibly **Line 24**) spiked above **30 days**.
     - Another spike around **20 days** occurred in a different line (likely Line 25 or 13).
   - These peaks dominate the delay pattern and should be investigated.

3. **Recent upward trend**
   - Toward later months, several lines (e.g., **Line 1, Line 2, Line 13**) are **consistently above 2 days**.
   - Suggests a **gradual worsening trend** across multiple lines.

4. **Stable performers**
   - Some lines remain **flat and consistently under 2 days** across months (e.g., Lines 6, 7, 10, 15, 18).
   - These represent **best practices** and process stability.

# Insights & recommendations
- **Investigate spikes (Line 24 & 25):**
  - Likely due to **major disruptions** (machine breakdowns, manpower shortage, or large complex batches).
  - Need root-cause analysis for those extreme delays.

- **Monitor emerging trends (Lines 1, 2, 13):**
  - Gradual creep above threshold signals **capacity stress**.
  - Address before they become chronic bottlenecks.

- **Learn from stable lines (6, 7, 15, 18):**
  - Capture **process discipline, scheduling, or resource allocation strategies** keeping them below threshold.
  - Use as benchmarks.

- **Consider rolling average visualization:**
  - A **3-month rolling average** would smooth out extreme spikes and reveal more stable trends.

# Next step suggestion
Would you like me to prepare a **heatmap (line vs. month with color = avg delay)**?
That would make spotting problematic months & lines much clearer than overlapping line plots.
        """
    })

# API endpoint for delayed batches per line
@app.get("/delayed-batches-by-line")
def get_delayed_batches_by_line():
    # Step 1: Compute batch-level processing_days
    batch_processing = (
        df.groupby(["WIP_BATCH_ID", "LINE_NO"])
          .agg({"WIP_ACT_START_DATE": "min", "WIP_CMPLT_DATE": "max"})
          .reset_index()
    )
    batch_processing["processing_days"] = (
        (batch_processing["WIP_CMPLT_DATE"] - batch_processing["WIP_ACT_START_DATE"]).dt.days
    )

    # Step 2: Mark delayed batches
    batch_processing["is_delayed"] = batch_processing["processing_days"] > 2

    # Step 3: Count delayed batches per line
    delayed_by_line = (
        batch_processing[batch_processing["is_delayed"]]
        .groupby("LINE_NO")
        .size()
        .reset_index(name="delayed_batches")
        .sort_values("delayed_batches", ascending=False)
    )

    return JSONResponse(content={
        "lines": delayed_by_line["LINE_NO"].astype(str).tolist(),        # x-axis
        "delayed_batches": delayed_by_line["delayed_batches"].tolist(),
        "ai_insights": """
        # What this chart shows
- The **number of delayed batches** (processing time > 2 days) per process line.
- Each bar represents a line, ranked from most to least delayed batches.

# Key observations
1. **Critical lines with highest delays**
   - Lines **1 to 10** consistently show **very high delays (around 1,500 delayed batches each)**.
   - These lines represent the **core bottlenecks** in the production system.

2. **Moderate problem lines**
   - Lines **11 to 19** show **800–1,000 delayed batches each**.
   - These are secondary contributors to overall delays.

3. **Low-delay lines**
   - Lines **20 to 23** show **few hundred delayed batches or less**.
   - Line 23 and 24 are **almost negligible contributors**, indicating either low volume or highly efficient processes.
   - Line 25 has **zero delayed batches**, making it the best performer.

# Insights & recommendations
- **Prioritize improvement efforts on Lines 1–10**
  - They are responsible for the majority of delays and will give the **biggest impact if optimized**.
  - Possible issues: capacity overload, frequent breakdowns, scheduling inefficiencies.

- **Focus secondary attention on Lines 11–19**
  - Moderate level of delays, worth monitoring and addressing after the top 10 lines are stabilized.

- **Study best practices from Lines 23–25**
  - Very low or zero delays → investigate **why they are so efficient** (lower workload, better resource management, or less complex products?).
  - Apply learnings to high-delay lines.

- **80/20 rule applies**: The top 10 lines (1–10) are likely contributing to **over 70% of total delays**.
- Improvements in these critical lines can drastically reduce system-wide production delays.
- A deeper drilldown (batch size, product type, resource availability per line) would help in root cause analysis.
        """
    })

# API endpoint for delayed vs total batches per line
@app.get("/delayed-vs-total-batches")
def get_delayed_vs_total_batches():
    # Step 1: Compute batch-level processing_days
    batch_processing = (
        df.groupby(["WIP_BATCH_ID", "LINE_NO"])
          .agg({"WIP_ACT_START_DATE": "min", "WIP_CMPLT_DATE": "max"})
          .reset_index()
    )
    batch_processing["processing_days"] = (
        (batch_processing["WIP_CMPLT_DATE"] - batch_processing["WIP_ACT_START_DATE"]).dt.days
    )
    batch_processing["is_delayed"] = batch_processing["processing_days"] > 2

    # Step 2: Aggregate per line
    line_stats = batch_processing.groupby("LINE_NO").agg(
        total_batches=("WIP_BATCH_ID", "count"),
        delayed_batches=("is_delayed", "sum")
    ).reset_index()

    # On-time = total - delayed
    line_stats["on_time_batches"] = line_stats["total_batches"] - line_stats["delayed_batches"]

    # Sort by total workload (largest first)
    line_stats = line_stats.sort_values("total_batches", ascending=False)

    return JSONResponse(content={
        "lines": line_stats["LINE_NO"].astype(str).tolist(),
        "total_batches": line_stats["total_batches"].tolist(),
        "delayed_batches": line_stats["delayed_batches"].tolist(),
        "on_time_batches": line_stats["on_time_batches"].tolist(),

        "ai_insights": """
        # What this chart shows
- **Total workload (batches)** per process line, split into:
  - **On Time batches** (light gray)
  - **Delayed batches** (blue, > 2 processing days)
- Lines are sorted by workload (highest total batches on the left).

# Key observations
1. **High-workload lines (1–10)**
   - Each handles ~5,800 batches, the **largest share of total production**.
   - Despite high volumes, a **large chunk (blue) is delayed**.
   - Indicates **capacity strain** or **systematic inefficiencies**.

2. **Medium-workload lines (11–19)**
   - Handle ~2,500–3,500 batches each.
   - Proportion of delayed batches remains **significant (~25–30%)**, but absolute delays are fewer compared to top 10 lines.

3. **Low-workload lines (20–25)**
   - Much smaller total volumes.
   - Some still show delays (e.g., line 20), while others (23–25) are mostly delay-free.
   - Suggests that **delays are not purely volume-driven** — process or resource issues may exist.

# Insights & recommendations
- **Critical pressure points: Lines 1–10**
  - They process the majority of batches and carry the **heaviest absolute delays**.
  - Improving efficiency here will have the **greatest system-wide impact**.

- **Balanced focus on throughput and quality**
  - While some delays may be expected in high-volume lines, the **delayed fraction is disproportionately high**, suggesting structural bottlenecks (machine downtime, labor capacity, scheduling).

- **Learnings from low-delay, low-volume lines (23–25)**
  - These lines run with minimal delays.
  - Investigating their **processes, product types, or resource allocation** could yield transferable improvements for higher-load lines.

# Conclusion
- The system follows a **Pareto distribution**: the top 10 lines account for most production and most delays.
- Optimizing these lines would yield the largest benefit.
- However, since delays also exist in medium/low-volume lines, **root cause analysis should go beyond workload** and check operational practices, resource constraints, and product complexity.

        """
    })

# API endpoint for top 15 formulas by delay rate
@app.get("/top-delay-formulas")
def get_top_delay_formulas():
    # --- Compute batch-level processing_days ---
    batch_processing = (
        df.groupby(["WIP_BATCH_ID", "FORMULA_ID"])
          .agg({"WIP_ACT_START_DATE": "min", "WIP_CMPLT_DATE": "max"})
          .reset_index()
    )
    batch_processing["processing_days"] = (
        (batch_processing["WIP_CMPLT_DATE"] - batch_processing["WIP_ACT_START_DATE"]).dt.days
    )
    batch_processing["is_delayed"] = batch_processing["processing_days"] > 2

    # --- Aggregate by formula: total & delayed ---
    delay_by_formula = batch_processing.groupby("FORMULA_ID").agg(
        total_batches=("WIP_BATCH_ID", "count"),
        delayed_batches=("is_delayed", "sum")
    ).reset_index()

    # --- Compute delay rate (%) ---
    delay_by_formula["delay_rate"] = (
        (delay_by_formula["delayed_batches"] / delay_by_formula["total_batches"]) * 100
    )

    # --- Top 15 formulas ---
    top_delay_formulas = delay_by_formula.sort_values("delay_rate", ascending=False).head(15)

    return JSONResponse(content={
        "formula_ids": top_delay_formulas["FORMULA_ID"].astype(str).tolist(),
        "delay_rates": top_delay_formulas["delay_rate"].round(2).tolist(),
        "ai_insights": """
        # What this chart shows
- The chart compares the **average scrap factor per production line**.
- Scrap factor indicates the proportion of material wasted (scrap) during production.
- Each bar corresponds to a **Line No**, with its respective average scrap factor.

# Key observations
1. **Most lines are clustered around ~0.03 (3%) scrap factor**
   - This indicates a relatively consistent performance across the majority of lines.

2. **Line 1 shows the lowest scrap factor (~0.018 / 1.8%)**
   - This suggests Line 1 is operating more efficiently, with less material waste compared to others.
   - Could be due to better machine calibration, newer equipment, or skilled operators.

3. **Lines 2, 13, 21, and 23 show slightly lower scrap rates (~2.5–2.8%)** compared to the ~3% benchmark.
   - These may be secondary efficient performers.

4. **No line shows excessively high scrap rates** (all are within a narrow range around 3%).
   - This suggests scrap is a systemic baseline issue rather than isolated to one problematic line.

# Insights & recommendations
- **Benchmark Line 1 practices**
  - Investigate why Line 1 has significantly lower scrap.
  - Replicate best practices (e.g., preventive maintenance, operator skill, material handling) across other lines.

- **Focus on small improvements across all lines**
  - Since most lines are near 3%, a **0.5% reduction plant-wide** could yield significant savings in material costs.

- **Check for systemic causes**
  - The uniformity of scrap factors indicates a **common process or formula-driven scrap rate**, rather than line-specific defects.
  - This means looking into **recipe design, raw material variability, or production setup standards** might be more impactful.

# Conclusion
- Scrap rates are generally stable but consistently around ~3%.
- Line 1 stands out as a model of efficiency (~40% lower scrap vs. average).
- By studying Line 1’s practices and applying them plant-wide, overall scrap can be reduced significantly
        """
    })

# API endpoint for average scrap factor per line


# API endpoint for monthly delay rate
@app.get("/monthly-delay-rate")
def get_monthly_delay_rate():
    # Compute batch-level processing days
    batch_processing = (
        df.groupby("WIP_BATCH_ID")
          .agg({"WIP_ACT_START_DATE": "min", "WIP_CMPLT_DATE": "max"})
          .reset_index()
    )
    batch_processing["processing_days"] = (
        (batch_processing["WIP_CMPLT_DATE"] - batch_processing["WIP_ACT_START_DATE"]).dt.days
    )

    # Extract month
    batch_processing["month"] = batch_processing["WIP_ACT_START_DATE"].dt.to_period("M")

    # Monthly delay stats
    delay_by_month = (
        batch_processing.assign(is_delayed=batch_processing["processing_days"] > 2)
        .groupby("month")
        .agg(
            total_batches=("WIP_BATCH_ID", "count"),
            delayed_batches=("is_delayed", "sum")
        )
        .reset_index()
    )
    delay_by_month["delay_rate"] = (
        delay_by_month["delayed_batches"] / delay_by_month["total_batches"] * 100
    )

    # Convert Period → Timestamp → string
    delay_by_month["month"] = delay_by_month["month"].dt.to_timestamp()
    delay_by_month["month"] = delay_by_month["month"].dt.strftime("%Y-%m")

    return JSONResponse(content={
        "months": delay_by_month["month"].tolist(),
        "delay_rates": delay_by_month["delay_rate"].round(2).tolist(),
        "threshold": 50,
        "ai_insights": """

# ⏱️ Monthly Delay Rate (%) – Analysis

### What the chart shows
- This line chart tracks the **delay rate (%) by month**.
- The dashed gray line at 50% is a **reference threshold** for acceptable delay levels.
- Red markers highlight the actual monthly delay performance.

---

### 🔑 Key Observations
1. **Extremely high volatility**
   - Delay rates fluctuate sharply month-to-month, often swinging from near zero to over **1000%+**.
   - Indicates unstable processes or external disruptions.

2. **Early period (left side)**
   - Several **spikes above 1200% delay rate**, followed by a gradual decline.
   - Suggests initial instability before some corrective measures.

3. **Mid-period (center of the chart)**
   - Delay rates are relatively **low and stable**, often hovering near or below the 50% threshold.
   - This was the **best performing phase**.

4. **Recent period (right side)**
   - Sustained **high delays (800%–1500%)** with sharp month-to-month swings.
   - Suggests recurrence of systemic problems, possibly capacity constraints, supply chain issues, or workforce inefficiencies.

---

### 💡 Insights & Recommendations
- **Investigate root causes of spikes**
  - Look into months with extreme delays (>1000%). These may align with **material shortages, machine breakdowns, or peak demand surges**.

- **Replicate mid-period stability**
  - The stable months (near/below 50%) should be studied as benchmarks — what processes worked then that are missing now?

- **Recent performance is concerning**
  - Sustained high delays suggest **systemic inefficiencies have returned**.
  - Requires urgent corrective action to avoid recurring customer dissatisfaction and financial losses.

- **Forecasting & resource planning**
  - Volatility suggests delays may not be random. Using **seasonality analysis** could help anticipate spikes and plan resources accordingly.

        """
    })


# API endpoint for average scrap factor per line
@app.get("/line-scrap-factor")
def get_line_scrap_factor():
    # Group by line to compute mean scrap factor
    line_scrap = df.groupby("LINE_NO")["SCRAP_FACTOR"].mean().reset_index()

    return JSONResponse(content={
        "lines": line_scrap["LINE_NO"].astype(str).tolist(),
        "avg_scrap_factor": line_scrap["SCRAP_FACTOR"].round(4).tolist(),
    "ai_insights": """
    # 🚨 Delay Reasons by Line – Analysis

### What the chart shows
- This stacked bar chart shows **delayed batch counts per line**, broken down by different **delay reasons**.
- The legend categorizes causes:
  - **Major:** Addition/deletion for Batch WIP, Capacity Constraints, RM Short, ERP/WIP Errors.
  - **Minor but recurring:** CR.LOW, HOLD BY SC, Holidays, Supply Chain instructions, Viscosity Variation.

---

### 🔑 Key Observations
1. **Line 1 is the biggest bottleneck**
   - Extremely high delays (~850+ counts), far above all other lines.
   - Mostly driven by **“Addition and deletion for Batch WIP”**.

2. **Lines 2–11 have consistent but moderate delays**
   - Each shows **~250–300 delayed batches**, again dominated by Batch WIP changes.
   - Secondary reasons (capacity constraints, RM short, ERP/WIP error) are present but comparatively minor.

3. **Lines 12–14 are nearly clean**
   - Very few delays logged, suggesting either **lower load or more efficient processes**.

4. **Root cause dominance**
   - Across all lines, **Batch WIP adjustments** are the overwhelming root cause.
   - Other categories (capacity, raw material shortage, ERP/WIP error) remain small contributors.

---

### 💡 Insights & Recommendations
- **Immediate focus: Line 1**
  - Investigate **Batch WIP process design** – why does Line 1 face disproportionate rework?
  - Possible causes: scheduling conflicts, incorrect batch planning, operator interventions.

- **Standardize WIP handling across lines**
  - Since Batch WIP is the dominant reason everywhere, a **cross-line process correction** could reduce delays significantly.

- **Preventive measures for secondary causes**
  - Build stronger **capacity buffers** (machine/operator availability).
  - Strengthen **raw material planning** to reduce RM shortages.
  - Audit **ERP/WIP data accuracy** to minimize system-driven delays.

- **Learn from Lines 12–14**
  - Study practices here (lower volumes? better planning? different operators?) and replicate to Lines 1–11.

    """

    })

# API endpoint: Monthly Delay Rate (%)


# 📌 Delay reasons by line
@app.get("/delay-reasons-by-line")
def get_delay_reasons_by_line():
    local_df = df.copy()

    # Ensure processing_days column exists
    if "processing_days" not in local_df.columns:
        local_df["processing_days"] = (
            (local_df["WIP_CMPLT_DATE"] - local_df["WIP_ACT_START_DATE"]).dt.days
        )

    # Filter delayed batches
    line_reason = (
        local_df[local_df["processing_days"] > 2]
        .dropna(subset=["REASON"])
        .groupby(["LINE_NO", "REASON"])
        .size()
        .reset_index(name="count")
    )

    # Convert to structured JSON
    result = {}
    for _, row in line_reason.iterrows():
        line = str(row["LINE_NO"])
        reason = row["REASON"]
        count = int(row["count"])
        if line not in result:
            result[line] = {}
        result[line][reason] = count

    return JSONResponse(content={
        "delay_reasons_by_line": result,
        "threshold_days": 2
    })


@app.get("/delay-reasons-top10")
def get_top_delay_reasons():
    if "processing_days" not in df.columns:
        df["processing_days"] = (df["WIP_CMPLT_DATE"] - df["WIP_ACT_START_DATE"]).dt.days

    delayed = df[df["processing_days"] > 2].dropna(subset=["REASON"])  # fixed threshold = 2

    delay_reasons = (
        delayed.groupby("REASON")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    total_delayed = delay_reasons["count"].sum()
    delay_reasons["share_percent"] = (delay_reasons["count"] / total_delayed * 100).round(2)

    return {
        "top_delay_reasons": delay_reasons.to_dict(orient="records"),
        "threshold_days": 2
    }


from datetime import timedelta
df["WIP_ACT_START_DATE"] = pd.to_datetime(df["WIP_ACT_START_DATE"], errors="coerce")
df["WIP_CMPLT_DATE"]     = pd.to_datetime(df["WIP_CMPLT_DATE"], errors="coerce")

# Per-batch table
batches = (
    df.groupby("WIP_BATCH_ID")
      .agg(start=("WIP_ACT_START_DATE","min"),
           end=("WIP_CMPLT_DATE","max"))
      .reset_index()
)

# Processing time (days) per batch
batches["processing_days"] = (batches["end"] - batches["start"]).dt.total_seconds() / 86400

# Helper: latest month bounds (based on latest date present, start or end)
latest_date = pd.to_datetime(
    max(batches["end"].max(), batches["start"].max())
).normalize()
CUR_START = latest_date.replace(day=1)
CUR_END   = (CUR_START + pd.offsets.MonthBegin(1)) - timedelta(days=1)

# Monthly average processing (for 3-month rolling)
batches["end_month"] = batches["end"].dt.to_period("M")
avg_by_month = (
    batches.dropna(subset=["end_month"])
           .groupby("end_month")["processing_days"]
           .mean()
           .sort_index()
)

def rolling_3mo_for(month_period):
    # last three months including the given month
    if month_period is None or avg_by_month.empty:
        return 0.0
    idx = avg_by_month.index.sort_values()
    if month_period not in idx:
        return float(avg_by_month.tail(3).mean()) if len(avg_by_month) else 0.0
    pos = list(idx).index(month_period)
    lo = max(0, pos - 2)
    return float(avg_by_month.iloc[lo:pos+1].mean())

@app.get("/overview")
def overview():
    # current period month as Period('YYYY-MM')
    cur_month = (CUR_START.to_period("M"))

    # 1) Total Batches (started in current month)
    total_batches = int(
        batches[(batches["start"] >= CUR_START) & (batches["start"] <= CUR_END)].shape[0]
    )

    # 2) Delayed Rate (completed in current month, processing_days > 2)
    completed_cur = batches[
        (batches["end"].notna()) & (batches["end"] >= CUR_START) & (batches["end"] <= CUR_END)
    ].copy()
    delayed_rate = round(
        (completed_cur["processing_days"] > 2).mean() * 100, 2
    ) if not completed_cur.empty else 0.0

    # 3) Avg Processing Days (current month) + rolling 3-month avg
    avg_proc_days = round(float(completed_cur["processing_days"].mean()), 2) if not completed_cur.empty else 0.0
    rolling_avg_3mo = round(rolling_3mo_for(cur_month), 2)

    # 4) Avg Scrap Factor (plant-wide mean)
    avg_scrap_factor = round(float(df["SCRAP_FACTOR"].mean() * 100), 2)

    return JSONResponse(content={
        "period": {
            "label": str(cur_month),                 # e.g., "2025-06"
            "start": str(CUR_START.date()),
            "end": str(CUR_END.date())
        },
        "stats": {
            "total_batches": total_batches,                          # across all lines
            "delayed_rate_percent": delayed_rate,                    # share of delayed (proc > 2d)
            "avg_processing_days": avg_proc_days,                    # current month
            "avg_processing_days_3mo": rolling_avg_3mo,              # rolling average
            "avg_scrap_factor_percent": avg_scrap_factor             # plant-wide mean
        }
    })
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# main.py
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "batch_details.csv")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def to_python(obj):
    """
    Recursively convert NumPy/Pandas types to vanilla Python/JSON types.
    Prevents Pydantic v2 serialization errors.
    """
    # Scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Datetimes
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return pd.to_datetime(obj).isoformat()

    # Missing
    if obj is None:
        return None
    try:
        if pd.isna(obj):  # catches NaN/NaT
            return None
    except Exception:
        pass

    # Containers
    if isinstance(obj, pd.Series):
        return to_python(obj.tolist())
    if isinstance(obj, (pd.Index,)):
        return to_python(list(obj))
    if isinstance(obj, pd.DataFrame):
        return [to_python(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_python(x) for x in obj]

    return obj


def load_and_analyze_csv():
    """Load CSV and return the full dataframe + detected date columns."""
    try:
        # low_memory=False avoids mixed-type chunk inference warning
        df = pd.read_csv(CSV_PATH, low_memory=False)

        # Try to parse date-like columns automatically
        date_columns = []
        for col in df.columns:
            if any(word in col.lower() for word in ['date', 'time', 'created', 'updated', 'timestamp']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    date_columns.append(col)
                except Exception:
                    pass

        return df, date_columns
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, []


def choose_best_date_col(df: pd.DataFrame, date_columns: list[str]) -> Optional[str]:
    """Pick the date column with the most non-NA timestamps."""
    if not date_columns:
        return None
    return max(date_columns, key=lambda c: df[c].notna().sum())


def get_time_period_filter(df: pd.DataFrame, period: str, date_columns: list[str]) -> pd.DataFrame:
    """Filter dataframe for a specific time period using the best date column."""
    if not date_columns:
        return df

    date_col = choose_best_date_col(df, date_columns)
    if not date_col:
        return df

    current_date = datetime.now()

    if 'this month' in period.lower():
        start_date = current_date.replace(day=1)
        return df[df[date_col] >= start_date]

    elif 'this quarter' in period.lower():
        q_start_month = ((current_date.month - 1) // 3) * 3 + 1
        quarter_start = current_date.replace(month=q_start_month, day=1)
        return df[df[date_col] >= quarter_start]

    elif 'this year' in period.lower():
        year_start = current_date.replace(month=1, day=1)
        return df[df[date_col] >= year_start]

    elif 'last month' in period.lower():
        last_month = current_date - relativedelta(months=1)
        start_date = last_month.replace(day=1)
        end_date = current_date.replace(day=1) - timedelta(days=1)
        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
        return df[mask]

    elif 'last quarter' in period.lower():
        current_q_start_month = ((current_date.month - 1) // 3) * 3 + 1
        if current_q_start_month == 1:
            last_quarter_start = (current_date - relativedelta(years=1)).replace(month=10, day=1)
        else:
            last_quarter_start = current_date.replace(month=current_q_start_month - 3, day=1)
        quarter_start = current_date.replace(month=current_q_start_month, day=1)
        mask = (df[date_col] >= last_quarter_start) & (df[date_col] < quarter_start)
        return df[mask]

    return df


def is_numeric(series: pd.Series) -> bool:
    """Robust numeric dtype check."""
    return pd.api.types.is_numeric_dtype(series)


def find_relevant_columns(df: pd.DataFrame, query: str):
    """Find columns relevant to the query based on keywords and dtypes."""
    query_lower = query.lower()
    relevant_cols: Dict[str, Any] = {}

    # Delay-related columns
    delay_keywords = ['delay', 'late', 'behind', 'overdue', 'slow']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in delay_keywords):
            relevant_cols['delay_column'] = col
            break

    # Rate/percentage columns
    rate_keywords = ['rate', 'percent', 'ratio', '%']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in rate_keywords):
            relevant_cols['rate_column'] = col
            break

    # Reason columns
    reason_keywords = ['reason', 'cause', 'issue', 'problem', 'type', 'category']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in reason_keywords):
            relevant_cols['reason_column'] = col
            break

    # Line/location columns
    line_keywords = ['line', 'location', 'station', 'area', 'zone', 'department']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in line_keywords):
            relevant_cols['line_column'] = col
            break

    # Numeric columns for calculations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    relevant_cols['numeric_columns'] = numeric_cols

    return relevant_cols


def analyze_advanced_query(df: pd.DataFrame, query: str, date_columns: list[str]):
    """Analyze complex queries with time periods and specific metrics."""
    query_lower = query.lower().strip()
    insights: Dict[str, Any] = {}
    result_data: Dict[str, Any] = {}

    # Relevant columns
    relevant_cols = find_relevant_columns(df, query)

    # Time period filtering
    filtered_df = df
    time_periods = ['this month', 'this quarter', 'this year', 'last month', 'last quarter']
    for period in time_periods:
        if period in query_lower:
            filtered_df = get_time_period_filter(df, period, date_columns)
            insights['time_period'] = period
            insights['filtered_records'] = len(filtered_df)
            break

    # Delay rate calculations
    if any(word in query_lower for word in ['delay rate', 'delay percentage', 'rate']):
        delay_col = relevant_cols.get('delay_column')
        if delay_col and delay_col in filtered_df.columns:
            if is_numeric(filtered_df[delay_col]):
                # If numeric, calculate mean rate/metric
                delay_rate = filtered_df[delay_col].mean()
                insights['delay_rate'] = round(float(delay_rate), 2)
            else:
                # If categorical, estimate percentage of delayed items
                delayed_count = filtered_df[delay_col].astype(str).str.contains(
                    'delay|late|behind', case=False, na=False
                ).sum()
                total_count = len(filtered_df)
                delay_rate = (delayed_count / total_count) * 100 if total_count > 0 else 0
                insights['delay_rate'] = round(float(delay_rate), 2)
                insights['delayed_items'] = int(delayed_count)
                insights['total_items'] = int(total_count)
        else:
            # Infer delay mentions across object columns
            delay_indicators = []
            for col in filtered_df.columns:
                if filtered_df[col].dtype == 'object':
                    delay_count = filtered_df[col].astype(str).str.contains(
                        'delay|late|behind', case=False, na=False
                    ).sum()
                    if delay_count > 0:
                        delay_indicators.append((col, int(delay_count)))

            if delay_indicators:
                delay_col, delay_count = max(delay_indicators, key=lambda x: x[1])
                delay_rate = (delay_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                insights['delay_rate'] = round(float(delay_rate), 2)
                insights['delay_column_used'] = delay_col

    # Top delay reasons
    if any(phrase in query_lower for phrase in ['top delay', 'delay reason', 'main cause']):
        reason_col = relevant_cols.get('reason_column')
        if reason_col and reason_col in filtered_df.columns:
            delay_mask = filtered_df[reason_col].astype(str).str.contains(
                'delay|late|behind', case=False, na=False
            )
            delay_reasons = filtered_df[delay_mask][reason_col].value_counts()
            numbers = re.findall(r'\d+', query)
            n = int(numbers[0]) if numbers else 5
            top_reasons = delay_reasons.head(n)
            result_data['top_delay_reasons'] = {str(k): int(v) for k, v in top_reasons.items()}
            insights['total_delay_reasons'] = int(delay_reasons.size)
        else:
            # Try any categorical column that might contain reasons
            categorical_cols = filtered_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if any(word in col.lower() for word in ['reason', 'cause', 'issue', 'type', 'category', 'problem']):
                    reason_counts = filtered_df[col].value_counts()
                    numbers = re.findall(r'\d+', query)
                    n = int(numbers[0]) if numbers else 5
                    result_data[f'top_{col.lower()}'] = {str(k): int(v) for k, v in reason_counts.head(n).items()}
                    break

    # Line/location with highest delay
    if any(phrase in query_lower for phrase in ['highest delay', 'most delay', 'worst line']):
        line_col = relevant_cols.get('line_column')
        delay_col = relevant_cols.get('delay_column')

        if line_col and line_col in filtered_df.columns:
            if delay_col and delay_col in filtered_df.columns and is_numeric(filtered_df[delay_col]):
                line_delays = filtered_df.groupby(line_col, dropna=False)[delay_col].mean().sort_values(ascending=False)
                result_data['line_delay_averages'] = {str(k): float(v) for k, v in line_delays.items()}
                if len(line_delays) > 0:
                    insights['highest_delay_line'] = str(line_delays.index[0])
                    insights['highest_delay_value'] = float(line_delays.iloc[0])
            else:
                # Count delay mentions by line across object columns
                best_counts = None
                for col in filtered_df.columns:
                    if col != line_col and filtered_df[col].dtype == 'object':
                        delay_by_line = filtered_df.groupby(line_col, dropna=False)[col].apply(
                            lambda x: x.astype(str).str.contains('delay|late|behind', case=False, na=False).sum()
                        ).sort_values(ascending=False)
                        if delay_by_line.sum() > 0:
                            best_counts = delay_by_line
                            break
                if best_counts is not None:
                    result_data['delay_count_by_line'] = {str(k): int(v) for k, v in best_counts.items()}
                    if len(best_counts) > 0:
                        insights['highest_delay_line'] = str(best_counts.index[0])

    # Average delay by line (generic)
    if any(phrase in query_lower for phrase in ['average delay', 'mean delay']):
        line_col = relevant_cols.get('line_column')
        numeric_cols = relevant_cols.get('numeric_columns', [])

        if line_col and line_col in filtered_df.columns and numeric_cols:
            for num_col in numeric_cols:
                if any(word in num_col.lower() for word in ['delay', 'time', 'duration', 'minutes', 'hours']):
                    line_averages = filtered_df.groupby(line_col, dropna=False)[num_col].mean().sort_values(ascending=False)
                    result_data['average_by_line'] = {str(k): float(v) for k, v in line_averages.items()}
                    insights['metric_used'] = num_col
                    break

    # General statistics fallback
    if not insights and not result_data:
        insights['total_records'] = int(len(filtered_df))
        insights['columns_available'] = [str(c) for c in filtered_df.columns]

        # Any delay mentions, broadly
        for col in filtered_df.columns:
            if filtered_df[col].dtype == 'object':
                delay_count = filtered_df[col].astype(str).str.contains(
                    'delay|late|behind', case=False, na=False
                ).sum()
                if delay_count > 0:
                    insights[f'delay_mentions_in_{col}'] = int(delay_count)

    return insights, result_data, filtered_df


def generate_suggested_questions(df: pd.DataFrame, current_query: str, insights: Dict, date_columns: list[str]) -> list[str]:
    """Generate contextually relevant suggested questions based on data and query."""
    suggestions: list[str] = []
    query_lower = current_query.lower()

    relevant_cols = find_relevant_columns(df, current_query)

    # Time-based suggestions
    if date_columns:
        if 'this month' in query_lower:
            suggestions.append("What was the delay rate last month?")
        elif 'this quarter' in query_lower:
            suggestions.append("How does this compare to last quarter?")
        else:
            suggestions.append("What is the delay rate this month?")

    # Delay-specific suggestions
    if 'delay rate' in query_lower:
        suggestions.extend([
            "Show top delay reasons this quarter",
            "Which line has the highest average delay?"
        ])
    elif 'top delay' in query_lower:
        suggestions.extend([
            "What is the overall delay rate?",
            "Which time period has the most delays?"
        ])
    elif 'highest delay' in query_lower or 'average delay' in query_lower:
        suggestions.extend([
            "Show top delay reasons for this line",
            "What is the delay trend over time?"
        ])
    else:
        if relevant_cols.get('delay_column'):
            suggestions.append("What is the delay rate this month?")
        if relevant_cols.get('reason_column'):
            suggestions.append("Show top delay reasons this quarter")
        if relevant_cols.get('line_column'):
            suggestions.append("Which line has the highest average delay?")
        if date_columns:
            suggestions.append("How do delays compare month over month?")

    # Ensure at least 2
    generic_suggestions = [
        "Show me data quality issues",
        "What are the main categories in the data?",
        "Show me the distribution of records by time",
        "What are the key performance indicators?"
    ]
    for suggestion in generic_suggestions:
        if len(suggestions) >= 2:
            break
        if suggestion not in suggestions:
            suggestions.append(suggestion)

    return suggestions[:2]


def generate_advanced_html_response(query: str, insights: Dict, result_data: Dict,
                                    original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> str:
    """Generate HTML response based on advanced insights."""
    html_parts = ["<div class='query-response'>"]
    query_lower = query.lower()

    # Show time period if filtered
    if 'time_period' in insights:
        html_parts.append(f"<p><em>Analysis for: {insights['time_period']}</em></p>")
        html_parts.append(f"<p>Records in this period: <strong>{insights['filtered_records']}</strong> out of {len(original_df)} total</p>")

    # Handle delay rate
    if 'delay_rate' in insights:
        html_parts.append(f"<p><strong>Delay Rate:</strong> {insights['delay_rate']}%</p>")
        if 'delayed_items' in insights:
            html_parts.append(f"<p>({insights['delayed_items']} delayed items out of {insights['total_items']} total)</p>")
        if 'delay_column_used' in insights:
            html_parts.append(f"<p><em>Based on analysis of: {insights['delay_column_used']}</em></p>")

    # Handle top delay reasons
    if 'top_delay_reasons' in result_data:
        html_parts.append("<h4>Top Delay Reasons:</h4>")
        html_parts.append("<div class='delay-reasons'>")
        for reason, count in result_data['top_delay_reasons'].items():
            html_parts.append(f"<p>• <strong>{reason}:</strong> {count} occurrences</p>")
        html_parts.append("</div>")

    # Other top categories
    for key, data in result_data.items():
        if key.startswith('top_') and key != 'top_delay_reasons':
            category_name = key.replace('top_', '').replace('_', ' ').title()
            html_parts.append(f"<h4>Top {category_name}:</h4>")
            html_parts.append("<ul>")
            for item, count in data.items():
                html_parts.append(f"<li><strong>{item}:</strong> {count}</li>")
            html_parts.append("</ul>")

    # Line with highest delay
    if 'highest_delay_line' in insights:
        html_parts.append(f"<p><strong>Line with Highest Delay:</strong> {insights['highest_delay_line']}</p>")
        if 'highest_delay_value' in insights and insights['highest_delay_value'] is not None:
            html_parts.append(f"<p><strong>Average Delay:</strong> {float(insights['highest_delay_value']):.2f}</p>")

    # Average delay by line table
    if 'line_delay_averages' in result_data:
        html_parts.append("<h4>Average Delay by Line:</h4>")
        html_parts.append("<table border='1' style='border-collapse: collapse; width: 100%;'>")
        html_parts.append("<tr><th style='padding: 8px; background-color: #f2f2f2;'>Line</th>"
                          "<th style='padding: 8px; background-color: #f2f2f2;'>Average Delay</th></tr>")
        for line, avg_delay in result_data['line_delay_averages'].items():
            html_parts.append(f"<tr><td style='padding: 8px;'>{line}</td><td style='padding: 8px;'>{float(avg_delay):.2f}</td></tr>")
        html_parts.append("</table>")

    # Generic average by line table
    if 'average_by_line' in result_data:
        metric_name = insights.get('metric_used', 'Value')
        html_parts.append(f"<h4>Average {metric_name} by Line:</h4>")
        html_parts.append("<table border='1' style='border-collapse: collapse; width: 100%;'>")
        html_parts.append(f"<tr><th style='padding: 8px; background-color: #f2f2f2;'>Line</th>"
                          f"<th style='padding: 8px; background-color: #f2f2f2;'>Average {metric_name}</th></tr>")
        for line, avg_value in result_data['average_by_line'].items():
            html_parts.append(f"<tr><td style='padding: 8px;'>{line}</td><td style='padding: 8px;'>{float(avg_value):.2f}</td></tr>")
        html_parts.append("</table>")

    # Delay count by line list
    if 'delay_count_by_line' in result_data:
        html_parts.append("<h4>Delay Count by Line:</h4>")
        html_parts.append("<ul>")
        for line, count in result_data['delay_count_by_line'].items():
            html_parts.append(f"<li><strong>{line}:</strong> {count} delays</li>")
        html_parts.append("</ul>")

    # General info if nothing specific
    has_specific = any(k in result_data for k in ['top_delay_reasons', 'line_delay_averages', 'average_by_line']) or ('delay_rate' in insights)
    if not has_specific:
        html_parts.append(f"<p>I found a dataset with <strong>{len(filtered_df)}</strong> filtered records "
                          f"out of {len(original_df)} total records.</p>")
        html_parts.append("<p><strong>Available columns:</strong></p>")
        html_parts.append("<ul>")
        for col in filtered_df.columns:
            html_parts.append(f"<li>{col} ({filtered_df[col].dtype})</li>")
        html_parts.append("</ul>")

        if len(filtered_df) > 0:
            html_parts.append("<h4>Sample Data:</h4>")
            sample_data = filtered_df.head(3)
            html_parts.append("<table border='1' style='border-collapse: collapse; width: 100%;'>")
            # header
            html_parts.append("<tr>")
            for col in sample_data.columns:
                html_parts.append(f"<th style='padding: 8px; background-color: #f2f2f2;'>{col}</th>")
            html_parts.append("</tr>")
            # rows
            for _, row in sample_data.iterrows():
                html_parts.append("<tr>")
                for value in row:
                    html_parts.append(f"<td style='padding: 8px;'>{value}</td>")
                html_parts.append("</tr>")
            html_parts.append("</table>")

    html_parts.append("</div>")
    return "".join(html_parts)

# -----------------------------------------------------------------------------
# FastAPI models & app
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7


class QueryResponse(BaseModel):
    status: str
    query: str
    result: str
    data_insights: Optional[Dict[str, Any]] = None
    suggested_questions: list[str]
    parameters: Dict[str, Any]


# app = FastAPI(title="Enhanced Delay Analysis API")


# -----------------------------------------------------------------------------
# Chatbot models and helpers
# -----------------------------------------------------------------------------
class ChatbotRequest(BaseModel):
    message: str
    include_data_context: bool = True
    sample_rows: int = 3
    max_columns: int = 20

class ChatbotResponse(BaseModel):
    status: str
    answer: str
    used_data_context: bool

def build_chat_context(df: pd.DataFrame, max_columns: int = 20, sample_rows: int = 3) -> str:
    """Create a compact textual context about the loaded dataframe for the chatbot."""
    if df is None or df.empty:
        return "No data is currently loaded."

    lines = []
    lines.append("DATA OVERVIEW")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")

    # Columns and dtypes (limited)
    cols = list(df.columns)[:max_columns]
    lines.append("\nCOLUMNS (limited):")
    for c in cols:
        lines.append(f"- {c}: {str(df[c].dtype)}")

    # Sample rows (limited)
    sample = df.head(sample_rows).copy()
    for c in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[c]):
            sample[c] = pd.to_datetime(sample[c], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    lines.append(f"\nSAMPLE ROWS (first {len(sample)}):")
    lines.append(sample.to_csv(index=False))

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# AI Agent for pandas/matplotlib code-gen + execution + prescriptive summary
# -----------------------------------------------------------------------------
class AgentDataStore:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

agent_data_store = AgentDataStore()

class ChatRequest(BaseModel):
    question: str
    include_code: bool = True
    include_visualization: bool = True

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: float

def agent_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the alarm-like dataframe with common columns if present."""
    # Event Time normalization (robust)
    event_col = None
    if 'Event Time' in df.columns:
        event_col = 'Event Time'
    else:
        for cand in ['timestamp', 'time', 'event_time', 'Datetime', 'Date', 'date', 'created_at', 'updated_at']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'Event Time'})
                event_col = 'Event Time'
                break
        # Manufacturing dataset fallback
        if event_col is None and 'WIP_ACT_START_DATE' in df.columns:
            df['Event Time'] = df['WIP_ACT_START_DATE']
            event_col = 'Event Time'
        # Generic first datetime-like column
        if event_col is None:
            date_like = [c for c in df.columns if any(w in c.lower() for w in ['date', 'time', 'timestamp'])]
            if date_like:
                df['Event Time'] = df[date_like[0]]
                event_col = 'Event Time'
        # If still none, create empty datetime column
        if event_col is None:
            df['Event Time'] = pd.NaT
            event_col = 'Event Time'

    df['Event Time'] = pd.to_datetime(df['Event Time'], errors='coerce')
    # Derive Hour/Date safely
    try:
        df['Hour'] = df['Event Time'].dt.hour
        df['Date'] = df['Event Time'].dt.date
    except Exception:
        df['Hour'] = np.nan
        df['Date'] = pd.NaT

    # Action normalization
    if 'Action' not in df.columns:
        df['Action'] = 'NO ACTION'
    else:
        df['Action'] = df['Action'].fillna('NO ACTION')

    # Condition/alarm flag
    if 'Condition' not in df.columns:
        df['Condition'] = 'UNKNOWN'
    alarm_keywords = ['ALARM', 'PVHIHI', 'PVHI', 'PVLO', 'PVLOW', 'PVLOLOW', 'HIHI', 'HI', 'LO', 'LOLO', 'FAIL']
    df['Is_Alarm'] = df['Condition'].astype(str).str.upper().str.contains('|'.join(alarm_keywords), na=False)

    # Source fallback
    if 'Source' not in df.columns:
        df['Source'] = 'UNKNOWN'

    return df

def agent_try_load_default_csv() -> bool:
    """Attempt to load batch_details.csv from the project root into the agent datastore."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, 'batch_details.csv')
        if not os.path.exists(candidate):
            return False

        # Try a few read strategies for robustness
        read_attempts = [
            {"low_memory": False},
            {"encoding": "utf-8-sig", "low_memory": False},
            {"engine": "python", "low_memory": False},
            {"sep": ";", "engine": "python", "low_memory": False},
            {"sep": "\t", "engine": "python", "low_memory": False},
        ]
        for opts in read_attempts:
            try:
                df_local = pd.read_csv(candidate, **opts)
                agent_data_store.df = agent_prepare_dataframe(df_local)
                return True
            except Exception:
                continue
        return False
    except Exception:
        return False

# Auto-load default CSV at startup if present (use directory of this file as root)
try:
    if agent_data_store.df is None:
        agent_try_load_default_csv()
except Exception:
    pass

def agent_build_dataframe_context(df: pd.DataFrame) -> str:
    """Build a concise context about the dataframe for AI."""
    if df is None or len(df) == 0:
        return "No data available."

    context = f"""
DATAFRAME INFO:
- Shape: {df.shape}
- Columns: {list(map(str, df.columns))}
- Date Range: {df['Event Time'].min()} to {df['Event Time'].max()}

COLUMN TYPES:
{df.dtypes.astype(str).to_dict()}

SAMPLE DATA (first 3 rows):
{df.head(3).to_dict('records')}

NUMERIC COLUMNS STATS:
{df.select_dtypes(include=np.number).describe().to_dict()}

CATEGORICAL VALUE COUNTS:
- Sources: {df['Source'].value_counts().head(5).to_dict() if 'Source' in df.columns else {}}
- Conditions: {df['Condition'].value_counts().head(5).to_dict() if 'Condition' in df.columns else {}}
- Actions: {df['Action'].value_counts().to_dict() if 'Action' in df.columns else {}}
- Alarms: {df['Is_Alarm'].value_counts().to_dict() if 'Is_Alarm' in df.columns else {}}
"""
    return context

def agent_generate_code_with_ai(question: str, context: str) -> str:
    """Generate Python code using OpenAI via the existing `client`."""
    system_prompt = """You are a data analyst expert in pandas and matplotlib.
Given a dataframe 'df' with manufacturing/operations data, write Python code to answer the user's question.

STRICT RULES:
1) Do NOT import libraries. Use the provided variables: df (pandas.DataFrame), pd, np, plt, datetime.
2) Never read/write files or make network calls.
3) Create charts with matplotlib only. Set `fig = plt.gcf()` when done. Do not call plt.show().
4) Put your main computed answer in a variable named `result` (DataFrame, Series, or scalar).
5) Prefer meaningful x-axis labels when ranking or plotting: choose the first available from
   ["WIP_BATCH_ID", "FORMULA_ID", "LINE_NO", "INVENTORY_ITEM_ID", "WIP_BATCH_NO", "BATCH_ID"].
6) For numeric measures like ["WIP_QTY", "ORIGINAL_QTY", "PLAN_QTY", "WIP_VALUE", "WIP_RATE", "SCRAP_FACTOR"],
   coerce to numeric with `pd.to_numeric(..., errors='coerce')` and drop NaNs before aggregations or ranking.
7) When plotting ranked bars, sort descending, set a readable figsize (e.g., (8, 4.5)), and apply thousands formatting on y-axis if values are large.
8) Keep code concise, no prints, no inline data previews.

Return ONLY the Python code in a code block, no explanations."""

    user_prompt = f"""
Context about the dataframe:
{context}

User Question: {question}

Write Python code to analyze the data and answer this question.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def agent_extract_code_block(text: str) -> str:
    """Extract Python code from markdown code block."""
    if '```python' in text:
        start = text.find('```python') + 9
        end = text.find('```', start)
        return text[start:end].strip()
    elif '```' in text:
        start = text.find('```') + 3
        end = text.find('```', start)
        return text[start:end].strip()
    return text.strip()

def agent_execute_code_safely(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment."""
    # Restricted importer to allow only safe, whitelisted modules and map to preloaded ones
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed = {
            'pandas': pd,
            'numpy': np,
            'matplotlib': matplotlib,
            'matplotlib.pyplot': plt,
            'datetime': datetime,
            'math': math,
        }
        if name in allowed:
            return allowed[name]
        # fromlist handling (e.g., from matplotlib import pyplot)
        if fromlist:
            base = allowed.get(name)
            if base is not None:
                return base
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")

    sandbox_builtins = {
        '__import__': _restricted_import,
        'len': len, 'sum': sum, 'min': min, 'max': max,
        'round': round, 'abs': abs, 'sorted': sorted,
        'enumerate': enumerate, 'zip': zip, 'range': range,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'str': str, 'int': int, 'float': float, 'bool': bool,
    }
    # Use a single namespace for both globals and locals so functions/lambdas
    # created during exec can resolve names like `pd` at runtime.
    sandbox = {
        '__builtins__': sandbox_builtins,
        'pd': pd,
        'np': np,
        'plt': plt,
        'matplotlib': matplotlib,
        'datetime': datetime,
        'math': math,
    }
    # Provide the dataframe as a variable in the sandbox
    sandbox['df'] = df.copy()

    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, sandbox, sandbox)

        result = sandbox.get('result', None)
        fig = sandbox.get('fig', None)

        if fig is None and plt.get_fignums():
            fig = plt.gcf()

        fig_base64 = None
        if fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            fig_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

        if isinstance(result, pd.DataFrame):
            result_formatted = {
                'type': 'dataframe',
                'data': result.to_dict('records'),
                'shape': result.shape,
                'columns': list(result.columns)
            }
        elif isinstance(result, pd.Series):
            result_formatted = {
                'type': 'series',
                'data': result.to_dict(),
                'name': result.name
            }
        elif result is not None:
            # Convert numpy arrays/scalars or other non-serializable types
            try:
                converted = to_python(result)
            except Exception:
                converted = result.tolist() if isinstance(result, np.ndarray) else result
            result_formatted = {
                'type': 'value',
                'data': converted
            }
        else:
            result_formatted = None

        # Final safety: coerce any nested numpy types to pure Python
        try:
            result_formatted = to_python(result_formatted) if result_formatted is not None else None
        except Exception:
            pass

        return {
            'success': True,
            'result': result_formatted,
            'figure': fig_base64,
            'stdout': stdout_buffer.getvalue(),
            'error': None
        }
    except Exception:
        return {
            'success': False,
            'result': None,
            'figure': None,
            'stdout': stdout_buffer.getvalue(),
            'error': traceback.format_exc()
        }
    finally:
        plt.close('all')

def agent_generate_summary(question: str, result: Any, code: str) -> str:
    """Ask LLM for an actionable prescriptive summary based on the result and code."""
    try:
        result_summary = str(result)[:1000] if result else "No numeric result"
        prompt = f"""
Question: {question}
Code executed: {code[:500]}...
Result: {result_summary}

Provide a brief, actionable summary of the findings and recommendations.
Keep it under 150 words and focus on practical insights.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=200,
            messages=[
                {"role": "system", "content": "You are an industrial alarm system analyst. Provide concise, actionable insights."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception:
        return "Analysis complete. See results above."


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------
async def process_query(query: str, max_tokens: int = 1000, temperature: float = 0.7):
    """Process the user query and return comprehensive analysis."""
    df_result = load_and_analyze_csv()
    if df_result[0] is None:
        return {
            "html_content": "<div class='error'><p>Sorry, I couldn't load the data file. Please check if the file exists.</p></div>",
            "data_insights": None,
            "suggested_questions": ["Check if the data file exists", "Try reloading the data"]
        }

    df, date_columns = df_result

    # Greetings
    query_lower = query.lower().strip()
    if query_lower in ['hi', 'hello', 'hey']:
        suggested_questions = generate_suggested_questions(df, query, {}, date_columns)
        return {
            "html_content": (
                f"<div class='greeting'><p>Hello! I'm here to help you analyze your data with "
                f"{len(df)} records and {len(df.columns)} columns. I can help with delay analysis, "
                f"time-based queries, and more!</p></div>"
            ),
            "data_insights": to_python({
                "total_records": len(df),
                "total_columns": len(df.columns),
                "date_columns": date_columns
            }),
            "suggested_questions": to_python(suggested_questions)
        }

    # Advanced analysis
    query_insights, result_data, filtered_df = analyze_advanced_query(df, query, date_columns)

    # Suggestions
    suggested_questions = generate_suggested_questions(df, query, query_insights, date_columns)

    try:
        html_response = generate_advanced_html_response(query, query_insights, result_data, df, filtered_df)
        return {
            "html_content": html_response,
            "data_insights": to_python(query_insights),          # sanitize for Pydantic
            "suggested_questions": to_python(suggested_questions)
        }
    except Exception as e:
        return {
            "html_content": f"<div class='error'><p>Sorry, I encountered an error while processing your query: {str(e)}</p></div>",
            "data_insights": None,
            "suggested_questions": ["Try a simpler query", "Check the data format"]
        }

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "title": "Enhanced Delay Analysis API",
        "description": "Advanced analysis of batch/operational data with focus on delay analysis, time-based queries, and performance metrics.",
        "features": [
            "Time-based filtering (this month, this quarter, etc.)",
            "Delay rate calculations and analysis",
            "Top delay reasons identification",
            "Line/location performance analysis",
            "Automatic date column detection",
            "Advanced pattern matching for complex queries"
        ],
        "endpoints": {
            "GET /query": {
                "description": "Query data using natural language with advanced delay analysis",
                "parameters": {
                    "q": {"type": "string", "required": True, "description": "Your query string"},
                    "max_tokens": {"type": "integer", "required": False, "default": 1000},
                    "temperature": {"type": "float", "required": False, "default": 0.7}
                },
                "example": "/query?q=What is the delay rate this month?"
            }
        },
        "sample_queries": [
            "What is the delay rate this month?",
            "Show top delay reasons this quarter",
            "Which line has the highest average delay?",
            "What are the main delay causes last month?",
            "How do delays compare by location?",
            "Show me delay trends over time",
            "What is the overall performance this year?",
            "Which areas have the most issues?"
        ]
    }


@app.get("/query", response_model=QueryResponse)
async def query_get(
    q: str = Query(..., description="Query string"),
    max_tokens: int = Query(1000, description="Maximum tokens for response"),
    temperature: float = Query(0.7, description="Response creativity (0.0-1.0)")
):
    """Enhanced query endpoint with advanced delay analysis capabilities."""
    try:
        result = await process_query(q, max_tokens, temperature)
        return QueryResponse(
            status="success",
            query=q,
            result=result["html_content"],
            data_insights=to_python(result.get("data_insights")),
            suggested_questions=to_python(result.get("suggested_questions", [])),
            parameters={
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "query": q,
                "error": str(e),
                "suggested_questions": ["Try a simpler query", "Check the data file"]
            }
        )


@app.post("/query", response_model=QueryResponse)
async def query_post(request: QueryRequest):
    """Enhanced query endpoint using POST method with advanced delay analysis."""
    try:
        result = await process_query(request.query, request.max_tokens, request.temperature)
        return QueryResponse(
            status="success",
            query=request.query,
            result=result["html_content"],
            data_insights=to_python(result.get("data_insights")),
            suggested_questions=to_python(result.get("suggested_questions", [])),
            parameters={
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "query": request.query,
                "error": str(e),
                "suggested_questions": ["Try a simpler query", "Check the data file"]
            }
        )

@app.post("/chatbot/ask", response_model=ChatbotResponse)
async def chatbot_ask(request: ChatbotRequest):
    """Lightweight chatbot that optionally uses a compact data context from the loaded dataframe."""
    try:
        system_prompt = (
            "You are a helpful manufacturing analytics assistant. "
            "Answer concisely using Markdown. If data context is provided, ground your answer in it."
        )

        used_context = False
        user_content = request.message

        # Build compact context from the global df if requested
        if request.include_data_context:
            try:
                context_text = build_chat_context(df, max_columns=request.max_columns, sample_rows=request.sample_rows)
                if context_text and context_text.strip() and context_text.strip() != "No data is currently loaded.":
                    user_content = f"Data Context:\n{context_text}\n\nQuestion: {request.message}"
                    used_context = True
            except Exception:
                # If context building fails, continue without it
                pass

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        answer = response.choices[0].message.content
        return ChatbotResponse(status="success", answer=answer, used_data_context=used_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/upload-data")
async def agent_upload_data(file: UploadFile = File(...)):
    """Upload CSV data file for the AI agent."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        df = agent_prepare_dataframe(df)
        agent_data_store.df = df

        return {
            "success": True,
            "message": "Data uploaded successfully",
            "stats": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "date_range": {
                    "start": str(df['Event Time'].min()),
                    "end": str(df['Event Time'].max())
                },
                "total_alarms": int(df['Is_Alarm'].sum()) if 'Is_Alarm' in df.columns else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def agent_chat_with_data(request: ChatRequest):
    """Generate pandas/matplotlib code with AI, execute safely, then ask LLM for a prescriptive summary."""
    start_time = datetime.now()
    try:
        if agent_data_store.df is None:
            agent_try_load_default_csv()
        if agent_data_store.df is None:
            default_path = os.path.join(BASE_DIR, 'batch_details.csv')
            exists = os.path.exists(default_path)
            return ChatResponse(
                success=False,
                answer=None,
                error=f"No data loaded. Expected default CSV at: {default_path} (exists={exists}). Place the file there or upload via /api/upload-data.",
                execution_time=0
            )

        context_text = agent_build_dataframe_context(agent_data_store.df)
        ai_response = agent_generate_code_with_ai(request.question, context_text)
        code = agent_extract_code_block(ai_response)

        execution_result = agent_execute_code_safely(code, agent_data_store.df)

        summary = ""
        if execution_result['success']:
            summary = agent_generate_summary(request.question, execution_result['result'], code)

        answer = {
            "question": request.question,
            "code": code if request.include_code else None,
            "execution": {
                "success": execution_result['success'],
                "result": execution_result['result'],
                "figure": execution_result['figure'] if request.include_visualization else None,
                "stdout": execution_result['stdout'],
                "error": execution_result['error']
            },
            "summary": summary
        }

        # Coerce any lingering numpy/pandas types to plain Python for Pydantic serialization
        try:
            answer = to_python(answer)
        except Exception:
            pass

        execution_time = (datetime.now() - start_time).total_seconds()
        return ChatResponse(success=execution_result['success'], answer=answer, error=execution_result['error'], execution_time=execution_time)
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        return ChatResponse(success=False, answer=None, error=str(e), execution_time=execution_time)

@app.get("/api/status")
async def agent_get_status():
    if agent_data_store.df is None:
        agent_try_load_default_csv()
    default_path = os.path.join(BASE_DIR, 'batch_details.csv')
    return {
        "data_loaded": agent_data_store.df is not None,
        "openai_available": True,
        "rows": int(len(agent_data_store.df)) if agent_data_store.df is not None else 0,
        "default_csv_path": default_path,
        "default_csv_exists": os.path.exists(default_path)
    }


# # ========= FORMULA vs ACTUAL (DATA-AWARE, NO INPUTS) =========
# import json
# import math
# import numpy as np
# import pandas as pd
# from fastapi.responses import JSONResponse
#
# TOP_N_BATCHES = 50  # how many batches to include in the no-input batch report
# EPS = 1e-6
# # --- JSON helper: NaN/Inf safe, pandas/numpy friendly ---
# def json_safe(obj):
#     if isinstance(obj, pd.DataFrame):
#         df2 = obj.replace([np.inf, -np.inf], np.nan)
#         return json.loads(df2.to_json(orient="records"))
#     if isinstance(obj, (np.floating, float)):
#         x = float(obj)
#         if math.isnan(x) or math.isinf(x):
#             return None
#         return x
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.bool_,)):
#         return bool(obj)
#     if isinstance(obj, pd.Timestamp):
#         return obj.isoformat()
#     if isinstance(obj, dict):
#         return {str(k): json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [json_safe(v) for v in obj]
#     return obj
#
#
# # --- DATA-AWARE normalizer (uses your actual columns) ---
# def _normalize_columns_data_aware(df: pd.DataFrame) -> pd.DataFrame:
#     need = [
#         "FORMULA_ID", "WIP_BATCH_ID", "WIP_BATCH_NO", "TRANSACTION_TYPE_NAME",
#         "INVENTORY_ITEM_ID", "PLAN_QTY", "ORIGINAL_QTY", "WIP_QTY", "WIP_RATE", "WIP_VALUE"
#     ]
#     slim = df.copy()
#     for m in need:
#         if m not in slim.columns:
#             slim[m] = np.nan
#     slim = slim[need].copy()
#     for c in ["PLAN_QTY","ORIGINAL_QTY","WIP_QTY","WIP_RATE","WIP_VALUE"]:
#         slim[c] = pd.to_numeric(slim[c], errors="coerce")
#     return slim
#
#
# # --- FG map: batch -> product (from WIP Completion lines) ---
# def _fg_map_from_completions(df: pd.DataFrame) -> pd.Series:
#     comp = df[df["TRANSACTION_TYPE_NAME"] == "WIP Completion"]
#     if comp.empty:
#         return pd.Series(dtype=float, name="PRODUCT_ID")
#     return (comp.groupby("WIP_BATCH_ID")["INVENTORY_ITEM_ID"]
#                 .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
#                 .rename("PRODUCT_ID"))
#
#
# # --- STANDARD formula per (FORMULA_ID, INGREDIENT_ID) from WIP Issue lines ---
# def _build_standard(df: pd.DataFrame) -> pd.DataFrame:
#     issue = df[df["TRANSACTION_TYPE_NAME"] == "WIP Issue"].copy()
#     if issue.empty:
#         return pd.DataFrame(columns=["FORMULA_ID","INGREDIENT_ID","STD_QTY","STD_UNIT_COST","STD_ING_COST"])
#     issue["INGREDIENT_ID"] = issue["INVENTORY_ITEM_ID"]
#     std = (issue.groupby(["FORMULA_ID","INGREDIENT_ID"], dropna=False)
#                 .agg(
#                     STD_QTY=("PLAN_QTY", lambda s: s.dropna().median() if s.notna().any() else np.nan),
#                     STD_UNIT_COST=("WIP_RATE", lambda s: s.dropna().median() if s.notna().any() else np.nan),
#                 )
#                 .reset_index())
#     std["STD_ING_COST"] = std["STD_QTY"] * std["STD_UNIT_COST"]
#     return std
#
#
# # --- ACTUAL batch composition from Issue/Return with correct signs ---
# def _build_actual(df: pd.DataFrame) -> pd.DataFrame:
#     ir = df[df["TRANSACTION_TYPE_NAME"].isin(["WIP Issue","WIP Return"])].copy()
#     if ir.empty:
#         return pd.DataFrame(columns=["BATCH_ID","FORMULA_ID","INGREDIENT_ID","ACT_QTY","ACT_COST","ACT_UNIT_COST_MED","PRODUCT_ID"])
#     ir["INGREDIENT_ID"] = ir["INVENTORY_ITEM_ID"]
#
#     # Quantities: Issues have negative WIP_QTY, Returns positive → both: consumed = -WIP_QTY
#     ir["CONS_QTY"] = -ir["WIP_QTY"]
#
#     # Costs: Issue cost positive, Return cost negative
#     ir["CONS_COST"] = np.where(ir["TRANSACTION_TYPE_NAME"]=="WIP Issue", ir["WIP_VALUE"], -ir["WIP_VALUE"])
#
#     # Unit rate
#     ir["UNIT_RATE"] = ir["WIP_RATE"]
#     mask = ir["UNIT_RATE"].isna() & ir["WIP_QTY"].ne(0)
#     ir.loc[mask, "UNIT_RATE"] = (ir.loc[mask, "WIP_VALUE"].abs() / ir.loc[mask, "WIP_QTY"].abs())
#
#     act = (ir.groupby(["WIP_BATCH_ID","FORMULA_ID","INGREDIENT_ID"], dropna=False)
#               .agg(
#                   ACT_QTY=("CONS_QTY","sum"),
#                   ACT_COST=("CONS_COST","sum"),
#                   ACT_UNIT_COST_MED=("UNIT_RATE", lambda s: s.dropna().median() if s.notna().any() else np.nan),
#               ).reset_index())
#     act.rename(columns={"WIP_BATCH_ID":"BATCH_ID"}, inplace=True)
#
#     # attach FG product per batch
#     fg_map = _fg_map_from_completions(df)
#     act = act.merge(fg_map, left_on="BATCH_ID", right_index=True, how="left")
#     return act
#
#
# # --- Compare: scale standard to batch size and compute variances ---
# EPS = 1e-6  # put this once near the top of your file (if not already there)
#
# def _compare_detail(act: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
#     # handle edge cases
#     if (act is None or std is None) or (act.empty and std.empty):
#         return pd.DataFrame()
#
#     # totals for scaling the standard to actual batch size
#     std_tot = (std.groupby(["FORMULA_ID"], dropna=False)["STD_QTY"]
#                   .sum().reset_index()
#                   .rename(columns={"STD_QTY": "STD_TOTAL_QTY"}))
#
#     act_tot = (act.groupby(["BATCH_ID","FORMULA_ID"], dropna=False)["ACT_QTY"]
#                   .sum().reset_index()
#                   .rename(columns={"ACT_QTY": "ACT_TOTAL_QTY"}))
#
#     scale = act_tot.merge(std_tot, on=["FORMULA_ID"], how="left")
#     scale["BATCH_SCALE"] = np.where(
#         scale["STD_TOTAL_QTY"].abs() > 0,
#         scale["ACT_TOTAL_QTY"] / scale["STD_TOTAL_QTY"],
#         np.nan
#     )
#
#     act2 = act.merge(
#         scale[["BATCH_ID","FORMULA_ID","BATCH_SCALE"]],
#         on=["BATCH_ID","FORMULA_ID"],
#         how="left"
#     )
#
#     # merge without using merge indicator
#     d = act2.merge(
#         std[["FORMULA_ID","INGREDIENT_ID","STD_QTY","STD_UNIT_COST"]],
#         on=["FORMULA_ID","INGREDIENT_ID"],
#         how="outer"
#     )
#
#     # compute scaled std, actual costs, and variances
#     d["STD_QTY_SCALED"]      = d["STD_QTY"] * d["BATCH_SCALE"]
#     d["STD_ING_COST_SCALED"] = d["STD_QTY_SCALED"] * d["STD_UNIT_COST"]
#     d["ACT_ING_COST_FINAL"]  = d["ACT_QTY"] * d["ACT_UNIT_COST_MED"]
#
#     d["QTY_VAR"]  = d["ACT_QTY"] - d["STD_QTY_SCALED"]
#     d["COST_VAR"] = d["ACT_ING_COST_FINAL"] - d["STD_ING_COST_SCALED"]
#
#     # STATUS bucketing (data-aware, no _merge)
#     s_std = d["STD_QTY_SCALED"].fillna(0).abs()
#     s_act = d["ACT_QTY"].fillna(0).abs()
#
#     d["STATUS"] = np.select(
#         [
#             (s_std <= EPS) & (s_act > EPS),                                    # actual used, no standard -> "ACT_ONLY"
#             (s_act <= EPS) & (s_std > EPS),                                    # standard expected, not used -> "STD_ONLY"
#             (s_std > EPS) & (s_act > EPS) & (d["QTY_VAR"].abs() > EPS),        # both present but different
#         ],
#         ["EXCESS_OR_NEW (ACT_ONLY)", "NOT_USED (STD_ONLY)", "QTY_CHANGED"],
#         default="MATCH"
#     )
#
#     return d  # <-- this is the line that prevents 'NoneType' later
#
#
# # --- Batch-level rollup ---
# def _rollup(detail: pd.DataFrame) -> pd.DataFrame:
#     if detail.empty:
#         return pd.DataFrame(columns=["BATCH_ID","STD_COST","ACT_COST","COST_VAR",
#                                      "EXCESS_ING_CNT","CHANGED_ING_CNT","UNUSED_STD_ING_CNT","VARIANCE_%"])
#     d = detail.copy()
#     for c in ["STD_ING_COST_SCALED","ACT_ING_COST_FINAL","COST_VAR"]:
#         d[c] = pd.to_numeric(d[c], errors="coerce")
#     roll = (d.groupby("BATCH_ID", dropna=False)
#               .agg(
#                   STD_COST=("STD_ING_COST_SCALED", lambda s: s.sum(min_count=1)),
#                   ACT_COST=("ACT_ING_COST_FINAL", lambda s: s.sum(min_count=1)),
#                   COST_VAR=("COST_VAR", lambda s: s.sum(min_count=1)),
#                   EXCESS_ING_CNT=("STATUS", lambda s: (s=="EXCESS_OR_NEW (ACT_ONLY)").sum()),
#                   CHANGED_ING_CNT=("STATUS", lambda s: (s=="QTY_CHANGED").sum()),
#                   UNUSED_STD_ING_CNT=("STATUS", lambda s: (s=="NOT_USED (STD_ONLY)").sum()),
#               ).reset_index())
#     roll["VARIANCE_%"] = np.where(roll["STD_COST"].abs()>0,
#                                   (roll["COST_VAR"]/roll["STD_COST"])*100,
#                                   np.nan)
#     return roll
#
#
# # --- Overview ---
# @app.get("/formula-vs-actual/overview")
# def formula_vs_actual_overview():
#     global df
#     base = _normalize_columns_data_aware(df)
#     std = _build_standard(base)
#     act = _build_actual(base)
#     detail = _compare_detail(act, std)
#     roll = _rollup(detail)
#
#     top = roll.reindex(roll["COST_VAR"].abs().sort_values(ascending=False).index).head(15)
#     payload = {
#         "status": "ok",
#         "counts": {
#             "standard_rows": int(len(std)),
#             "detail_rows": int(len(detail)),
#             "batches": int(roll["BATCH_ID"].nunique()) if not roll.empty else 0,
#         },
#         "top_batches_by_cost_variance": json_safe(top),
#     }
#     return JSONResponse(content=json_safe(payload))
#
#
# # --- Batch report (no input; top N by |variance|) ---
# @app.get("/formula-vs-actual/batch-report")
# def formula_vs_actual_batch_report():
#     global df
#     base = _normalize_columns_data_aware(df)
#     std = _build_standard(base)
#     act = _build_actual(base)
#     detail = _compare_detail(act, std)
#     roll = _rollup(detail)
#
#     if roll.empty:
#         return JSONResponse(content=json_safe({
#             "status": "ok",
#             "selection": {"criteria":"top_abs_cost_variance","top_n": 0},
#             "counts": {"batches_available": 0, "batches_in_report": 0},
#             "items": []
#         }))
#
#     roll2 = roll.copy()
#     roll2["__abs__"] = roll2["COST_VAR"].abs()
#     roll2 = roll2.sort_values("__abs__", ascending=False).drop(columns="__abs__")
#     top_ids = roll2["BATCH_ID"].head(TOP_N_BATCHES).tolist()
#
#     items = []
#     for bid in top_ids:
#         sub = detail[detail["BATCH_ID"] == bid].copy()
#         summary = json_safe(roll2[roll2["BATCH_ID"] == bid].head(1))[0]
#
#         excess = sub.loc[sub["STATUS"]=="EXCESS_OR_NEW (ACT_ONLY)",
#                          ["FORMULA_ID","INGREDIENT_ID","ACT_QTY","ACT_ING_COST_FINAL","COST_VAR"]]
#         changes = sub.loc[sub["STATUS"]=="QTY_CHANGED",
#                           ["FORMULA_ID","INGREDIENT_ID","STD_QTY_SCALED","ACT_QTY","QTY_VAR","STD_ING_COST_SCALED","ACT_ING_COST_FINAL","COST_VAR"]]
#         not_used = sub.loc[sub["STATUS"]=="NOT_USED (STD_ONLY)",
#                            ["FORMULA_ID","INGREDIENT_ID","STD_QTY_SCALED","STD_ING_COST_SCALED"]]
#
#         items.append({
#             "batch_id": json_safe(bid),
#             "summary": summary,
#             "excess_ingredients": json_safe(excess.sort_values("COST_VAR", ascending=False).head(100)),
#             "qty_changes": json_safe(changes.sort_values("COST_VAR", ascending=False).head(100)),
#             "not_used_standard": json_safe(not_used.head(100)),
#         })
#
#     payload = {
#         "status": "ok",
#         "selection": {"criteria": "top_abs_cost_variance", "top_n": TOP_N_BATCHES},
#         "counts": {"batches_available": int(roll["BATCH_ID"].nunique()), "batches_in_report": len(items)},
#         "items": items,
#     }
#     return JSONResponse(content=json_safe(payload))
#
#
# # --- Formula sheet + formula→top product codes ---
# @app.get("/formula-vs-actual/formula-sheet")
# def formula_vs_actual_formula_sheet():
#     global df
#     base = _normalize_columns_data_aware(df)
#     std = _build_standard(base)
#
#     # compact summary per formula
#     form_sum = (std.groupby("FORMULA_ID", dropna=False)
#                   .agg(ING_COUNT=("INGREDIENT_ID","nunique"),
#                        STD_TOTAL_QTY=("STD_QTY","sum"),
#                        STD_TOTAL_COST=("STD_ING_COST","sum"))
#                   .reset_index()
#                   .sort_values("FORMULA_ID"))
#
#     # map formula -> top product codes (from WIP Completion lines)
#     comp = base[base["TRANSACTION_TYPE_NAME"]=="WIP Completion"].copy()
#     formula_products = []
#     if not comp.empty:
#         # each completion row has its FORMULA_ID and INVENTORY_ITEM_ID (FG)
#         fp = (comp.groupby(["FORMULA_ID","INVENTORY_ITEM_ID"])
#                  .size().reset_index(name="batch_count")
#                  .sort_values(["FORMULA_ID","batch_count"], ascending=[True, False]))
#         # pack top 5 products per formula
#         for f, grp in fp.groupby("FORMULA_ID"):
#             tops = grp.head(5)[["INVENTORY_ITEM_ID","batch_count"]]
#             formula_products.append({
#                 "FORMULA_ID": f,
#                 "TOP_PRODUCTS": json_safe(tops)
#             })
#
#     payload = {
#         "status": "ok",
#         "formulas": json_safe(form_sum),
#         "details": json_safe(std.sort_values(["FORMULA_ID","INGREDIENT_ID"])),
#         "formula_to_top_products": formula_products
#     }
#     return JSONResponse(content=json_safe(payload))
#
#
# # --- Diagnostics (sanity check) ---
# @app.get("/formula-vs-actual/diagnostics")
# def formula_vs_actual_diagnostics():
#     global df
#     base = _normalize_columns_data_aware(df)
#
#     by_type = (base.groupby("TRANSACTION_TYPE_NAME")
#                     .size().reset_index(name="rows")
#                     .sort_values("rows", ascending=False))
#
#     pos_issue = base[(base["TRANSACTION_TYPE_NAME"]=="WIP Issue") & (base["WIP_QTY"].notna())]
#     pos_ret   = base[(base["TRANSACTION_TYPE_NAME"]=="WIP Return") & (base["WIP_QTY"].notna())]
#
#     return JSONResponse(content=json_safe({
#         "status": "ok",
#         "rows": int(len(base)),
#         "by_transaction_type": json_safe(by_type),
#         "nonzero_issue_qty_rows": int((pos_issue["WIP_QTY"] != 0).sum()),
#         "nonzero_return_qty_rows": int((pos_ret["WIP_QTY"] != 0).sum())
#     }))
# # ========= END BLOCK =========
#
#

# ========= FORMULA (standard) vs ACTUAL (from batch_details) =========
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder



# --- Config / constants ---
EPS = float(os.getenv("EPS", 1e-6))                 # tolerance for zero-ish comparisons
DELAY_THRESHOLD_DAYS = int(os.getenv("DELAY_THRESHOLD_DAYS", 2))
FORMULA_XLSX_PATH = os.getenv("FORMULA_XLSX_PATH", "formula.xlsx")
BATCH_XLSX_PATH   = os.getenv("BATCH_XLSX_PATH",   "batch_details.xlsx")

# --- Load once (reuse your existing df if it matches the same file) ---
try:
    _batch_df_for_formula = pd.read_excel(BATCH_XLSX_PATH)
except Exception:
    # fallback: if you already have a global df from earlier in the file, use it
    _batch_df_for_formula = df.copy() if 'df' in globals() else pd.DataFrame()

try:
    _formula_df = pd.read_excel(FORMULA_XLSX_PATH)
except Exception:
    _formula_df = pd.DataFrame()

# --- Helpers: standard and actual builders ---
def _build_standard_from_formula(formula_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build standard BOM per (FORMULA_ID, INVENTORY_ITEM_ID).
    Uses QTY as STD_QTY and STD_RATE (unit cost). Falls back if missing.
    """
    if formula_df.empty:
        return pd.DataFrame(columns=["FORMULA_ID","INGREDIENT_ID","STD_QTY","STD_UNIT_COST","STD_ING_COST"])
    std = formula_df.copy()
    std["INGREDIENT_ID"] = std["INVENTORY_ITEM_ID"]
    # choose rate: STD_RATE preferred; else STD_RATE_WITH_LOSS if available
    rate = std["STD_RATE"] if "STD_RATE" in std.columns else std.get("STD_RATE_WITH_LOSS")
    std["__STD_RATE__"] = pd.to_numeric(rate, errors="coerce")
    std["__STD_QTY__"]  = pd.to_numeric(std["QTY"], errors="coerce")
    g = (std.groupby(["FORMULA_ID","INGREDIENT_ID"], dropna=False)
             .agg(STD_QTY=("__STD_QTY__", "median"),
                  STD_UNIT_COST=("__STD_RATE__", "median"))
             .reset_index())
    g["STD_ING_COST"] = g["STD_QTY"] * g["STD_UNIT_COST"]
    return g

def _build_actual_from_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build actual consumption per (BATCH_ID, FORMULA_ID, INGREDIENT_ID) from Issue/Return.
    CONS_QTY: Issues (-WIP_QTY) + Returns (-WIP_QTY but WIP_QTY will be negative for return rows in some ERPs).
    CONS_COST: Issue cost positive; Return cost negative.
    """
    if batch_df.empty:
        return pd.DataFrame(columns=["BATCH_ID","FORMULA_ID","INGREDIENT_ID","ACT_QTY","ACT_COST","ACT_UNIT_COST_MED"])
    req_cols = {"TRANSACTION_TYPE_NAME","WIP_QTY","WIP_VALUE","WIP_RATE","WIP_BATCH_ID","FORMULA_ID","INVENTORY_ITEM_ID"}
    if not req_cols.issubset(set(batch_df.columns)):
        return pd.DataFrame(columns=["BATCH_ID","FORMULA_ID","INGREDIENT_ID","ACT_QTY","ACT_COST","ACT_UNIT_COST_MED"])

    ir = batch_df[batch_df["TRANSACTION_TYPE_NAME"].isin(["WIP Issue","WIP Return"])].copy()

    ir["INGREDIENT_ID"] = ir["INVENTORY_ITEM_ID"]
    # Quantities: convert to consumed positive where possible
    ir["CONS_QTY"] = -pd.to_numeric(ir["WIP_QTY"], errors="coerce")
    # Costs: Issue positive, Return negative
    ir["CONS_COST"] = np.where(
        ir["TRANSACTION_TYPE_NAME"].eq("WIP Issue"),
        pd.to_numeric(ir["WIP_VALUE"], errors="coerce"),
        -pd.to_numeric(ir["WIP_VALUE"], errors="coerce")
    )
    # Unit rate: prefer WIP_RATE; fallback value/qty
    ir["UNIT_RATE"] = pd.to_numeric(ir["WIP_RATE"], errors="coerce")
    need_rate = ir["UNIT_RATE"].isna() & ir["WIP_QTY"].ne(0)
    ir.loc[need_rate, "UNIT_RATE"] = (ir.loc[need_rate, "WIP_VALUE"].abs() /
                                      ir.loc[need_rate, "WIP_QTY"].abs().replace(0, np.nan))

    act = (ir.groupby(["WIP_BATCH_ID","FORMULA_ID","INGREDIENT_ID"], dropna=False)
             .agg(ACT_QTY=("CONS_QTY","sum"),
                  ACT_COST=("CONS_COST","sum"),
                  ACT_UNIT_COST_MED=("UNIT_RATE", lambda s: s.dropna().median() if s.notna().any() else np.nan))
             .reset_index()
             .rename(columns={"WIP_BATCH_ID":"BATCH_ID"}))
    return act

def _compare_formula_vs_actual(act: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the standard to each batch size and compute line-level variances + status buckets.
    """
    if (act is None or std is None) or (act.empty and std.empty):
        return pd.DataFrame()

    # totals for scaling standard per formula to each batch
    std_tot = std.groupby("FORMULA_ID", dropna=False)["STD_QTY"].sum().reset_index().rename(columns={"STD_QTY":"STD_TOTAL_QTY"})
    act_tot = act.groupby(["BATCH_ID","FORMULA_ID"], dropna=False)["ACT_QTY"].sum().reset_index().rename(columns={"ACT_QTY":"ACT_TOTAL_QTY"})
    scale  = act_tot.merge(std_tot, on="FORMULA_ID", how="left")
    scale["BATCH_SCALE"] = np.where(scale["STD_TOTAL_QTY"].abs() > 0, scale["ACT_TOTAL_QTY"]/scale["STD_TOTAL_QTY"], np.nan)

    d = (act.merge(scale[["BATCH_ID","FORMULA_ID","BATCH_SCALE"]], on=["BATCH_ID","FORMULA_ID"], how="left")
            .merge(std[["FORMULA_ID","INGREDIENT_ID","STD_QTY","STD_UNIT_COST"]],
                   on=["FORMULA_ID","INGREDIENT_ID"], how="outer"))

    # math
    d["STD_QTY_SCALED"]      = d["STD_QTY"] * d["BATCH_SCALE"]
    d["STD_ING_COST_SCALED"] = d["STD_QTY_SCALED"] * d["STD_UNIT_COST"]
    d["ACT_ING_COST_FINAL"]  = d["ACT_QTY"] * d["ACT_UNIT_COST_MED"]
    d["QTY_VAR"]  = d["ACT_QTY"] - d["STD_QTY_SCALED"]
    d["COST_VAR"] = d["ACT_ING_COST_FINAL"] - d["STD_ING_COST_SCALED"]

    # robust bucketing (EPS tolerance)
    a0 = pd.to_numeric(d["ACT_QTY"], errors="coerce").fillna(0.0)
    s0 = pd.to_numeric(d["STD_QTY_SCALED"], errors="coerce").fillna(0.0)

    excess  = (s0 <= EPS) & (a0 > EPS)                          # ACT_ONLY
    unused  = (s0 > EPS)  & (a0 <= EPS)                         # STD_ONLY
    changed = (s0 > EPS)  & (a0 > EPS) & ((a0 - s0).abs() > EPS)

    d["STATUS"] = np.select(
        [excess, unused, changed],
        ["EXCESS_OR_NEW (ACT_ONLY)", "NOT_USED (STD_ONLY)", "QTY_CHANGED"],
        default="MATCH"
    )
    return d

def _rollup_by_formula(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame(columns=[
            "FORMULA_ID","STD_COST","ACT_COST","COST_VAR","VARIANCE_%",
            "EXCESS_CNT","UNUSED_CNT","CHANGED_CNT","UNCHANGED_CNT","BATCHES_TOUCHED"
        ])
    d = detail.copy()
    for c in ["STD_ING_COST_SCALED","ACT_ING_COST_FINAL","COST_VAR"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # batch coverage per formula
    batches_touched = d.groupby("FORMULA_ID")["BATCH_ID"].nunique().rename("BATCHES_TOUCHED")
    # counts
    counts = (d.groupby(["FORMULA_ID","STATUS"]).size().unstack(fill_value=0)
                .rename(columns={
                    "EXCESS_OR_NEW (ACT_ONLY)":"EXCESS_CNT",
                    "NOT_USED (STD_ONLY)":"UNUSED_CNT",
                    "QTY_CHANGED":"CHANGED_CNT",
                    "MATCH":"UNCHANGED_CNT"
                }))

    # costs
    sums = (d.groupby("FORMULA_ID", dropna=False)
              .agg(STD_COST=("STD_ING_COST_SCALED", lambda s: s.sum(min_count=1)),
                   ACT_COST=("ACT_ING_COST_FINAL",  lambda s: s.sum(min_count=1)),
                   COST_VAR=("COST_VAR", lambda s: s.sum(min_count=1))))

    roll = (sums.join(counts, how="left").join(batches_touched, how="left")).fillna(0)
    roll["VARIANCE_%"] = np.where(roll["STD_COST"].abs() > 0,
                                  (roll["COST_VAR"] / roll["STD_COST"]) * 100,
                                  np.nan)
    roll = roll.reset_index()
    return roll

# --- Endpoint: overview for charts (per FORMULA_ID) ---
@app.get("/formula-diff/overview")
def formula_diff_overview():
    std = _build_standard_from_formula(_formula_df)
    act = _build_actual_from_batch(_batch_df_for_formula)
    detail = _compare_formula_vs_actual(act, std)
    roll = _rollup_by_formula(detail)

    if roll.empty:
        return JSONResponse(content={
            "status": "ok",
            "formulas": 0,
            "items": [],
            "ai_insights": "No data available to compare standard vs actual."
        })

    # Top formulas by absolute cost variance
    top = roll.assign(_abs=np.abs(roll["COST_VAR"])) \
              .sort_values("_abs", ascending=False) \
              .drop(columns="_abs") \
              .head(20)

    # Graph-friendly packs
    out = {
        "status": "ok",
        "formulas": int(roll["FORMULA_ID"].nunique()),
        "items": top.to_dict(orient="records"),
        "stacked_counts": {
            "FORMULA_ID": roll["FORMULA_ID"].astype(str).tolist(),
            "EXCESS_CNT": roll.get("EXCESS_CNT", pd.Series([0]*len(roll))).astype(int).tolist(),
            "UNUSED_CNT": roll.get("UNUSED_CNT", pd.Series([0]*len(roll))).astype(int).tolist(),
            "CHANGED_CNT": roll.get("CHANGED_CNT", pd.Series([0]*len(roll))).astype(int).tolist(),
            "UNCHANGED_CNT": roll.get("UNCHANGED_CNT", pd.Series([0]*len(roll))).astype(int).tolist()
        },
        "ai_insights": """
# Formula vs Actual (Overview)

The comparison highlights how production batches differ from the defined standard formula.

**Key metrics shown:**
- Standard vs Actual cost, with variance in both absolute ($) and percentage terms.  
- Counts of ingredients that were excess (used but not in the standard), unused (in the standard but not used), changed (used in different quantity), or unchanged.  
- Number of batches impacted for each formula.

**Suggested visualizations:**
1. Horizontal bar chart of formulas ranked by highest cost variance.  
2. Stacked bar chart showing the mix of Excess / Unused / Changed / Unchanged ingredients.  
3. Scatterplot of Standard Cost vs Cost Variance, with bubble size representing changed ingredients.

This helps quickly identify where the biggest cost overruns or recipe deviations are happening and which formulas are most at risk.
"""
    }
    return JSONResponse(content=jsonable_encoder(out))


# --- Endpoint: drilldown for a single formula (per-ingredient diffs) ---
@app.get("/formula-diff/formula")
def formula_diff_for_formula():
    std = _build_standard_from_formula(_formula_df)
    act = _build_actual_from_batch(_batch_df_for_formula)
    detail = _compare_formula_vs_actual(act, std)

    if detail.empty:
        return JSONResponse(content={"status":"ok","items": [], "ai_insights": "No rows found."})

    # Aggregate per ingredient across all formulas
    agg = (detail.groupby(["FORMULA_ID","INGREDIENT_ID","STATUS"], dropna=False)
              .agg(STD_QTY_SCALED=("STD_QTY_SCALED","mean"),
                   ACT_QTY=("ACT_QTY","mean"),
                   QTY_VAR=("QTY_VAR","mean"),
                   STD_UNIT_COST=("STD_UNIT_COST","median"),
                   ACT_UNIT_COST_MED=("ACT_UNIT_COST_MED","median"),
                   STD_COST=("STD_ING_COST_SCALED","sum"),
                   ACT_COST=("ACT_ING_COST_FINAL","sum"),
                   COST_VAR=("COST_VAR","sum"),
                   BATCHES=("BATCH_ID","nunique"))
              .reset_index())

    # Top ingredients across all formulas
    top_ing = agg.assign(_abs=np.abs(agg["COST_VAR"])) \
                 .sort_values("_abs", ascending=False) \
                 .drop(columns="_abs") \
                 .head(50)

    tornado = {
        "INGREDIENT_ID": top_ing["INGREDIENT_ID"].astype(str).tolist(),
        "COST_VAR": top_ing["COST_VAR"].round(2).tolist(),
        "STATUS": top_ing["STATUS"].astype(str).tolist()
    }
    slope_qty = {
        "INGREDIENT_ID": top_ing["INGREDIENT_ID"].astype(str).tolist(),
        "STD_QTY_SCALED": (top_ing["STD_QTY_SCALED"].fillna(0)).round(4).tolist(),
        "ACT_QTY": (top_ing["ACT_QTY"].fillna(0)).round(4).tolist()
    }

    out = {
        "status": "ok",
        "ingredients": top_ing.to_dict(orient="records"),
        "charts": {
            "tornado_cost_var": tornado,
            "slope_qty": slope_qty
        },
        "ai_insights": """
# Ingredient-Level Drilldown (All Formulas)

This endpoint aggregates ingredient performance across every formula and batch.  
It identifies which raw materials are most responsible for cost variance and usage deviations.

- **Tornado Chart**: Top 50 ingredients ranked by absolute cost variance, with color coding for Excess, Unused, or Changed status.  
- **Slope Chart (Quantities)**: Compares average standard vs actual consumption for each ingredient, showing where usage consistently deviates.  
- Use this drilldown to spot high-impact ingredients that repeatedly drive cost overruns across multiple formulas.
"""
    }
    return JSONResponse(content=jsonable_encoder(out))


############################

# =========================
# Config for chart endpoints
# =========================
TOP_N = int(os.getenv("TOP_N", "25"))  # used by /charts/* endpoints

# =========================
# Small numeric helpers
# =========================
def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _round_list(s, nd=2, fill_zero=True):
    ser = _num(s)
    if fill_zero:
        ser = ser.fillna(0)
    if nd is not None:
        ser = ser.round(nd)
    return ser.tolist()

def _tolist_or_none(s, nd=2):
    """Round but preserve None for NaNs."""
    ser = _num(s)
    if nd is not None:
        ser = ser.round(nd)
    return [None if pd.isna(v) else float(v) for v in ser]

# =========================
# Utilities for these charts
# =========================
def _build_portfolio_tornado(detail: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Aggregate across ALL formulas & batches; pick the status per ingredient with max |COST_VAR|.
    Robust to NaNs and empty groups.
    """
    cols = ["INGREDIENT_ID", "STATUS", "COST_VAR", "STD_QTY_SCALED", "ACT_QTY"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=cols)

    agg = (
        detail.groupby(["INGREDIENT_ID", "STATUS"], dropna=False)
              .agg(COST_VAR=("COST_VAR", "sum"),
                   STD_QTY_SCALED=("STD_QTY_SCALED", "mean"),
                   ACT_QTY=("ACT_QTY", "mean"))
              .reset_index()
    )
    if agg.empty:
        return pd.DataFrame(columns=cols)

    # Choose a single row per ingredient by max |COST_VAR| (treat NaN as 0 to avoid idxmax errors)
    agg["_abs"] = _num(agg["COST_VAR"]).abs().fillna(0)
    idx = agg.groupby("INGREDIENT_ID")["_abs"].idxmax()
    pick = agg.loc[idx].drop(columns="_abs")

    # Sort by absolute variance desc and cap to top_n
    pick["_abs"] = _num(pick["COST_VAR"]).abs().fillna(0)
    pick = pick.sort_values("_abs", ascending=False).drop(columns="_abs")
    return pick.head(max(1, int(top_n)))

def _top_formulas_abs(roll: pd.DataFrame, top_n: int) -> pd.DataFrame:
    base_cols = ["FORMULA_ID", "COST_VAR", "STD_COST", "ACT_COST", "VARIANCE_%"]
    if roll is None or roll.empty:
        return pd.DataFrame(columns=base_cols)
    x = roll.copy()
    # Ensure numeric columns are numeric
    for c in ["COST_VAR", "STD_COST", "ACT_COST", "VARIANCE_%"]:
        if c in x.columns:
            x[c] = _num(x[c])
    x["_abs"] = _num(x["COST_VAR"]).abs().fillna(0)
    x = x.sort_values("_abs", ascending=False).drop(columns="_abs")
    # Keep only expected cols if present
    keep = [c for c in base_cols if c in x.columns]
    x = x[keep] if keep else x
    return x.head(max(1, int(top_n)))

def _status_mix_top(roll: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Return EXCESS/UNUSED/CHANGED/MATCH counts for the top |variance| formulas.
    Ensures missing count columns are added with zeros.
    """
    cols = ["FORMULA_ID", "EXCESS_CNT", "UNUSED_CNT", "CHANGED_CNT", "UNCHANGED_CNT"]
    if roll is None or roll.empty:
        return pd.DataFrame(columns=cols)

    x = roll.copy()
    x["_abs"] = _num(x["COST_VAR"]).abs().fillna(0)
    x = x.sort_values("_abs", ascending=False).drop(columns="_abs").head(max(1, int(top_n)))

    for c in cols[1:]:
        if c not in x.columns:
            x[c] = 0
    # enforce integer counts
    for c in cols[1:]:
        x[c] = _num(x[c]).fillna(0).astype(int)
    return x[cols]

def _ensure_frames():
    """Build std/act/detail/roll once per request from globals or configured paths."""
    std = _build_standard_from_formula(_formula_df)
    act = _build_actual_from_batch(_batch_df_for_formula)
    detail = _compare_formula_vs_actual(act, std)
    roll = _rollup_by_formula(detail)
    return std, act, detail, roll

# =========================
# 1) Portfolio Tornado (NO inputs)
# =========================
@app.get("/charts/portfolio-tornado", tags=["BOM Variance"])
def charts_portfolio_tornado_no_inputs():
    _, _, detail, _ = _ensure_frames()
    pick = _build_portfolio_tornado(detail, top_n=TOP_N)

    payload = {
        "title": "Portfolio Tornado: Top Ingredients by Cost Variance (All Formulas)",
        "top_n": TOP_N,
        "ingredient_ids": pick.get("INGREDIENT_ID", pd.Series(dtype=object)).astype(str).tolist(),
        "cost_variance": _round_list(pick.get("COST_VAR", pd.Series(dtype=float)), nd=2),
        "status": pick.get("STATUS", pd.Series(dtype=object)).astype(str).tolist(),
        "std_qty_scaled": _round_list(pick.get("STD_QTY_SCALED", pd.Series(dtype=float)), nd=6),
        "act_qty": _round_list(pick.get("ACT_QTY", pd.Series(dtype=float)), nd=6),
        "table": jsonable_encoder(pick.replace({np.nan: None}).to_dict(orient="records")),
    }
    return JSONResponse(content=payload)

# =========================
# 2) Top Formulas by Total Cost Variance (NO inputs)
# =========================
@app.get("/charts/top-formulas-variance", tags=["BOM Variance"])
def charts_top_formulas_variance_no_inputs():
    _, _, _, roll = _ensure_frames()
    top = _top_formulas_abs(roll, top_n=TOP_N)

    # variance_pct: preserve None for NaNs
    variance_pct = _tolist_or_none(top.get("VARIANCE_%", pd.Series([None] * len(top))), nd=2)

    payload = {
        "title": "Top Formulas by Total Cost Variance (Actual − Standard)",
        "top_n": TOP_N,
        "formula_ids": top.get("FORMULA_ID", pd.Series(dtype=object)).astype(str).tolist(),
        "cost_variance": _round_list(top.get("COST_VAR", pd.Series(dtype=float)), nd=2),
        "std_cost": _round_list(top.get("STD_COST", pd.Series(dtype=float)), nd=2),
        "act_cost": _round_list(top.get("ACT_COST", pd.Series(dtype=float)), nd=2),
        "variance_pct": variance_pct,
        "table": jsonable_encoder(top.replace({np.nan: None}).to_dict(orient="records")),
    }
    return JSONResponse(content=payload)

# =========================
# 3) Status Mix for Top Formulas (NO inputs)
# =========================
@app.get("/charts/status-mix", tags=["BOM Variance"])
def charts_status_mix_no_inputs():
    _, _, _, roll = _ensure_frames()
    x = _status_mix_top(roll, top_n=TOP_N)

    payload = {
        "title": "Ingredient Status Mix per Formula (Top by |Variance|)",
        "top_n": TOP_N,
        "formula_ids": x.get("FORMULA_ID", pd.Series(dtype=object)).astype(str).tolist(),
        "excess_cnt": x.get("EXCESS_CNT", pd.Series([0] * len(x))).astype(int).tolist(),
        "unused_cnt": x.get("UNUSED_CNT", pd.Series([0] * len(x))).astype(int).tolist(),
        "changed_cnt": x.get("CHANGED_CNT", pd.Series([0] * len(x))).astype(int).tolist(),
        "unchanged_cnt": x.get("UNCHANGED_CNT", pd.Series([0] * len(x))).astype(int).tolist(),
        "table": jsonable_encoder(x.replace({np.nan: None}).to_dict(orient="records")),
    }
    return JSONResponse(content=payload)




from chatbot import query_bot

class QueryRequest(BaseModel):
    query: str



@app.post("/CHATBOT")
async def bot_chat(request: QueryRequest):
    result = await query_bot(request.query)
    return result



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
