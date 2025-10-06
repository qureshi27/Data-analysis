from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import json
import os
import io
import contextlib
import traceback
import base64
import re

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional OpenAI import (handled gracefully if not installed)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(title="Engro Alarm Analytics API", version="1.0.0")

# CORS configuration for React frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage (in production, use proper database/cache)
class DataStore:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.last_upload: Optional[datetime] = None

data_store = DataStore()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    sample_rows: int = Field(default=5, ge=1, le=20)
    include_code: bool = True
    include_context: bool = False

class FilterRequest(BaseModel):
    date_range: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    show_alarms_only: bool = False

class AnalyticsResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: float

class DataUploadResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict[str, Any]]

# Utility functions
def load_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataframe with necessary columns and transformations"""
    if 'Event Time' not in df.columns:
        # Attempt to find a plausible datetime column fallback
        for cand in ['timestamp', 'time', 'event_time', 'Datetime', 'Date']:
            if cand in df.columns:
                df.rename(columns={cand: 'Event Time'}, inplace=True)
                break
    if 'Event Time' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an 'Event Time' column or equivalent")

    df['Event Time'] = pd.to_datetime(df['Event Time'], errors='coerce')
    if df['Event Time'].isna().all():
        raise HTTPException(status_code=400, detail="Failed to parse 'Event Time' as datetime")

    df['Hour'] = df['Event Time'].dt.hour
    df['Minute'] = df['Event Time'].dt.minute
    df['Date'] = df['Event Time'].dt.date

    if 'Action' not in df.columns:
        df['Action'] = 'NO ACTION'
    else:
        df['Action'] = df['Action'].fillna('NO ACTION')

    if 'Condition' not in df.columns:
        df['Condition'] = 'UNKNOWN'

    # Create alarm categories
    alarm_conditions = ['ALARM', 'PVHIHI', 'PVHI', 'PVLO', 'PVLOW', 'PVLOLOW',
                        'HIHI', 'HI', 'LO', 'LOLO', 'FAIL']
    df['Is_Alarm'] = df['Condition'].astype(str).str.upper().str.contains(
        '|'.join(alarm_conditions), na=False
    )

    if 'Source' not in df.columns:
        df['Source'] = 'UNKNOWN'

    return df

def apply_filters(df: pd.DataFrame, filters: FilterRequest) -> pd.DataFrame:
    """Apply filters to dataframe"""
    filtered_df = df.copy()

    if filters.date_range and len(filters.date_range) == 2:
        start_date = pd.to_datetime(filters.date_range[0]).date()
        end_date = pd.to_datetime(filters.date_range[1]).date()
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) &
            (filtered_df['Date'] <= end_date)
        ]

    if filters.sources:
        filtered_df = filtered_df[filtered_df['Source'].isin(filters.sources)]

    if filters.conditions:
        filtered_df = filtered_df[filtered_df['Condition'].isin(filters.conditions)]

    if filters.show_alarms_only:
        filtered_df = filtered_df[filtered_df['Is_Alarm'] == True]

    return filtered_df

def build_ai_context(df: pd.DataFrame, max_numeric_cols: int = 6,
                     max_cat_cols: int = 6, sample_rows: int = 5) -> str:
    """Build context for AI from dataframe"""
    if df is None or len(df) == 0:
        return "No data available."

    lines = []
    lines.append("DATAFRAME OVERVIEW")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("\nCOLUMNS:")

    for col in list(df.columns)[:50]:
        dtype = str(df[col].dtype)
        nonnull = int(df[col].notna().sum())
        lines.append(f"- {col}: {dtype}, non-null={nonnull:,}")

    # Numeric summaries
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        lines.append("\nNUMERIC SUMMARIES:")
        for col in num_cols[:max_numeric_cols]:
            desc = df[col].describe()
            mean = float(desc.get('mean', np.nan))
            std = float(desc.get('std', np.nan))
            minv = float(desc.get('min', np.nan))
            maxv = float(desc.get('max', np.nan))
            lines.append(f"- {col}: mean={mean:.2f}, std={std:.2f}, min={minv:.2f}, max={maxv:.2f}")

    # Categorical summaries
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        lines.append("\nCATEGORICAL TOP VALUES:")
        for col in cat_cols[:max_cat_cols]:
            vc = df[col].value_counts(dropna=True).head(5)
            compact = ", ".join([f"{idx} ({cnt})" for idx, cnt in vc.items()])
            lines.append(f"- {col}: {compact}")

    # Sample rows
    sample = df.head(sample_rows)
    lines.append(f"\nSAMPLE ROWS (first {len(sample)}):")
    sample_cp = sample.copy()
    for c in sample_cp.columns:
        if np.issubdtype(sample_cp[c].dtype, np.datetime64):
            sample_cp[c] = sample_cp[c].dt.strftime('%Y-%m-%d %H:%M:%S')
    lines.append(sample_cp.to_csv(index=False))

    return "\n".join(lines)

def ask_openai(question: str, context: str, code_mode: bool = False) -> str:
    """Query OpenAI for analysis or code generation"""
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=500, detail="OpenAI package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        client = OpenAI(api_key=api_key)

        if code_mode:
            system_prompt = (
                "You are a data analyst writing Python for alarm data analysis. "
                "Return ONLY a Python code block that: "
                "1) Analyzes the DataFrame df, "
                "2) Assigns results to 'result' variable, "
                "3) Optionally creates a Plotly figure as 'fig'. "
                "Do not use print(), show() or any I/O operations."
            )
            user_prompt = f"Context:\n{context}\n\nTask: {question}"
        else:
            system_prompt = (
                "You are an industrial alarm system analyst. Provide concise, "
                "actionable insights based on the data analysis results."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")

def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response supporting fenced blocks."""
    if not text:
        return ""

    # Prefer ```python ... ``` blocks
    pattern_py = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern_py.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: any fenced code
    pattern_any = re.compile(r"```\s*(.*?)```", re.DOTALL)
    match = pattern_any.search(text)
    if match:
        return match.group(1).strip()

    return ""

def run_code_sandbox(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Execute code in sandboxed environment"""
    safe_builtins = {
        'len': len, 'range': range, 'min': min, 'max': max, 'sum': sum,
        'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'abs': abs,
        'round': round, 'any': any, 'all': all
    }

    globals_dict = {
        '__builtins__': safe_builtins,
        'pd': pd, 'np': np, 'px': px, 'go': go, 'datetime': datetime
    }

    locals_dict = {'df': df}
    stdout_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, globals_dict, locals_dict)

        return {
            'success': True,
            'result': locals_dict.get('result'),
            'fig': locals_dict.get('fig'),
            'stdout': stdout_buf.getvalue(),
            'error': None
        }
    except Exception:
        return {
            'success': False,
            'result': None,
            'fig': None,
            'stdout': stdout_buf.getvalue(),
            'error': traceback.format_exc(limit=2)
        }

def format_result(result: Any) -> Dict[str, Any]:
    """Format execution result for API response"""
    if result is None:
        return {"type": "none", "data": None}

    if isinstance(result, pd.DataFrame):
        return {
            "type": "dataframe",
            "data": result.to_dict('records'),
            "columns": list(result.columns),
            "shape": result.shape
        }
    elif isinstance(result, pd.Series):
        return {
            "type": "series",
            "data": result.to_dict(),
            "name": result.name
        }
    elif isinstance(result, (int, float, str, bool)):
        return {
            "type": "scalar",
            "data": result
        }
    else:
        return {
            "type": "object",
            "data": str(result)
        }

# API Endpoints
@app.post("/api/upload", response_model=DataUploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file"""
    try:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Prepare data
        df = load_and_prepare_data(df)

        # Store in memory
        data_store.df = df
        data_store.last_upload = datetime.now()

        # Calculate stats
        stats = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "date_range": {
                "start": df['Event Time'].min().isoformat() if not pd.isna(df['Event Time'].min()) else None,
                "end": df['Event Time'].max().isoformat() if not pd.isna(df['Event Time'].max()) else None
            },
            "unique_sources": int(df['Source'].nunique()),
            "total_alarms": int(df['Is_Alarm'].sum()),
            "columns": list(df.columns)
        }

        return DataUploadResponse(
            success=True,
            message=f"Successfully uploaded {file.filename}",
            stats=stats
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=AnalyticsResponse)
async def query_data(request: QueryRequest):
    """Process natural language query on data"""
    start_time = datetime.now()

    try:
        if data_store.df is None:
            raise HTTPException(status_code=400, detail="No data uploaded")

        # Apply filters if provided
        df = data_store.df
        if request.filters:
            filter_req = FilterRequest(**request.filters)
            df = apply_filters(df, filter_req)

        # Build context
        context = build_ai_context(df, sample_rows=request.sample_rows)

        # Generate code
        code_response = ask_openai(request.question, context, code_mode=True)
        code = extract_python_code(code_response)

        if not code:
            raise HTTPException(status_code=500, detail="Failed to generate code")

        # Execute code
        execution = run_code_sandbox(code, df)

        # Format results
        response_data = {
            "question": request.question,
            "code": code if request.include_code else None,
            "context": context if request.include_context else None,
            "execution": {
                "success": execution['success'],
                "result": format_result(execution['result']),
                "stdout": execution['stdout'],
                "error": execution['error']
            }
        }

        # Convert Plotly figure to JSON if present
        if execution['fig'] is not None:
            try:
                response_data['execution']['figure'] = json.loads(
                    pio.to_json(execution['fig'])
                )
            except Exception:
                # If figure serialization fails, ignore silently
                pass

        # Generate prescriptive summary
        if execution['success']:
            summary_context = f"""
            Question: {request.question}
            Result: {execution['result']}
            """
            summary = ask_openai(
                "Provide actionable insights and recommendations",
                summary_context,
                code_mode=False
            )
            response_data['summary'] = summary

        execution_time = (datetime.now() - start_time).total_seconds()

        return AnalyticsResponse(
            success=True,
            data=response_data,
            error=None,
            execution_time=execution_time
        )

    except HTTPException as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        return AnalyticsResponse(
            success=False,
            data=None,
            error=str(e.detail) if hasattr(e, 'detail') else str(e),
            execution_time=execution_time
        )
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        return AnalyticsResponse(
            success=False,
            data=None,
            error=str(e),
            execution_time=execution_time
        )

@app.get("/api/statistics")
async def get_statistics(filters: Optional[str] = None):
    """Get basic statistics from loaded data"""
    try:
        if data_store.df is None:
            raise HTTPException(status_code=400, detail="No data uploaded")

        df = data_store.df

        # Apply filters if provided
        if filters:
            filter_dict = json.loads(filters)
            filter_req = FilterRequest(**filter_dict)
            df = apply_filters(df, filter_req)

        # Calculate statistics
        stats = {
            "overview": {
                "total_events": int(len(df)),
                "total_alarms": int(df['Is_Alarm'].sum()),
                "alarm_percentage": float(df['Is_Alarm'].mean() * 100) if len(df) else 0.0,
                "unique_sources": int(df['Source'].nunique()),
                "unique_conditions": int(df['Condition'].nunique()),
                "acknowledged_count": int(df['Action'].astype(str).str.contains('ACK', na=False).sum())
            },
            "top_sources": {str(k): int(v) for k, v in df['Source'].value_counts().head(10).to_dict().items()},
            "top_conditions": {str(k): int(v) for k, v in df['Condition'].value_counts().head(10).to_dict().items()},
            "hourly_distribution": {int(k): int(v) for k, v in df['Hour'].value_counts().sort_index().to_dict().items()},
            "value_statistics": {
                "mean": float(df['Value'].mean()) if 'Value' in df.columns else None,
                "median": float(df['Value'].median()) if 'Value' in df.columns else None,
                "std": float(df['Value'].std()) if 'Value' in df.columns else None,
                "min": float(df['Value'].min()) if 'Value' in df.columns else None,
                "max": float(df['Value'].max()) if 'Value' in df.columns else None
            }
        }

        return {"success": True, "data": stats}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metadata")
async def get_metadata():
    """Get metadata about loaded data"""
    if data_store.df is None:
        return {
            "loaded": False,
            "message": "No data uploaded"
        }

    return {
        "loaded": True,
        "upload_time": data_store.last_upload.isoformat() if data_store.last_upload else None,
        "shape": data_store.df.shape,
        "columns": list(map(str, data_store.df.columns)),
        "sources": list(map(str, data_store.df['Source'].unique())) if 'Source' in data_store.df.columns else [],
        "conditions": list(map(str, data_store.df['Condition'].unique())) if 'Condition' in data_store.df.columns else []
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_available": OPENAI_AVAILABLE,
        "data_loaded": data_store.df is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
