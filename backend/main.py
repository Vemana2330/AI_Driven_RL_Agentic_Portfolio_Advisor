# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from workflow.workflow_portfolio import PortfolioWorkflow

# Instantiate workflow once (better performance)
workflow = PortfolioWorkflow()

app = FastAPI(
    title="RL Agentic Portfolio System",
    description="Multi-Agent + Reinforcement Learning Portfolio Optimization AI",
    version="1.0.0"
)

# ----------------------------
# CORS (Frontend Streamlit / React)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Input Schema
# ----------------------------
class PortfolioRequest(BaseModel):
    investment_amount: float = Field(..., gt=0, description="Total amount to invest")
    risk_level: str = Field(
        default="medium",
        description="Investor risk tolerance",
        pattern="^(low|medium|high)$"
    )
    duration_months: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Investment duration horizon"
    )

# ----------------------------
# POST — Run Portfolio Workflow
# ----------------------------
@app.post("/run-portfolio-analysis")
def run_portfolio(req: PortfolioRequest):
    response = workflow.run(
        investment_amount=req.investment_amount,
        risk_level=req.risk_level,
        duration_months=req.duration_months
    )
    return {
        "api_version": "1.0.0",
        "request_summary": {
            "investment_amount": req.investment_amount,
            "risk_level": req.risk_level,
            "duration_months": req.duration_months
        },
        "response": response
    }

# ----------------------------
# GET — Health Check
# ----------------------------
@app.get("/")
def root():
    return {"message": "RL-Agentic Portfolio System API is running 🚀"}
