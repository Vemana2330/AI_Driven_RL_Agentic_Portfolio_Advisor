# AI-Driven RL Agentic Portfolio Advisor

## Overview

The **AI-Driven RL Agentic Portfolio Advisor** is a multi-agent, reinforcement learning–enhanced financial analysis system.

It combines:
- **Deep Reinforcement Learning** (DQN + PPO)
- **Multi-agent intelligence** (Sentiment, Risk, Macro, Technical, Fundamentals, Performance)
- A **FastAPI backend**
- A **Streamlit frontend**

to generate investment strategies, portfolio allocations, and market intelligence summaries tailored to a user-selected investor profile.

This project satisfies the **Building Agentic Systems** assignment by demonstrating:
- Multi-agent orchestration
- Tool integration
- Reinforcement learning feedback loops
- Persistent agent scoring
- Complete end-to-end system execution

---

## Key Features

### Multi-Agent Architecture
Controller Agent orchestrates 10 specialized agents, ensuring clean workflow execution.

### Reinforcement Learning Portfolio Allocation
DQN Agent + PPO Policy Agent determine optimal allocations based on signals.

### Market Intelligence Suite
Sentiment analysis, macroeconomic evaluation, technical indicators, fundamentals scoring, and trend classification.

### Risk Analysis Engine
Volatility, drawdowns, and risk-profile matching.

### Comprehensive Streamlit UI
Includes dashboards, charts, RL actions, trend tables, and full AI-generated reports.

### FastAPI Backend
Processes requests, orchestrates agents, and returns structured JSON outputs.

---

## System Architecture
<img width="2408" height="1358" alt="System_Architecture_Portfolio" src="https://github.com/user-attachments/assets/8c4a3c0b-7897-451c-a519-1fa296fe1ce9" />

### 1. Frontend (Streamlit)
- Investor profile input
- One-click portfolio analysis
- Tabs for metrics, RL decisions, market intelligence, and full reports

### 2. Backend (FastAPI)
- Receives analysis requests
- Triggers the Controller Agent
- Runs RL, sentiment, macro, technical, and risk evaluations
- Returns structured response to frontend

### 3. Multi-Agent Workflow
- Controller Agent manages entire pipeline
- Specialists analyze sentiment, fundamentals, trends, volatility, and macro conditions
- RL agents finalize portfolio construction
- Report Generator Agent produces natural-language summaries

### 4. Reinforcement Learning Engine
- **DQN Agent**: Action-value learning for optimal allocations
- **PPO Policy Agent**: Phase-based allocation strategy
- **LLM-Based Feedback**: Reward signals for each agent
- **Rule-Based Feedback**: Penalizes failures in workflow

---

##  Project Structure
```
RL-AGENTIC-PORTFOLIO-SYSTEM/
│
├── agents/
│   ├── rl_agents/
│   │   ├── __init__.py
│   │   ├── rl_agent_dqn.py
│   │   ├── rl_agent_policy.py
│   │
│   │
│   ├── controller_agent.py
│   ├── macro_agent.py
│   ├── performance_agent.py
│   ├── risk_manager_agent.py
│   ├── sentiment_agent.py
│   ├── summary_report_agent.py
│   ├── technical_analysis_agent.py
│   ├── __init__.py
│   
│
├── backend/
│   ├── main.py                      # FastAPI backend entrypoint
│   
│
├── data/
│   ├── external/
│   └── raw/
│       ├── ai_stocks.csv
│       ├── benchmark_qqq.csv
│       └── risk_free_rate.csv
│
├── frontend/
│   ├── app.py                       # Streamlit UI
│   ├── .streamlit/config.toml
│   
│   
│
├── models/
│   ├── rl_model_v1/                 # Trained DQN model
│   ├── rl_policy_model/             # Trained Policy Gradient model
│   ├── model_definition.py
│   ├── __init__.py
│   
│
├── reinforcement_learning/
│   ├── feedback_manager.py          # RL Feedback Loop
│   ├── __init__.py
│   
│
├── rlenv/                           # Virtual Environment
│   
│
├── tools/
│   ├── tool_base.py
│   ├── built_in_tools.py
│   ├── custom_tools.py
│   ├── phase_policy_inference.py
│   ├── portfolio_allocator_agent_inference.py
│   ├── __init__.py
│   
│
├── utils/
│   ├── agent_utils.py
│   ├── __init__.py
│   
│
├── workflow/
│   ├── workflow_portfolio.py        # Main workflow executed by Controller
│   ├── __init__.py
│   
│
├── logs/
│   
│
├── config.py
├── download_data.py
├── logging_utils.py
├── rl_memory_portfolio.json         # RL Experience replay memory
│
├── README.md
├── requirements.txt
│
├── .env

```

---

## Quick Start Guide

### Get Running in 5 Minutes

#### Step 1: Install Dependencies
```bash
# Activate virtual environment
source rlenv/bin/activate       # macOS/Linux
# or
rlenv\Scripts\activate          # Windows

# Install all required packages
pip install -r requirements.txt
```

#### Step 2: Set Up Environment Variables

Create a `.env` file in project root:
```env
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

**Where to get keys:**
- OpenAI → https://platform.openai.com/api-keys
- Tavily → https://tavily.com/
- FRED → https://fred.stlouisfed.org/docs/api/api_key.html

#### Step 3: Verify System Setup
```bash
python setup_and_test.py
```

You should see all tests marked with green checkmarks.

#### Step 4: Start the Application

**Option A: Start everything at once (recommended)**
```bash
./start_all.sh
```

**Option B: Start backend and frontend separately**

Terminal 1:
```bash
./start_backend.sh
```

Terminal 2:
```bash
./start_frontend.sh
```

#### Step 5: Open the UI

Go to: http://localhost:8501

1. Set your investor profile
2. Press " Run Portfolio Intelligence Analysis"
3. Wait 30–60 seconds
4. Explore all dashboard tabs

---

## Application Workflow

### 1. Key Metrics Dashboard
- Return %
- Volatility
- Drawdown
- Final portfolio value

### 2. Portfolio Overview
- DQN allocations
- Portfolio growth curve
- Asset breakdowns

### 3. RL Agent Decisions
- DQN action distribution
- PPO phase allocations
- Reward summaries

### 4. Market Intelligence
- Sentiment scores
- Macro regime analysis
- Bullish/bearish breakdowns

### 5. Risk Analysis
- Portfolio risk score
- Recommended vs not-recommended tickers
- Volatility & drawdown tables

### 6. Technical Analysis
- RSI, momentum, trend direction
- Entry zone identification

### 7. Fundamentals
- PE, PB, ROE, revenue growth
- Fundamental scorecards

### 8. Full AI Report
- End-to-end natural-language summary

---

## Troubleshooting

### "Module Not Found"
```bash
pip install -r requirements.txt
```

### Backend not connecting

Run:
```bash
curl http://127.0.0.1:8000/
```

- Ensure backend is running
- Make sure port 8000 is free

### API Key errors
- `.env` must exist in project root
- Confirm key names are correct

### Import errors
- Activate your virtual environment
- Run from project root
- Re-run `python setup_and_test.py`

---

## Agents Included

| Agent | Purpose |
|-------|---------|
| Controller Agent | Orchestrates entire workflow |
| Sentiment Agent | Bullish/bearish scoring |
| Macro Agent | Economic regime classification |
| Risk Manager Agent | Drawdown, volatility, profile matching |
| Performance Agent | Financial KPIs and fundamentals |
| Technical Agent | RSI, momentum, trends |
| DQN RL Agent | Optimal allocation strategy |
| PPO Policy Agent | Phase-based policy allocation |
| Summary Report Agent | Final natural-language summary |

---

## Reinforcement Learning Feedback Loop

### Two Layers of RL Feedback

#### Rule-Based Feedback
- `+1` for valid agent output
- `-1` for failures or missing data

#### LLM-Based Evaluation
- Rewards based on clarity, completeness, and usefulness

Feedback is stored persistently and shown in the UI.

---
## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
2. Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518, 529–533.
3. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347.
4. Silver, D. et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature, 529, 484–489.
5. Fama, E. F. (1970). “Efficient Capital Markets: A Review of Theory and Empirical Work.” The Journal of Finance, 25(2), 383–417.
6. Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77–91.
7. Bodie, Z., Kane, A., & Marcus, A. (2014). Investments (10th ed.). McGraw-Hill.
8. Yahoo Finance API. Market Data Retrieval. https://finance.yahoo.com/
9. Vaswani, A. et al. (2017). "Attention Is All You Need." Advances in Neural Information Processing Systems.
10. CrewAI Documentation. Agentic Framework for Multi-Agent Systems. https://docs.crewai.com/
11. Streamlit Documentation. Application Development Framework. https://docs.streamlit.io/
12. OpenAI API Documentation. LLM-Based Reasoning and Evaluation. https://platform.openai.com/docs
13. Amodei, D. et al. (2016). “Concrete Problems in AI Safety.” arXiv preprint arXiv:1606.06565.
14. ISO 31000 (2018). Risk Management – Guidelines. International Organization for Standardization.

## License

This project is released under the MIT License, which permits unrestricted use, modification, and distribution of the software. Users are free to use the code for personal, academic, or commercial purposes as long as the license notice is included in any copies or substantial portions of the software.

MIT License
The software is provided free of charge and allows anyone to:
Use the software for any purpose
Copy and modify the source code
Distribute the software or its modified versions
Include the software in commercial applications
The license also states that:
The original copyright notice and permission notice must be included.
The software is provided "as is" without warranties of any kind.
The authors are not liable for any damages or claims arising from the use of the software.

---

## 👨‍💻 Author

**Vemana** (002475002) 
Northeastern University
