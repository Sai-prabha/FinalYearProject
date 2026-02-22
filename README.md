# BTC/ETH Ratio Trading Project

Final Year Project: Machine Learning-based BTC/ETH ratio trading strategy with live model server and React dashboard.

## Project Structure

```
FinalYearProject/
├── FrontEnd/              # React + Vite dashboard for live trading signals
├── api/                   # FastAPI model server for real-time signal streaming
├── src/                   # Core Python modules (features, models, backtest)
├── scripts/               # Utility scripts and runners
├── data/                  # Data files (gitignored, structure preserved)
├── reports/               # Generated reports and analysis
├── documentation/         # Project documentation and guides
├── notebooks/             # Jupyter notebooks
├── archive/               # Deprecated files
├── start_model_server.py  # Model server entry point
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Clone and Fetch Latest Changes

If pulling from GitLab:

```bash
# Clone the repository
git clone https://gitlab.cs.nuim.ie/u230520/final-year-project.git
cd final-year-project

# Or fetch latest changes if already cloned
git fetch origin
git pull origin main
```

### 2. Python Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Model Server

The model server connects to Binance WebSocket, calculates features, and streams trading signals.

```bash
# From project root
python start_model_server.py
```

The server will be available at:
- **API:** http://localhost:8888
- **Status:** http://localhost:8888/status
- **WebSocket:** ws://localhost:8888/ws/signals

### 4. Run Frontend Dashboard

In a separate terminal:

```bash
cd FrontEnd

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The dashboard will be available at: **http://localhost:5173**

### 5. Verify Setup

Test the model server status:

```bash
curl http://localhost:8888/status
```

Expected response:
```json
{
  "status": "healthy",
  "service": "trading-model-server",
  "timestamp": "..."
}
```

## Development Workflow

### Running v4 Model Training & Backtest

```bash
# Prepare data
python src/prepare_data.py

# Train and backtest
python src/train_and_backtest.py

# Or use a specific version
python scripts/run_v4_unified.py
```

### Running Optimizations

```bash
# Optuna hyperparameter tuning
python scripts/optuna_tune_v4.py

# Monitor optimization progress
bash scripts/monitor_optuna.sh

# Check specific trial results
bash scripts/show_latest_trial.sh
```

### Data Quality Checks

```bash
# Run data quality control
python scripts/data_quality_control.py

# Check for overfitting
python scripts/check_overfitting.py
```

## Key Documentation

- **[Model Integration Guide](documentation/MODEL_INTEGRATION_README.md)** - Detailed model server setup
- **[Quick Start Guide](documentation/QUICKSTART_MODEL_INTEGRATION.md)** - Fast integration walkthrough  
- **[GitLab Pull Instructions](documentation/GITLAB_PULL_INSTRUCTIONS.md)** - How to pull and merge changes
- **[Frontend README](FrontEnd/README.md)** - Frontend architecture and components
- **[v4 Complete Guide](documentation/V4_COMPLETE_GUIDE.md)** - v4 model documentation
- **[Project Overview](documentation/PROJECT_OVERVIEW.md)** - High-level project summary

## Technology Stack

### Backend
- **Python 3.9+** with scikit-learn, XGBoost, TensorFlow
- **FastAPI** + **Uvicorn** for model server
- **WebSockets** for real-time data streaming
- **Pandas** + **NumPy** for data processing

### Frontend
- **React 19** with TypeScript
- **Vite** for fast development and builds
- **TailwindCSS** for styling
- **Lightweight Charts** for candlestick visualization

### Data Sources
- Binance WebSocket API (BTC/USDT, ETH/USDT)
- Historical parquet data for backtesting

## Model Versions

- **v4.x** - Current production model (XGBoost-based, ratio features, Z-score strategy)
- **v5** - Experimental model (cross-exchange features, hybrid approach)

See `reports/` for detailed backtest results and performance metrics.

## Project Status

✅ **v4 Model:** Trained and optimized (see reports/4.11/, reports/4.12/)  
✅ **Backend API:** FastAPI model server with live signal streaming  
✅ **Frontend Dashboard:** React UI for monitoring live signals  
✅ **Data Pipeline:** Automated data collection and processing  
🔄 **v5 Model:** Under development (cross-exchange integration)

## Backup & Safety

A backup branch was created before the latest integration:

```bash
# View backup
git log backup/local-main-before-master-merge-20260205

# Restore backup if needed (CAUTION: will lose current work)
git checkout backup/local-main-before-master-merge-20260205
git checkout -b restored-from-backup
```

## Troubleshooting

### Model Server Won't Start

```bash
# Check if dependencies are installed
pip list | grep -E "fastapi|uvicorn|websockets"

# Reinstall if needed
pip install -r requirements.txt

# Check if port 8888 is in use
lsof -i :8888  # On Mac/Linux
netstat -ano | findstr :8888  # On Windows
```

### Frontend Won't Start

```bash
cd FrontEnd

# Clean install
rm -rf node_modules package-lock.json
npm install

# Check if port 5173 is in use
lsof -i :5173  # On Mac/Linux
```

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/FinalYearProject

# Activate virtual environment
source .venv/bin/activate

# Verify Python can find modules
python -c "import src.models; print('OK')"
python -c "from api.model_server import app; print('OK')"
```

## Contact

For questions or issues, refer to the documentation in `documentation/` or check recent reports in `reports/`.

---

**Last Updated:** February 5, 2026  
**Integration Status:** GitLab master (FrontEnd + v5 api) successfully merged into local main

