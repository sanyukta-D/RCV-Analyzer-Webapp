# RCV Election Analyzer - Web Application

A simple web interface for analyzing Ranked Choice Voting elections.

## Quick Start

### Local Development

1. Install dependencies:
```bash
cd Optimal_Strategies_in_RCV
pip install -r requirements.txt
pip install -r webapp/requirements.txt
```

2. Run the app:
```bash
streamlit run webapp/app.py
```

3. Open http://localhost:8501 in your browser

### Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo and set:
   - Main file path: `webapp/app.py`
5. Click "Deploy"

That's it! Your app will be live at `https://your-app.streamlit.app`

## Features

- **Upload any RCV election data** (CSV format)
- **Automatic format detection** (Choice_1/rank1 formats)
- **Social choice order** - See the final ranking of candidates
- **Victory gap analysis** - How many votes each candidate needs to win
- **Ballot exhaustion** - Track exhausted ballots through rounds
- **Strategy computation** - Optimal vote-addition strategies
- **Export results** - Download analysis as CSV

## Data Format

Your CSV should have columns like:

**Standard format:**
```csv
Choice_1,Choice_2,Choice_3
Alice,Bob,Charlie
Bob,Alice,
Charlie,Alice,Bob
```

**Alternative format:**
```csv
rank1,rank2,rank3
Alice,Bob,Charlie
Bob,Alice,
Charlie,Alice,Bob
```

## Configuration

Use the sidebar to configure:
- **Number of Winners**: 1 for single-winner (Mayor), higher for multi-winner (City Council)
- **Budget %**: Maximum additional votes to consider for strategy analysis
- **Max candidates**: Limit analysis for large elections (>9 candidates)
