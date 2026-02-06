# RCV Election Analyzer

A web application for analyzing Ranked Choice Voting elections, built on the research from:

- *Optimal Strategies in Ranked Choice Voting*
- *Simpler Than You Think: The Practical Dynamics of Ranked Choice Voting*

**Live app:** [rcv-analyzer.streamlit.app](https://rcv-analyzer.streamlit.app)

## What It Does

Upload any RCV election CSV and get:

1. **Victory Gap & Margin of Victory** — How many additional votes does each candidate need to win?
2. **Ballot Exhaustion Impact** — Could completing exhausted ballots change the outcome? Includes 6 probability models from the paper.
3. **Strategic Complexity** — Are optimal strategies simple (self-support) or complex (support rivals)?
4. **Preference Order Alignment** — Does the elimination order reflect true competitiveness?

## Run Locally

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

## Data Format

Your CSV should have columns in one of these formats:
- `Choice_1, Choice_2, Choice_3, ...`
- `rank1, rank2, rank3, ...`

Each row is one ballot. Cell values are candidate names. Empty cells for unranked positions.

Have raw Cast Vote Records? Use [FairVote's RCV Cruncher](https://github.com/fairvotereform/rcv_cruncher) to convert them — its default output is directly compatible.

## Built-in Examples

- Alaska 2022 US House Special Election
- Alaska 2020 Presidential
- Minneapolis 2021 Mayor
- San Francisco 2011 Mayor
- Burlington 2009 Mayor
- Portland 2024 Districts 1-4 (multi-winner, k=3)

## Project Structure

```
webapp/app.py                  — Streamlit web application
rcv_strategies/                — Core RCV analysis library
  core/stv_irv.py              — STV and IRV election simulation
  core/strategy.py             — Optimal strategy computation
  core/candidate_removal.py    — Irrelevant candidate removal
  core/optimization.py         — Linear optimization for vote addition
  utils/helpers.py             — Ballot manipulation utilities
  utils/case_study_helpers.py  — Analysis pipeline helpers
ballot_exhaustion/             — Probability models for ballot completion
  probability_models.py        — 6 models (3 Beta, 3 Bootstrap)
case_studies/                  — Example election data
```

## Full Repository

The complete research codebase (with all datasets, case studies, notebooks, and analysis scripts) is at [sanyukta-D/Optimal_Strategies_in_RCV](https://github.com/sanyukta-D/Optimal_Strategies_in_RCV).

## License

MIT
