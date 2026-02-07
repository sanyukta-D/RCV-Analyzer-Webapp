"""
RCV Election Analyzer - Web Application

A simple web interface for analyzing Ranked Choice Voting elections
using the algorithms from the Optimal Strategies in RCV research.

Based on:
- "Optimal Strategies in Ranked Choice Voting"
- "Simpler Than You Think: The Practical Dynamics of Ranked Choice Voting"

Run with: streamlit run webapp/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from string import ascii_uppercase, ascii_lowercase
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RCV analysis functions
from rcv_strategies.core.stv_irv import IRV_optimal_result, IRV_ballot_exhaust, STV_ballot_exhaust, STV_optimal_result_simple
from rcv_strategies.core.candidate_removal import remove_irrelevant
from rcv_strategies.constants import MAX_TRACTABLE_CANDIDATES
from rcv_strategies.utils.helpers import get_new_dict, return_main_sub
from rcv_strategies.utils.case_study_helpers import (
    get_ballot_counts_df,
    process_ballot_counts_post_elim_no_print
)

# Import probability models for ballot exhaustion analysis (6 models from paper)
try:
    from ballot_exhaustion.probability_models import (
        # Single-winner models (IRV)
        beta_probability,           # Gap-Based Beta
        direct_posterior_beta,      # Similarity Beta
        prior_posterior_beta,       # Prior-Posterior Beta
        category_based_bootstrap,   # Similarity Bootstrap
        limited_ranking_bootstrap,  # Rank-Restricted Bootstrap
        unconditional_bootstrap,    # Unconditional Bootstrap
        # Multi-winner models (STV) - compare candidate vs ALL active candidates
        analyze_preference_patterns_multi_winner,
        beta_probability_multi_winner,
        similarity_beta_multi_winner,
        prior_posterior_beta_multi_winner,
        category_bootstrap_multi_winner,
        limited_ranking_bootstrap_multi_winner,
        unconditional_bootstrap_multi_winner
    )
    PROB_MODELS_AVAILABLE = True
except ImportError as e:
    PROB_MODELS_AVAILABLE = False
    print(f"Probability models not available: {e}")

# === CLEANUP ZOMBIE WORKERS ON STREAMLIT RERUN ===
# When Streamlit reruns (widget change, navigation), kill any leftover pool workers
# from a previous run. This runs at the TOP of every rerun before any analysis starts.
from rcv_strategies.core.strategy import _cleanup_pool
_cleanup_pool()

# === PAGE CONFIG ===
st.set_page_config(
    page_title="RCV Election Analyzer",
    page_icon="🗳️",
    layout="wide",
)

# === CUSTOM CSS (matching paper colors) ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        color: white;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
<div class="main-header">
    <h1>🗳️ RCV Election Analyzer</h1>
    <p>Computational analysis of ranked choice voting dynamics</p>
</div>
""", unsafe_allow_html=True)

# === INTRODUCTORY SECTIONS FOR NEW USERS ===

with st.expander("What is Ranked Choice Voting?", expanded=False):
    st.markdown("""
**Ranked Choice Voting (RCV)** is an electoral system where voters rank candidates
in order of preference instead of choosing just one.

**How it works (single-winner / IRV):**
1. Each voter ranks candidates: 1st choice, 2nd choice, 3rd choice, etc.
2. If no candidate has a majority of first-choice votes, the candidate with the
   fewest votes is eliminated.
3. Voters who ranked that candidate first have their votes transferred to their
   next-ranked choice.
4. This process repeats until one candidate reaches a majority and wins.

**Multi-winner elections (STV):**
When multiple seats are being filled (e.g., a city council), the system uses
**Single Transferable Vote (STV)**. Candidates who exceed a winning threshold
(called the **Droop quota**) are elected, and their surplus votes are transferred
proportionally. The process continues, eliminating the lowest candidates and
transferring votes, until all seats are filled.

**Ballot exhaustion** occurs when all of a voter's ranked choices have been
eliminated, so their ballot can no longer be counted in subsequent rounds.

**Droop quota** (shown in the results): the minimum number of votes needed to
guarantee a seat. It equals floor(total votes / (seats + 1)) + 1. For a
single-winner race, this simplifies to a simple majority (> 50%).
    """)

with st.expander("What does this tool analyze?", expanded=False):
    st.markdown("""
This tool evaluates four key attributes of an RCV election:

**1. Victory Gap & Competitiveness**
How many additional votes would each candidate need to win? A small gap means the
race was close; a large gap means the outcome was decisive.

**2. Ballot Exhaustion Impact**
Could the outcome have changed if voters who exhausted their ballots had ranked
more candidates? Six statistical models estimate this probability.

**3. Strategic Complexity**
Is the best path to victory simply getting more of your own supporters to vote
(a "selfish" strategy), or does it require more complex maneuvering like boosting
a weaker rival to split the opposition (a "non-selfish" strategy)?

**4. Preference Order Alignment**
Does the order in which candidates were eliminated match how close they actually
were to winning? When these align, the election results are straightforward
to interpret.

    """)

# === HELPER FUNCTIONS ===

# Color scheme from the paper
CATEGORY_COLORS = {
    "Winner": {"bg": "rgb(189, 223, 167)", "hex": "#bddfa7"},           # Soft mint green
    "Near Winner": {"bg": "rgb(223, 240, 216)", "hex": "#dff0d8"},      # Very light green
    "Contender": {"bg": "rgb(253, 245, 206)", "hex": "#fdf5ce"},        # Pale cream/yellow
    "Competitive": {"bg": "rgb(253, 231, 208)", "hex": "#fde7d0"},      # Soft peach
    "Distant": {"bg": "rgb(248, 218, 205)", "hex": "#f8dacd"},          # Light salmon
    "Far Behind": {"bg": "rgb(242, 201, 198)", "hex": "#f2c9c6"},       # Muted red/pink
    "Beyond Threshold": {"bg": "rgb(220, 220, 220)", "hex": "#dcdcdc"}, # Light gray - strategy not computed
}

def detect_format(df):
    """Detect the format of the uploaded CSV."""
    cols = df.columns.tolist()
    if any(col.startswith('Choice_') for col in cols):
        return 'choice'
    elif any(col.startswith('rank') for col in cols):
        return 'rank'
    return None

def convert_rank_to_choice(df):
    """Convert rank1, rank2... format to Choice_1, Choice_2... format."""
    rename_map = {}
    for col in df.columns:
        if col.startswith('rank') and col[4:].isdigit():
            rename_map[col] = f"Choice_{col[4:]}"
    return df.rename(columns=rename_map)

def get_candidates_from_df(df):
    """Extract unique candidates from the dataframe."""
    choice_cols = [col for col in df.columns if col.startswith('Choice_')]
    all_candidates = set()
    for col in choice_cols:
        candidates = df[col].dropna().unique()
        all_candidates.update(candidates)

    exclude = {'', 'skipped', 'overvote', 'undervote', 'writein', 'exhausted', 'nan', 'none'}

    def should_exclude(name):
        """Check if candidate should be excluded (write-ins, special values, etc.)"""
        name_lower = str(name).strip().lower()
        # Exclude exact matches
        if name_lower in exclude:
            return True
        # Exclude write-in candidates (various formats)
        if 'write-in' in name_lower or 'write in' in name_lower or 'writein' in name_lower:
            return True
        return False

    candidates = [c for c in all_candidates
                  if not should_exclude(c)
                  and pd.notna(c)
                  and str(c).strip() != '']

    # Convert all to strings for consistency
    candidates = [str(c).strip() for c in candidates]

    return sorted(set(candidates), key=str)

def categorize_gap(gap, k=1):
    """
    Categorize a candidate based on their victory gap (from paper).
    Thresholds are scaled for multi-winner: k=1 uses base thresholds,
    k=3 uses half (normalized from 50% quota to 25% quota).
    """
    # Scale factor: for k=1, quota ~50%; for k=3, quota ~25% (half)
    # Thresholds for k=1: 5, 20, 30, 45
    # Thresholds for k=3: 2.5, 10, 15, 22.5 (scaled by quota ratio)
    scale = 2 / (k + 1)  # For k=1: 1.0, for k=3: 0.5

    if gap == 0:
        return "Winner"
    elif gap <= 5 * scale:
        return "Near Winner"
    elif gap <= 20 * scale:
        return "Contender"
    elif gap <= 30 * scale:
        return "Competitive"
    elif gap <= 45 * scale:
        return "Distant"
    else:
        return "Far Behind"

def is_selfish_strategy(strategy_detail, candidate_code):
    """Check if strategy is selfish (only self-support) or non-selfish."""
    if not strategy_detail:
        return True  # Default to selfish if no detail
    # Strategy is selfish if all votes go to self
    for cand, votes in strategy_detail.items():
        if cand != candidate_code and votes > 0:
            return False
    return True

def compute_preference_order_alignment(results, strategies):
    """
    Check if Social Choice Order matches Victory Gap Order.
    Returns: (matches, victory_gap_order, mismatches)
    """
    # Get victory gaps for each candidate
    gap_data = []
    for code in results:
        strat = strategies.get(code, None)
        if strat and isinstance(strat, (list, tuple)) and len(strat) > 0:
            gap = strat[0]
        else:
            gap = float('inf')
        gap_data.append((code, gap))

    # Sort by victory gap to get Victory Gap Order
    victory_gap_order = [x[0] for x in sorted(gap_data, key=lambda x: x[1])]

    # Check alignment
    matches = results == victory_gap_order

    # Find mismatches
    mismatches = []
    for i, (sco, vgo) in enumerate(zip(results, victory_gap_order)):
        if sco != vgo:
            mismatches.append((i+1, sco, vgo))

    return matches, victory_gap_order, mismatches

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## Settings")

    k = st.number_input(
        "Number of Winners",
        min_value=1, max_value=10,
        value=st.session_state.get('pending_k', 1),
        help="Set to 1 for single-winner elections (Mayor, Governor)"
    )

    budget_percent = st.slider(
        "Budget / Allowance (% of total votes)",
        0.0, 100.0,
        value=st.session_state.get('pending_budget', 10.0),
        step=0.5,
        help="Maximum additional votes to consider for strategy analysis (algorithmic tradeability threshold)"
    )

    with st.expander("Advanced Options"):
        keep_at_least = st.slider(
            "Keep at least (candidates)",
            3, 20,
            value=st.session_state.get('pending_keep', 7),
            help="Minimum candidates to retain after removal. Lower = faster. Portland uses 7-8 for k=3."
        )
        rigorous_check = st.checkbox("Rigorous candidate removal", value=True)
        check_strategies = st.checkbox("Compute optimal strategies", value=True)
        max_rankings = st.number_input(
            "Max Rankings (for Rank-Restricted Bootstrap)",
            min_value=3, max_value=20, value=5,
            help="Maximum rankings allowed per ballot. NYC=5, Portland=6. "
                 "If no rank limit is set by the election, set this to the number of candidates "
                 "(treats all exhausted ballots as completable). Used by the Rank-Restricted Bootstrap model."
        )

    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    Based on:
    - *Optimal Strategies in Ranked Choice Voting*
    - *Simpler Than You Think: The Practical Dynamics of RCV*

    [GitHub](https://github.com/sanyukta-D/Optimal_Strategies_in_RCV)
    """)

# === MAIN CONTENT ===

# File upload
st.markdown("## Upload Election Data")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose your election CSV file",
        type=["csv"],
        help="Upload a CSV with ranked choice voting data"
    )

with col2:
    with st.expander("📋 Accepted File Formats", expanded=False):
        st.markdown("""
        **Column naming (one of):**
        - `Choice_1, Choice_2, Choice_3, ...`
        - `rank1, rank2, rank3, ...`

        **Structure:**
        - One row per ballot
        - Each column = one rank position
        - Cell values = candidate names
        - Empty cells for unranked positions

        **Example:**
        | Choice_1 | Choice_2 | Choice_3 |
        |----------|----------|----------|
        | Alice    | Bob      | Carol    |
        | Bob      | Alice    |          |
        | Carol    | Bob      | Alice    |

        **Notes:**
        - Extra columns (RowNumber, ID) are ignored
        - Values like 'skipped', 'overvote', 'undervote', 'writein' are excluded

        **Have raw Cast Vote Records (CVR)?**
        Use [FairVote's RCV Cruncher](https://github.com/fairvotereform/rcv_cruncher)
        to parse and convert CVR files. Its default "rank" format output
        (`rank1, rank2, ...`) is directly compatible with this tool.
        """)
    use_example = st.checkbox("Use example data")

# Load example data if requested
if use_example and uploaded_file is None:
    base_path = Path(__file__).parent.parent / "case_studies"
    examples_path = base_path / "examples"

    # Curated collection of interesting elections with metadata
    # Format: display_name -> (file_path, k, budget, keep_at_least)
    curated_examples = {}

    # === HIGH-PROFILE SINGLE-WINNER ELECTIONS ===
    single_winner_files = {
        "Alaska 2022 US House Special": ("Alaska_08162022_HouseofRepresentativesSpecial.csv", 1, 10.0, 7),
        "Alaska 2020 Presidential": ("Alaska_04102020_PRESIDENTOFTHEUNITEDSTATES.csv", 1, 40.0, 7),
        "Minneapolis 2021 Mayor": ("Minneapolis_20211102_Mayor.csv", 1, 10.0, 7),
        "San Francisco 2011 Mayor": ("SanFrancisco_20111108_Mayor.csv", 1, 10.0, 7),
        "Burlington 2009 Mayor": ("Burlington_20090303_Mayor.csv", 1, 40.0, 7),
    }

    for name, (filename, ex_k, ex_budget, ex_keep) in single_winner_files.items():
        filepath = examples_path / filename
        if filepath.exists():
            curated_examples[name] = (filepath, ex_k, ex_budget, ex_keep)

    # === MULTI-WINNER ELECTIONS (Portland k=3) ===
    portland_path = base_path / "portland" / "data"
    portland_configs = {
        "Portland 2024 District 1 (k=3)": ("Dis_1/Election_results_dis1.csv", 3, 4.5, 8),
        "Portland 2024 District 2 (k=3)": ("Dis_2/Election_results_dis2.csv", 3, 6.5, 8),
        "Portland 2024 District 3 (k=3)": ("Dis_3/Election_results_dis3.csv", 3, 13.0, 8),
        "Portland 2024 District 4 (k=3)": ("Dis_4/Election_results_dis4.csv", 3, 9.5, 8),
    }

    for name, (rel_path, ex_k, ex_budget, ex_keep) in portland_configs.items():
        filepath = portland_path / rel_path
        if filepath.exists():
            curated_examples[name] = (filepath, ex_k, ex_budget, ex_keep)

    if curated_examples:
        # Group examples by category for better UX
        example_names = list(curated_examples.keys())

        selected_example = st.selectbox(
            "Select example election",
            options=example_names,
            help="Single-winner (k=1) or multi-winner Portland (k=3)"
        )

        filepath, rec_k, rec_budget, rec_keep = curated_examples[selected_example]
        uploaded_file = filepath

        # Detect example change and update pending values for sidebar
        prev_example = st.session_state.get('_prev_example', None)
        if prev_example != selected_example:
            # Example changed - set pending values and rerun
            st.session_state['_prev_example'] = selected_example
            st.session_state['pending_k'] = rec_k
            st.session_state['pending_budget'] = rec_budget
            st.session_state['pending_keep'] = rec_keep
            st.rerun()  # Force rerun to update sidebar with new values

        if rec_k > 1:
            st.info(f"💡 **Recommended settings:** k={rec_k}, Budget={rec_budget}%, Keep at least={rec_keep}")
        else:
            st.info(f"💡 **Single-winner election.** Default settings should work well.")

# Process uploaded file
if uploaded_file is not None:
    try:
        # Load data
        if isinstance(uploaded_file, Path):
            df = pd.read_csv(uploaded_file)
            file_name = uploaded_file.name
        else:
            df = pd.read_csv(uploaded_file)
            file_name = uploaded_file.name

        # Detect and convert format
        data_format = detect_format(df)
        if data_format == 'rank':
            df = convert_rank_to_choice(df)
        elif data_format is None:
            st.error("Could not detect data format.")
            st.markdown(f"""
            **Your columns:** `{', '.join(df.columns[:10])}`{'...' if len(df.columns) > 10 else ''}

            **Expected formats:**
            - `Choice_1, Choice_2, Choice_3, ...` (Portland/NYC style)
            - `rank1, rank2, rank3, ...` (alternative style)

            Please rename your columns to match one of these formats.
            """)
            st.stop()

        # Data preview
        st.markdown("## Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ballots", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            choice_cols = [c for c in df.columns if c.startswith('Choice_')]
            st.metric("Ranking Depth", len(choice_cols))

        with st.expander("View Raw Data"):
            st.dataframe(df.head(10), use_container_width=True)

        # Detect candidates
        candidates = get_candidates_from_df(df)

        if len(candidates) == 0:
            st.error("No candidates detected in the data.")
            choice_cols = [c for c in df.columns if c.startswith('Choice_')]
            st.markdown(f"**Choice columns found:** {choice_cols}")
            if choice_cols:
                st.markdown(f"**Sample values in {choice_cols[0]}:** {df[choice_cols[0]].dropna().unique()[:10].tolist()}")
            st.stop()
        elif len(candidates) > 52:
            st.error(f"Too many candidates ({len(candidates)}). Maximum supported is 52.")
            st.stop()
        elif len(candidates) < k:
            st.error(f"Not enough candidates ({len(candidates)}) for {k} winner(s). Need at least {k} candidates.")
            st.stop()
        elif len(candidates) == 1:
            st.warning(f"Only 1 candidate detected - they win by default.")
            st.metric("Winner", candidates[0])
            st.stop()
        elif len(candidates) == k:
            st.warning(f"Number of candidates ({len(candidates)}) equals number of winners ({k}) - all candidates win by default.")
            st.markdown("**Winners:** " + ", ".join(candidates))
            st.stop()
        else:
            st.success(f"Detected {len(candidates)} candidates")

        # Create candidate mapping
        candidates_mapping = {name: (ascii_uppercase + ascii_lowercase)[i] for i, name in enumerate(candidates)}
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}

        with st.expander("Candidate Mapping"):
            mapping_df = pd.DataFrame([
                {"Letter": v, "Candidate": k}
                for k, v in candidates_mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        # Run analysis
        if st.button("Run Analysis", type="primary", use_container_width=True):

            progress = st.progress(0)
            status = st.empty()

            try:
                # ============================================================
                # WEBAPP ANALYSIS PIPELINE
                # ============================================================
                #
                # Overview:
                # 1. Convert CSV to ballot_counts with arbitrary letter mapping
                # 2. Run STV to get social choice order (winner first)
                # 3. REMAP so A=winner, B=runner-up, etc. (intuitive display)
                # 4. Run STV again on remapped data
                # 5. Compute strategies using process_ballot_counts_post_elim_no_print
                #
                # For large elections (> 8 candidates):
                # - Strategy computation is intractable
                # - Use binary search to find highest budget where removal works
                # - remove_irrelevant() reduces to tractable set (< 9 candidates)
                #
                # For multi-winner (k > 1):
                # - May encounter "early winners" exceeding quota after removal
                # - Uses "small election method" with k-1 seats
                # - See case_study_helpers.py for detailed logic
                # ============================================================

                # Step 1: Initial ballot counts with alphabetical mapping
                status.text("Converting ballots...")
                progress.progress(10)

                initial_mapping = candidates_mapping.copy()
                initial_ballot_counts = get_ballot_counts_df(initial_mapping, df)
                total_votes = sum(initial_ballot_counts.values())
                initial_candidates_list = list(initial_mapping.values())

                if total_votes == 0:
                    st.error("No valid ballots found. Check your data format.")
                    st.stop()

                # Step 2: First STV run to determine social choice order
                status.text("Determining social choice order...")
                progress.progress(20)

                Q = round(total_votes / (k + 1) + 1, 3)
                if k == 1:
                    Q = Q * (k + 1)

                rt_initial, dt_initial, _ = STV_optimal_result_simple(
                    initial_candidates_list, initial_ballot_counts, k, Q
                )
                initial_results, _ = return_main_sub(rt_initial)

                # Step 3: CRITICAL - Remap so Winner=A, Runner-up=B, etc.
                status.text("Remapping candidates by social choice order...")
                progress.progress(30)

                # Get candidate names in winning order
                initial_reverse = {v: k for k, v in initial_mapping.items()}
                ordered_candidate_names = [initial_reverse[code] for code in initial_results]

                # Create final mapping: Winner→A, Runner-up→B, etc.
                final_mapping = {name: (ascii_uppercase + ascii_lowercase)[i] for i, name in enumerate(ordered_candidate_names)}
                reverse_mapping = {v: k for k, v in final_mapping.items()}

                # Rebuild ballot counts with social-choice-based mapping
                ballot_counts = get_ballot_counts_df(final_mapping, df)
                candidates_list = list(final_mapping.values())  # Now A=winner, B=runner-up, etc.

                # Step 4: Second STV run on remapped data
                status.text("Running comprehensive RCV analysis...")
                progress.progress(40)

                rt2, dt2, _ = STV_optimal_result_simple(candidates_list, ballot_counts, k, Q)
                results_alphabetical, _ = return_main_sub(rt2)

                # Step 5: Run strategy analysis
                status.text("Computing optimal strategies...")
                progress.progress(60)

                effective_keep_at_least = keep_at_least
                max_for_strats = MAX_TRACTABLE_CANDIDATES - 1  # TRACTABILITY LIMIT: < MAX_TRACTABLE_CANDIDATES for exact strategies
                n_candidates = len(candidates_list)

                # ============================================================
                # DIVIDE-AND-CONQUER: Finding Optimal Budget Threshold
                # ============================================================
                #
                # PROBLEM: With many candidates (>= MAX_TRACTABLE_CANDIDATES), direct strategy computation
                # is intractable. We need remove_irrelevant() to reduce the set,
                # but removal depends on budget - higher budget = more removal.
                #
                # SOLUTION: Binary search for the highest budget where:
                # 1. remove_irrelevant() succeeds (stop=True)
                # 2. retained candidates <= max_for_strats (tractable)
                #
                # Two-Phase Approach:
                # Phase 1: Try full candidate set (works for most elections)
                #          Portland Dis 1,2,3 succeed with this phase
                # Phase 2: If Phase 1 fails (removal too aggressive), pre-filter
                #          to top N candidates and retry. Portland Dis 4 needs this.
                #
                # The computed_threshold is then used for final strategy computation.
                # ============================================================
                effective_bc = ballot_counts
                effective_cands = candidates_list

                computed_threshold = budget_percent

                if n_candidates > max_for_strats and check_strategies:
                    status.text("Finding reduction threshold...")

                    def removal_works_with(bc, cands, test_budget):
                        """Quick check (no strategy computation)."""
                        r = process_ballot_counts_post_elim_no_print(
                            ballot_counts=bc, k=k, candidates=cands,
                            elim_cands=[], check_strats=False,
                            budget_percent=test_budget,
                            check_removal_here=True,
                            keep_at_least=effective_keep_at_least,
                            rigorous_check=rigorous_check,
                            spl_check=(k > 1)
                        )
                        removed = r.get("candidates_removed", [])
                        retained = r.get("candidates_retained", [])
                        return bool(removed) and len(retained) >= k and len(retained) <= max_for_strats

                    def find_threshold(bc, cands):
                        """Binary search for highest working budget."""
                        if removal_works_with(bc, cands, budget_percent):
                            return budget_percent
                        lo, hi = 0.5, budget_percent
                        best = None
                        while hi - lo > 0.5:
                            mid = round((lo + hi) / 2, 1)
                            if removal_works_with(bc, cands, mid):
                                best = mid
                                lo = mid
                            else:
                                hi = mid
                        return best

                    # Phase 1: try full candidate set
                    threshold = find_threshold(ballot_counts, candidates_list)

                    if threshold is not None:
                        # Full set works (Dis 1, 2, 3)
                        computed_threshold = threshold
                        effective_bc = ballot_counts
                        effective_cands = candidates_list
                    elif k > 1 and n_candidates > max_for_strats + 2:
                        # Phase 2: pre-filter and retry (Dis 4)
                        n_to_keep = max_for_strats + 2
                        elim_cands = list(results_alphabetical[n_to_keep:])
                        elim_string = ''.join(elim_cands)
                        filtered_bc = {}
                        for key, value in ballot_counts.items():
                            new_key = ''.join(c for c in key if c not in elim_string)
                            if new_key:
                                filtered_bc[new_key] = filtered_bc.get(new_key, 0) + value
                        effective_bc = filtered_bc
                        effective_cands = [c for c in candidates_list if c not in elim_cands]
                        threshold2 = find_threshold(effective_bc, effective_cands)
                        if threshold2 is not None:
                            computed_threshold = threshold2

                    # Compute strategies at the reduction threshold
                    progress.progress(60)
                    status.text(f"Computing strategies at {computed_threshold:.1f}% threshold...")
                    analysis_result = process_ballot_counts_post_elim_no_print(
                        ballot_counts=effective_bc, k=k,
                        candidates=effective_cands, elim_cands=[],
                        check_strats=True,
                        budget_percent=computed_threshold,
                        check_removal_here=True,
                        keep_at_least=effective_keep_at_least,
                        rigorous_check=rigorous_check,
                        spl_check=(k > 1)
                    )
                else:
                    # Small election (≤ max_for_strats candidates): compute strategies directly
                    analysis_result = process_ballot_counts_post_elim_no_print(
                        ballot_counts=effective_bc, k=k,
                        candidates=effective_cands, elim_cands=[],
                        check_strats=check_strategies,
                        budget_percent=budget_percent,
                        check_removal_here=False,
                        keep_at_least=effective_keep_at_least,
                        rigorous_check=rigorous_check,
                        spl_check=(k > 1)
                    )

                results = analysis_result.get("overall_winning_order", candidates_list)
                strategies = analysis_result.get("Strategies", {})
                Q = analysis_result.get("quota", Q)

                # Results should now be ['A', 'B', 'C', ...] where first k are winners
                # For multi-winner (k > 1), first k candidates are all winners
                winner_codes = results[:k]
                winners = [reverse_mapping.get(code, code) for code in winner_codes]
                winner = ", ".join(winners) if k > 1 else winners[0]

                # Store social choice order for display (names in winning order)
                social_choice_order = ordered_candidate_names

                # Update candidates_mapping to final_mapping for display
                candidates_mapping = final_mapping

                # Step 6: Ballot exhaustion
                status.text("Analyzing ballot exhaustion...")
                progress.progress(80)

                # Use appropriate exhaustion function based on k
                if k == 1:
                    # IRV_ballot_exhaust returns CUMULATIVE exhaustion (total - remaining)
                    exhausted_list, exhausted_dict = IRV_ballot_exhaust(candidates_list, ballot_counts)
                    exhausted_pct = {key: round(val/total_votes*100, 2) for key, val in exhausted_dict.items()}
                    total_exhausted_final = max(exhausted_dict.values()) if exhausted_dict else 0
                else:
                    exhausted_list, exhausted_dict, stv_winners = STV_ballot_exhaust(candidates_list, ballot_counts, k, Q)
                    # exhausted_dict stores exhaustion BEFORE each candidate's event (paper convention)
                    exhausted_pct = {key: round(val/total_votes*100, 2) for key, val in exhausted_dict.items()}
                    total_exhausted_final = sum(exhausted_list)  # final cumulative from incremental list

                exhaustion_rate = round(total_exhausted_final / total_votes * 100, 2)

                progress.progress(100)
                status.text("Analysis complete!")

                # ========================================
                # RESULTS DISPLAY
                # ========================================
                st.markdown("---")
                st.markdown("# Election Analysis Results")

                with st.expander("How to read these results", expanded=False):
                    st.markdown("""
**Color coding in the Victory Gap table:**
- **Green** = Winner or very close to winning (gap < 5%)
- **Yellow** = Contender with a realistic path (gap 5-20%)
- **Peach/Salmon** = Competitive but distant (gap 20-45%)
- **Pink/Gray** = Far behind or beyond the analysis threshold

*For multi-winner elections (k > 1), these thresholds are scaled proportionally.*

**Ballot Exhaustion Impact:**
- If a candidate's **exhaustion % > victory gap %**, completing those
  exhausted ballots *could* theoretically change the outcome.
- The probability models estimate how likely that change actually is.
- A high probability means ballot exhaustion meaningfully affected
  competitiveness; a low probability means the outcome is robust.

**Strategic Complexity:**
- **Selfish** = the candidate's best strategy is simply getting more of their own supporters to vote.
  This is the straightforward, expected case.
- **Non-Selfish** = the candidate benefits from a more complex strategy, such as supporting
  a weaker rival to change the elimination order. This is rare in real elections.

**Preference Order Alignment:**
- A **match** means the order candidates were eliminated lines up with how
  close they were to winning. The results tell a clear story.
- A **mismatch** means some candidates were eliminated earlier than their
  competitiveness would suggest, or vice versa.
                    """)

                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if k > 1:
                        st.metric(f"Winners ({k})", winner)
                    else:
                        st.metric("Winner", winner)
                with col2:
                    st.metric("Total Votes", f"{total_votes:,}")
                with col3:
                    st.metric("Quota (Droop)", f"{Q:,.0f}")
                with col4:
                    st.metric("Final Exhaustion", f"{exhaustion_rate:.1f}%")

                # Candidate removal and threshold info
                candidates_removed = analysis_result.get("candidates_removed", [])
                candidates_retained = analysis_result.get("candidates_retained", [])

                # Debug info about reduction

                if candidates_removed:
                    removed_names = [reverse_mapping.get(c, c) for c in candidates_removed if c in reverse_mapping or c in candidates_removed]
                    removal_msg = f"**Candidate Reduction:** Reduced from {len(candidates_list)} to {len(candidates_list) - len(candidates_removed)} candidates for tractable analysis."
                    if computed_threshold < budget_percent:
                        removal_msg += f" Found working threshold at **{computed_threshold:.1f}%** budget."
                    st.info(removal_msg)
                    with st.expander("Removed candidates"):
                        st.write(", ".join(removed_names) if removed_names else str(candidates_removed))
                elif computed_threshold < budget_percent and strategies:
                    st.info(f"**Note:** Due to election complexity ({len(candidates_list)} candidates), strategies were computed at **{computed_threshold:.1f}%** budget threshold (reduced from your {budget_percent:.0f}% setting).")

                # ========================================
                # ATTRIBUTE 1: VICTORY GAP & MARGIN OF VICTORY
                # ========================================
                st.markdown("## 1. Victory Gap & Competitiveness")
                st.markdown("""
                The **Victory Gap** shows how many additional votes (as % of total) each candidate needs to win.
                The **Margin of Victory** is the smallest gap among non-winners - lower = more competitive.
                """)

                # Build results table
                order_data = []
                non_winner_gaps = []
                all_selfish = True

                for i, code in enumerate(results):
                    name = reverse_mapping.get(code, code)
                    strat_data = strategies.get(code, None)

                    if strat_data is not None and isinstance(strat_data, (list, tuple)) and len(strat_data) > 0:
                        gap = strat_data[0]
                        strategy_detail = strat_data[1] if len(strat_data) > 1 else {}
                    else:
                        gap = float('inf')
                        strategy_detail = {}

                    # For multi-winner, first k are all winners regardless of computed gap
                    # Non-winners (i >= k) should NEVER be shown as "Winner" even if gap=0
                    if i < k:
                        category = "Winner"
                        gap = 0.0  # Ensure winners always show gap=0
                    elif gap != float('inf') and gap > 0:
                        category = categorize_gap(gap, k)
                    elif gap == 0:
                        # Non-winner with gap=0 is an artifact - treat as near winner
                        category = "Near Winner"
                    elif gap == float('inf'):
                        # Strategy not computed - we only know gap >= threshold, not actual category
                        category = "Beyond Threshold"
                    else:
                        category = "-"

                    # Check if strategy is selfish
                    selfish = is_selfish_strategy(strategy_detail, code)
                    if not selfish and gap > 0:
                        all_selfish = False

                    # Track non-winner gaps for margin of victory
                    if i >= k and gap > 0 and gap != float('inf'):
                        non_winner_gaps.append(gap)

                    # Format strategy description
                    # Only first k candidates are actual winners
                    if i < k:
                        strategy_desc = "Actual winner" if k > 1 else "Current winner"
                        strategy_type = "-"
                    elif gap == float('inf'):
                        if strategies:
                            # We computed strategies at a threshold, so this candidate needs >= threshold
                            strategy_desc = f"≥ {computed_threshold:.1f}% needed"
                        else:
                            strategy_desc = "Not computed"
                        strategy_type = "-"
                    elif gap == 0 and i >= k:
                        # Non-winner with gap=0 is an edge case (artifact of reduced state)
                        strategy_desc = "Very close to winning"
                        strategy_type = "-"
                    elif strategy_detail and not selfish:
                        support_parts = []
                        for cand, votes in strategy_detail.items():
                            if votes > 0:
                                cand_name = reverse_mapping.get(cand, cand)
                                support_parts.append(f"{cand_name}: +{votes:.1f}%")
                        strategy_desc = ", ".join(support_parts)
                        strategy_type = "Non-Selfish"
                    else:
                        strategy_desc = f"Self-support: +{gap:.2f}%"
                        strategy_type = "Selfish"

                    # Check if candidate's strategy was not computed
                    # This happens if: (1) explicitly removed by remove_irrelevent, OR
                    # (2) not in the reduced candidate set's strategy results
                    # If strategies were computed at threshold X, any uncomputed candidate has gap >= X
                    was_filtered = code in candidates_removed
                    strategy_not_computed = (gap == float('inf')) and (i >= k)  # Non-winner without strategy

                    order_data.append({
                        "Rank": i + 1,
                        "ID": code,
                        "Candidate": name,
                        "Victory Gap (%)": gap if gap != float('inf') else None,
                        "Is Winner": (i < k),  # First k candidates are winners in multi-winner
                        "Gap Computed": (gap != float('inf')),
                        "Was Filtered": was_filtered,
                        "Strategy Not Computed": strategy_not_computed,
                        "Category": category,
                        "Strategy Type": strategy_type,
                        "Required Strategy": strategy_desc,
                        "Exhaustion (%)": exhausted_pct.get(code, 0)
                    })

                order_df = pd.DataFrame(order_data)

                # Calculate Margin of Victory
                margin_of_victory = min(non_winner_gaps) if non_winner_gaps else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Margin of Victory", f"{margin_of_victory:.2f}%",
                              help="Smallest victory gap among non-winners")
                with col2:
                    competitiveness = "High" if margin_of_victory < 10 else "Medium" if margin_of_victory < 25 else "Low"
                    st.metric("Competitiveness", competitiveness)

                # Display table with paper-style coloring
                def style_victory_table(row):
                    cat = row['Category']
                    color = CATEGORY_COLORS.get(cat, {}).get('bg', 'white')
                    return [f'background-color: {color}'] * len(row)

                display_df = order_df[['Rank', 'ID', 'Candidate', 'Victory Gap (%)', 'Was Filtered', 'Strategy Not Computed', 'Category', 'Required Strategy', 'Exhaustion (%)']].copy()

                # Format victory gap: show ≥ X% for any candidate without computed strategy
                # If we successfully computed strategies at threshold X, uncomputed candidates have gap >= X
                def format_gap(row):
                    gap = row['Victory Gap (%)']
                    if pd.notna(gap):
                        return f"{gap:.2f}"
                    elif (row['Was Filtered'] or row['Strategy Not Computed']) and strategies:
                        # Strategies were computed at computed_threshold, so this candidate has gap >= threshold
                        return f"≥ {computed_threshold:.1f}"
                    else:
                        return "N/A"

                display_df['Victory Gap (%)'] = display_df.apply(format_gap, axis=1)
                display_df = display_df.drop(columns=['Was Filtered', 'Strategy Not Computed'])
                display_df['Exhaustion (%)'] = display_df['Exhaustion (%)'].apply(lambda x: f"{x:.2f}")

                styled_df = display_df.style.apply(style_victory_table, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Victory Gap Chart
                chart_data = [d for d in order_data if d['Victory Gap (%)'] is not None]
                if chart_data:
                    fig = px.bar(
                        chart_data,
                        x="Candidate",
                        y="Victory Gap (%)",
                        color="Category",
                        title="Victory Gap: Additional Votes Needed to Win",
                        color_discrete_map={k: v['hex'] for k, v in CATEGORY_COLORS.items()}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                # ========================================
                # ATTRIBUTE 2: BALLOT EXHAUSTION IMPACT
                # ========================================
                st.markdown("## 2. Ballot Exhaustion Impact")
                st.markdown("""
                **Ballot exhaustion** occurs when a voter's ranked choices are all eliminated.
                If exhaustion % > victory gap %, completing those ballots *could* change the outcome.
                """)

                # Analyze exhaustion impact
                impact_data = []
                candidates_with_potential = []

                for d in order_data:
                    code = d['ID']
                    gap = d['Victory Gap (%)']
                    exhaust = exhausted_pct.get(code, 0)
                    name = d['Candidate']
                    is_winner = d['Is Winner']
                    gap_computed = d['Gap Computed']

                    if is_winner and (gap == 0 or gap is None):
                        # Actual winner
                        impact = "Winner"
                        could_win = False
                        gap_display = "0.00"
                        excess_display = "-"
                    elif not gap_computed:
                        # Strategy not computed - use threshold logic
                        if exhaust < computed_threshold:
                            # Exhaust is below threshold where we CAN compute, so definitely no impact
                            impact = "No impact"
                        else:
                            # Exhaust is above threshold, we can't say for sure
                            impact = "Not computed"
                        could_win = False
                        gap_display = "N/C"
                        excess_display = "-"
                    elif exhaust > gap:
                        impact = "Potential impact"
                        could_win = True
                        gap_display = f"{gap:.2f}"
                        excess_display = f"{exhaust - gap:.2f}"
                        candidates_with_potential.append({
                            'name': name,
                            'code': code,
                            'gap': gap,
                            'exhaust': exhaust,
                            'excess': exhaust - gap
                        })
                    else:
                        impact = "No impact"
                        could_win = False
                        gap_display = f"{gap:.2f}"
                        excess_display = f"{exhaust - gap:.2f}"

                    impact_data.append({
                        'Candidate': name,
                        'Victory Gap (%)': gap_display,
                        'Exhaustion (%)': f"{exhaust:.2f}",
                        'Excess': excess_display,
                        'Impact': impact
                    })

                impact_df = pd.DataFrame(impact_data)

                def style_impact_table(row):
                    impact = row['Impact']
                    if impact == "Winner":
                        return ['background-color: rgb(189, 223, 167)'] * len(row)
                    elif impact == "Potential impact":
                        return ['background-color: rgb(253, 245, 206)'] * len(row)
                    elif impact == "Not computed":
                        return ['background-color: rgb(220, 220, 220)'] * len(row)
                    return [''] * len(row)

                styled_impact = impact_df.style.apply(style_impact_table, axis=1)
                st.dataframe(styled_impact, use_container_width=True, hide_index=True)

                # Show info about computed threshold if different from user's budget
                if computed_threshold < budget_percent:
                    st.info(f"**Note:** Strategies computed at **{computed_threshold:.1f}%** budget threshold (reduced from your {budget_percent:.0f}% setting) due to election complexity. Only candidates with victory gaps below {computed_threshold:.1f}% were analyzed.")

                if candidates_with_potential:
                    st.warning(f"**{len(candidates_with_potential)} candidate(s)** have exhaustion > victory gap. Completing ballots could theoretically change the outcome.")

                    with st.expander("Detailed Exhaustion Impact Analysis with Probability Models"):
                        for cand in candidates_with_potential:
                            st.markdown(f"### {cand['name']} ({cand['code']})")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Victory Gap", f"{cand['gap']:.2f}%")
                            with col2:
                                st.metric("Exhaustion at Elimination", f"{cand['exhaust']:.2f}%")
                            with col3:
                                st.metric("Excess Available", f"{cand['excess']:.2f}%")

                            # Calculate required preference percentage
                            if cand['exhaust'] > 0:
                                required_net_advantage = (cand['gap'] / cand['exhaust']) * 100
                                required_pref_pct = (1 + required_net_advantage / 100) / 2 * 100

                                st.markdown(f"**Required Preference:** {required_pref_pct:.1f}% of exhausted voters must prefer this candidate over the winner")

                                if PROB_MODELS_AVAILABLE:
                                    # ====== SIX PROBABILITY MODELS FROM PAPER ======
                                    st.markdown("#### Probability Models (Chance of Winning via Ballot Completion)")
                                    st.markdown("""
**Beta Distribution Models:**
- **Gap-Based Beta:** Uses a Beta distribution parameterized solely by the victory gap. Does not use observed ballot data.
- **Similarity Beta:** Groups exhausted ballots by first preference; estimates B>A vs A>B rates from non-exhausted ballots with the same first preference.
- **Prior-Posterior Beta:** Bayesian update combining Gap-Based Beta prior with observed first-preference-conditioned rates.

**Bootstrap Simulation Models:**
- **Similarity Bootstrap:** For each exhausted ballot, samples completions from non-exhausted ballots sharing the same first preference.
- **Rank-Restricted Bootstrap:** Like Similarity Bootstrap, but only completes ballots with room for additional rankings (e.g., <5 ranks in NYC).
- **Unconditional Bootstrap:** Samples completions from all non-exhausted ballots regardless of first preference.
                                    """)

                                    # Use 200 iterations for bootstrap (fast enough, reasonably accurate)
                                    n_bootstrap = 200

                                    gap = cand['gap']
                                    exhaust = cand['exhaust']
                                    cand_code = cand['code']
                                    candidates_list_for_model = list(set(results))

                                    with st.spinner(f"Computing probability models ({n_bootstrap} bootstrap iterations)..."):
                                        if k == 1:
                                            # ====== SINGLE-WINNER (IRV) MODELS ======
                                            # For single-winner: compare candidate (B) vs winner (A)
                                            # Exhausted ballots = ballots not ranking A or B
                                            exhausted_ballots_for_model = {
                                                ballot: count for ballot, count in ballot_counts.items()
                                                if 'A' not in ballot and 'B' not in ballot
                                            }

                                            # 1. Gap-Based Beta (fast)
                                            gap_based_beta = beta_probability(required_pref_pct, gap) * 100

                                            # 2. Similarity Beta (fast)
                                            similarity_beta = direct_posterior_beta(
                                                required_pref_pct, ballot_counts, candidates_list_for_model,
                                                exhausted_ballots_for_model, gap
                                            ) * 100

                                            # 3. Prior-Posterior Beta (fast)
                                            prior_post_beta = prior_posterior_beta(
                                                required_pref_pct, ballot_counts, candidates_list_for_model,
                                                exhausted_ballots_for_model, gap
                                            ) * 100

                                            # 4. Similarity Bootstrap (uses iterations)
                                            sim_bootstrap, sim_ci, _ = category_based_bootstrap(
                                                ballot_counts, candidates_list_for_model, exhausted_ballots_for_model,
                                                gap_to_win_pct=gap, exhaust_pct=exhaust,
                                                required_preference_pct=required_pref_pct, n_bootstrap=n_bootstrap
                                            )
                                            sim_bootstrap *= 100

                                            # 5. Rank-Restricted Bootstrap (uses iterations)
                                            rank_bootstrap, rank_ci, _ = limited_ranking_bootstrap(
                                                ballot_counts, candidates_list_for_model, exhausted_ballots_for_model,
                                                gap_to_win_pct=gap, exhaust_pct=exhaust,
                                                required_preference_pct=required_pref_pct, n_bootstrap=n_bootstrap,
                                                max_rankings=max_rankings
                                            )
                                            rank_bootstrap *= 100

                                            # 6. Unconditional Bootstrap (uses iterations)
                                            uncond_bootstrap, uncond_ci, _ = unconditional_bootstrap(
                                                ballot_counts, candidates_list_for_model, exhausted_ballots_for_model,
                                                gap_to_win_pct=gap, exhaust_pct=exhaust,
                                                required_preference_pct=required_pref_pct, n_bootstrap=n_bootstrap
                                            )
                                            uncond_bootstrap *= 100

                                        else:
                                            # ====== MULTI-WINNER (STV) MODELS ======
                                            # For multi-winner: compare candidate vs ALL active candidates
                                            # Get event_log to determine active candidates at each elimination
                                            event_log, _, _ = STV_optimal_result_simple(candidates_list, ballot_counts, k, Q)

                                            # Build mapping of candidate -> active candidates when eliminated/won
                                            active_at_event = {}
                                            candidates_remaining = set(candidates_list)
                                            for candidate, is_winner in event_log:
                                                active_at_event[candidate] = candidates_remaining.copy()
                                                candidates_remaining.remove(candidate)

                                            # Get active candidates for this specific candidate
                                            active_candidates = active_at_event.get(cand_code, set(candidates_list))

                                            # Multi-winner exhausted = ballots not ranking ANY active candidate
                                            exhausted_ballots_multi = {
                                                ballot: count for ballot, count in ballot_counts.items()
                                                if ballot and not any(c in active_candidates for c in ballot)
                                            }

                                            # Analyze preference patterns (candidate vs all other active)
                                            preference_analysis = analyze_preference_patterns_multi_winner(
                                                ballot_counts, exhausted_ballots_multi, cand_code, active_candidates
                                            )

                                            # 1. Gap-Based Beta (fast)
                                            gap_based_beta = beta_probability_multi_winner(required_pref_pct, gap) * 100

                                            # 2. Similarity Beta (fast)
                                            similarity_beta = similarity_beta_multi_winner(
                                                preference_analysis, cand_code, active_candidates,
                                                required_pref_pct, gap
                                            ) * 100

                                            # 3. Prior-Posterior Beta (fast)
                                            prior_post_beta = prior_posterior_beta_multi_winner(
                                                preference_analysis, cand_code, active_candidates,
                                                required_pref_pct, gap
                                            ) * 100

                                            # 4. Similarity Bootstrap (uses iterations)
                                            sim_bootstrap, sim_ci = category_bootstrap_multi_winner(
                                                preference_analysis, cand_code, active_candidates,
                                                required_pref_pct, gap, n_bootstrap
                                            )
                                            sim_bootstrap *= 100

                                            # 5. Rank-Restricted Bootstrap (uses iterations)
                                            rank_bootstrap, rank_ci = limited_ranking_bootstrap_multi_winner(
                                                ballot_counts, exhausted_ballots_multi, cand_code, active_candidates,
                                                required_pref_pct, gap, n_bootstrap, max_rankings
                                            )
                                            rank_bootstrap *= 100

                                            # 6. Unconditional Bootstrap (uses iterations)
                                            uncond_bootstrap, uncond_ci = unconditional_bootstrap_multi_winner(
                                                ballot_counts, exhausted_ballots_multi, cand_code, active_candidates,
                                                required_pref_pct, gap, n_bootstrap, max_rankings
                                            )
                                            uncond_bootstrap *= 100

                                    # Combined weighted probability (emphasizing empirical methods)
                                    combined_prob = (
                                        0.10 * gap_based_beta +
                                        0.15 * similarity_beta +
                                        0.15 * prior_post_beta +
                                        0.25 * sim_bootstrap +
                                        0.20 * rank_bootstrap +
                                        0.15 * uncond_bootstrap
                                    )

                                    # Display probability results with paper model names
                                    if k == 1:
                                        # Single-winner descriptions (A vs B)
                                        prob_models = [
                                            ("Gap-Based Beta", gap_based_beta,
                                             f"Beta distribution calibrated to victory gap ({gap:.1f}%). Larger gaps shift distribution toward the leader."),
                                            ("Similarity Beta", similarity_beta,
                                             "Uses observed B>A vs A>B preference ratios by first-preference category to fit Beta parameters."),
                                            ("Prior-Posterior Beta", prior_post_beta,
                                             "Bayesian update: combines gap-based prior with observed preference evidence."),
                                            ("Similarity Bootstrap", sim_bootstrap,
                                             f"Bootstrap ({n_bootstrap} iterations) grouping exhausted ballots by first preference, sampling completions from category-specific ratios."),
                                            ("Rank-Restricted Bootstrap", rank_bootstrap,
                                             f"Like Similarity Bootstrap but respects ranking limits (max {max_rankings}). More conservative estimate."),
                                            ("Unconditional Bootstrap", uncond_bootstrap,
                                             f"Bootstrap ({n_bootstrap} iterations) assuming random completion without first-preference conditioning."),
                                        ]
                                    else:
                                        # Multi-winner descriptions (candidate vs all active)
                                        n_active = len(active_candidates)
                                        prob_models = [
                                            ("Gap-Based Beta", gap_based_beta,
                                             f"Beta distribution calibrated to victory gap ({gap:.1f}%). Compares candidate vs all {n_active} active candidates."),
                                            ("Similarity Beta", similarity_beta,
                                             "Uses observed preference patterns: candidate ranked highest vs others ranked higher among active candidates."),
                                            ("Prior-Posterior Beta", prior_post_beta,
                                             "Bayesian update: combines gap-based prior with observed multi-candidate preference evidence."),
                                            ("Similarity Bootstrap", sim_bootstrap,
                                             f"Bootstrap ({n_bootstrap} iterations) sampling completions based on first-preference category patterns."),
                                            ("Rank-Restricted Bootstrap", rank_bootstrap,
                                             f"Like Similarity Bootstrap but only completes partial ballots (< {max_rankings} rankings)."),
                                            ("Unconditional Bootstrap", uncond_bootstrap,
                                             f"Bootstrap ({n_bootstrap} iterations) sampling from all ballots ranking any active candidate."),
                                        ]

                                    # Summary table
                                    st.markdown("| Model | Probability |")
                                    st.markdown("|:------|----------:|")
                                    for model_name, prob, _ in prob_models:
                                        st.markdown(f"| {model_name} | {prob:.1f}% |")
                                    st.markdown(f"| **Combined** | **{combined_prob:.1f}%** |")

                                    # Interpretation
                                    if combined_prob >= 40:
                                        st.success(f"**High probability ({combined_prob:.0f}%)** - Completing exhausted ballots could plausibly change the outcome")
                                    elif combined_prob >= 15:
                                        st.warning(f"**Moderate probability ({combined_prob:.0f}%)** - Outcome change possible but not likely")
                                    else:
                                        st.info(f"**Low probability ({combined_prob:.0f}%)** - Outcome change unlikely even with ballot completion")

                                elif not PROB_MODELS_AVAILABLE:
                                    st.warning("Probability models not available. Install ballot_exhaustion module.")

                            st.markdown("---")
                else:
                    st.success("**No candidates** have exhaustion rates exceeding their victory gaps. The election outcome is robust to ballot completion.")

                # Exhaustion chart
                col1, col2 = st.columns(2)
                with col1:
                    exhaust_chart = []
                    for code in reversed(results):
                        name = reverse_mapping.get(code, code)
                        pct = exhausted_pct.get(code, 0)
                        exhaust_chart.append({"Eliminated": name, "Cumulative Exhaustion (%)": pct})

                    fig = px.area(exhaust_chart, x="Eliminated", y="Cumulative Exhaustion (%)",
                                  title="Ballot Exhaustion Over Rounds")
                    fig.update_traces(fill='tozeroy', line_color='#e74c3c')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Gap vs Exhaustion scatter
                    scatter_data = [d for d in order_data if d['Victory Gap (%)'] is not None and d['Victory Gap (%)'] > 0]
                    if scatter_data:
                        fig = px.scatter(
                            scatter_data,
                            x="Victory Gap (%)",
                            y="Exhaustion (%)",
                            text="ID",
                            title="Victory Gap vs Exhaustion at Elimination",
                            color="Category",
                            color_discrete_map={k: v['hex'] for k, v in CATEGORY_COLORS.items()}
                        )
                        # Add diagonal line (exhaustion = gap)
                        max_val = max(max([d['Victory Gap (%)'] for d in scatter_data]), max([d['Exhaustion (%)'] for d in scatter_data]))
                        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                                  line=dict(dash='dash', color='gray'), name='Exhaust = Gap'))
                        fig.update_traces(textposition='top center')
                        st.plotly_chart(fig, use_container_width=True)

                # ========================================
                # ATTRIBUTE 3: STRATEGIC COMPLEXITY
                # ========================================
                st.markdown("## 3. Strategic Complexity")
                st.markdown("""
                **Selfish Strategy**: Optimal path to victory is simply adding votes for oneself.
                **Non-Selfish Strategy**: Optimal strategy requires supporting other candidates (spoiler effects).
                """)

                strategy_types = [d['Strategy Type'] for d in order_data if d['Strategy Type'] not in ['-']]
                selfish_count = strategy_types.count('Selfish')
                non_selfish_count = strategy_types.count('Non-Selfish')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Selfish Strategies", selfish_count)
                with col2:
                    st.metric("Non-Selfish Strategies", non_selfish_count)
                with col3:
                    complexity = "Simple" if all_selfish else "Complex"
                    st.metric("Strategic Complexity", complexity)

                if all_selfish:
                    st.success(f"**All optimal strategies are selfish** (self-support only) at {budget_percent}% allowance. This election exhibits plurality-like strategic dynamics.")
                else:
                    st.warning(f"**Some candidates have non-selfish optimal strategies** at {budget_percent}% allowance. Supporting rivals (spoiler effects) could be advantageous.")

                # ========================================
                # ATTRIBUTE 4: PREFERENCE ORDER ALIGNMENT
                # ========================================
                st.markdown("## 4. Preference Order Alignment")
                st.markdown("""
                Does the **Social Choice Order** (elimination sequence) match the **Victory Gap Order** (sorted by closeness to winning)?
                **Match** = RCV results are transparent. **No Match** = formal results may obscure true competitiveness.
                """)

                matches, victory_gap_order, mismatches = compute_preference_order_alignment(results, strategies)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Social Choice Order** (actual RCV result):")
                    sco_names = [f"{i+1}. {reverse_mapping.get(c, c)}" for i, c in enumerate(results)]
                    st.write(" → ".join(sco_names[:5]) + ("..." if len(sco_names) > 5 else ""))

                with col2:
                    st.markdown("**Victory Gap Order** (sorted by gap):")
                    vgo_names = [f"{i+1}. {reverse_mapping.get(c, c)}" for i, c in enumerate(victory_gap_order)]
                    st.write(" → ".join(vgo_names[:5]) + ("..." if len(vgo_names) > 5 else ""))

                if matches:
                    st.success("**Perfect Match!** The elimination order reflects true competitive dynamics. RCV results are transparent.")
                else:
                    st.warning(f"**No Match** at {len(mismatches)} position(s). The elimination sequence differs from victory gap ranking.")
                    with st.expander("View Mismatches"):
                        for pos, sco, vgo in mismatches:
                            sco_name = reverse_mapping.get(sco, sco)
                            vgo_name = reverse_mapping.get(vgo, vgo)
                            st.write(f"Position {pos}: SCO has **{sco_name}**, VGO has **{vgo_name}**")

                # ========================================
                # SUMMARY INSIGHTS
                # ========================================
                st.markdown("## Summary: Key Insights")

                insights = []

                # Competitiveness insight
                if margin_of_victory < 10:
                    insights.append(f"**Highly competitive election** - margin of victory is only {margin_of_victory:.2f}%")
                elif margin_of_victory < 25:
                    insights.append(f"**Moderately competitive election** - margin of victory is {margin_of_victory:.2f}%")
                else:
                    insights.append(f"**Decisive victory** - margin of victory is {margin_of_victory:.2f}%")

                # Exhaustion insight
                if candidates_with_potential:
                    insights.append(f"**Exhaustion could matter** - {len(candidates_with_potential)} candidate(s) have exhaust > gap")
                else:
                    insights.append("**Robust to exhaustion** - completing ballots unlikely to change outcome")

                # Strategy insight
                if all_selfish:
                    insights.append("**Simple strategic dynamics** - all optimal strategies are self-support")
                else:
                    insights.append("**Complex strategic dynamics** - some candidates benefit from supporting rivals")

                # Alignment insight
                if matches:
                    insights.append("**Transparent results** - elimination order matches competitiveness ranking")
                else:
                    insights.append("**Some opacity** - elimination order differs from competitiveness ranking")

                for insight in insights:
                    st.markdown(f"- {insight}")

                # ========================================
                # RAW DATA & EXPORT
                # ========================================
                with st.expander("Raw Analysis Data"):
                    st.json({
                        "num_candidates": analysis_result.get("num_candidates"),
                        "total_votes": analysis_result.get("total_votes"),
                        "quota": analysis_result.get("quota"),
                        "margin_of_victory": margin_of_victory,
                        "overall_winning_order": results,
                        "candidates_removed": candidates_removed,
                        "strategies": {k: v for k, v in strategies.items()}
                    })

                st.markdown("## Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = display_df.to_csv(index=False)
                    st.download_button("Download Victory Gap Table (CSV)", data=csv_data,
                                       file_name=f"rcv_victory_gap_{file_name.replace('.csv', '')}.csv",
                                       mime="text/csv")
                with col2:
                    impact_csv = impact_df.to_csv(index=False)
                    st.download_button("Download Exhaustion Impact (CSV)", data=impact_csv,
                                       file_name=f"rcv_exhaustion_{file_name.replace('.csv', '')}.csv",
                                       mime="text/csv")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                with st.expander("Error Details"):
                    st.exception(e)

    except Exception as e:
        st.error(f"Failed to load file: {e}")
        with st.expander("Error Details"):
            st.exception(e)

else:
    # Instructions
    st.markdown("""
    ## Getting Started

    1. **Upload** your election CSV file
    2. **Configure** settings in the sidebar
    3. **Click** "Run Analysis" to see results

    ### What You'll Get

    Based on the research paper *"Simpler Than You Think: The Practical Dynamics of RCV"*:

    1. **Victory Gap & Margin of Victory** - How close is each candidate to winning?
    2. **Ballot Exhaustion Impact** - Could completing exhausted ballots change the outcome?
    3. **Strategic Complexity** - Are optimal strategies simple (self-support) or complex (support rivals)?
    4. **Preference Order Alignment** - Does elimination order reflect true competitiveness?

    ### Data Format

    Your CSV should have columns like:
    - `Choice_1, Choice_2, Choice_3, ...` (preferred)
    - `rank1, rank2, rank3, ...` (also supported)
    """)

# Footer
st.markdown("---")
