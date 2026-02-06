"""
Case Study Helper Functions for RCV Strategy Analysis
======================================================

Repository: https://github.com/sanyukta-D/Optimal_Strategies_in_RCV

This module provides utility functions for processing ballot data and computing
optimal strategies in Ranked Choice Voting (RCV) elections, supporting both
single-winner (IRV) and multi-winner (STV) elections.

Key Concepts:
-------------
- ballot_counts: Dict mapping ballot strings to vote counts
  Example: {'ABC': 100, 'BAC': 80} means 100 voters ranked A>B>C, 80 ranked B>A>C

- Victory Gap: Additional votes (% of total) a candidate needs to join the winning set

- Droop Quota: Q = total_votes / (k+1) + 1, where k = number of winners
  A candidate must reach Q to be declared a winner in STV

- Tractable Election: < MAX_TRACTABLE_CANDIDATES allows exact strategy computation
  For larger elections, we use remove_irrelevant() to reduce candidate count

- Collection: List of ballot states at each STV round (from STV_optimal_result_simple)
  collection[i] = ballot state after i events (eliminations + wins)
  Used for "small election method" when handling early winners

Key Functions:
--------------
- get_ballot_counts_df(): Convert CSV DataFrame to ballot_counts dictionary
- process_ballot_counts_post_elim_no_print(): Main strategy computation (see detailed docs below)
- permit_STV_removal(): Verify early winner removal is valid (Theorem: Irrelevant Extension)
- convert_combination_strats_to_candidate_strats(): Convert multi-winner results to per-candidate format

Paper References:
-----------------
- "Optimal Strategies in Ranked Choice Voting" (Strategy computation, candidate removal theorems)
- "Simpler Than You Think: The Practical Dynamics of RCV" (Victory gaps, practical interpretation)
"""

from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.core.strategy import reach_any_winners_campaign, reach_any_winners_campaign_memoization, reach_any_winners_campaign_parallel
from rcv_strategies.core.candidate_removal import remove_irrelevant, strict_support, predict_losses
from rcv_strategies.constants import MAX_TRACTABLE_CANDIDATES
import time
from copy import deepcopy
from rcv_strategies.utils import helpers as utils
import os
import pandas as pd
import json


# ============================================================================
# BALLOT COUNTING FUNCTIONS
# ============================================================================

def get_ballot_counts_df(candidates_mapping, df):
    """
    Convert ranked choice dataframe to ballot counts dictionary for standard format.
    
    Parameters:
        candidates_mapping (dict): Mapping from candidate names to identifiers.
        df (DataFrame): DataFrame containing voter choices.
    
    Returns:
        dict: Counts of each ballot type.
    """
    # Dynamically detect choice columns (e.g., "Choice_1", "Choice_2", ...)
    choice_columns = [col for col in df.columns if col.startswith("Choice_")]
    # Sort the choice columns by their numerical order
    choice_columns = sorted(choice_columns, key=lambda x: int(x.split('_')[1]))
    
    ballot_counts = {}
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        valid_candidates = []
        # Iterate through the dynamically determined choice columns
        for col in choice_columns:
            candidate = row[col]
            if candidate in candidates_mapping:
                valid_candidates.append(candidates_mapping[candidate])
        
        # If at least one valid candidate is found, form the ballot type string
        if valid_candidates:
            ballot_type = ''.join(valid_candidates)
            ballot_counts[ballot_type] = ballot_counts.get(ballot_type, 0) + 1
                
    return ballot_counts

def get_ballot_counts_df_republican_primary(candidates_mapping, df):
    """
    Convert ranked choice dataframe to ballot counts dictionary for Republican primary format.
    
    Parameters:
    candidates_mapping (dict): Mapping from candidate names to identifiers
    df (DataFrame): DataFrame containing voter choices with weights
    
    Returns:
    dict: Weighted counts of each ballot type
    """
    # Initialize a dictionary to store the counts of each ballot type as strings
    ballot_counts = {}
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Initialize an empty list to store valid candidates
        valid_candidates = []

        # Iterate through columns rank1 to rank13
        for i in range(1, 14):
            candidate = row[f'rank{i}']
            valid_candidates.append(candidates_mapping[candidate])

        ballot_type = ''.join(valid_candidates)

        # Use the 'weight' column to determine the number of voters for this ballot type
        weight = row['weight']

        # Add the weight to the count for this ballot type in the dictionary
        if ballot_type not in ballot_counts:
            ballot_counts[ballot_type] = weight
        else:
            ballot_counts[ballot_type] += weight
    
    return ballot_counts


# ============================================================================
# BALLOT PROCESSING FUNCTIONS
# ============================================================================

def process_ballot_counts_post_elim(ballot_counts, k, candidates, elim_cands, check_strats=False, 
                                   budget_percent=0, check_removal_here=False, keep_at_least=8, c_l = [], zeros = 0, rigorous_check=True ):
    """
    Process ballot counts after eliminating certain candidates.
    
    Parameters:
    ballot_counts (dict): Dictionary of ballot counts
    k (int): Number of winners to select
    candidates (list): List of candidate identifiers
    elim_cands (list): List of candidates to eliminate
    check_strats (bool): Whether to check strategies
    budget_percent (float): Budget percentage for strategy calculation
    check_removal_here (bool): Whether to check candidate removal
    keep_at_least (int): Minimum number of candidates to keep
    rigorous_check (bool): Whether to perform rigorous check in candidate removal
    
    Returns:
    None: Results are printed rather than returned
    """
    # Initialize a dictionary for filtered data
    filtered_data = {}
    elim_strings = ''.join(char for char in elim_cands)

    # Remove eliminated candidates while retaining the rest of the string
    for key, value in ballot_counts.items():
        new_key = ''.join(char for char in key if char not in elim_strings)
        filtered_data[new_key] = filtered_data.get(new_key, 0) + value
    filtered_data.pop('', None)

    # Get remaining candidates
    elec_cands = [cand for cand in candidates if cand not in elim_cands]
    
    # Calculate vote counts
    full_aggre_v_dict = utils.get_new_dict(ballot_counts)
    aggre_v_dict = utils.get_new_dict(filtered_data)

    # Calculate quota
    Q = round(sum(full_aggre_v_dict[cand] for cand in candidates)/(k+1)+1, 3)
    print("\n" + "="*50)
    print(f"Q = {Q}")
    print("="*50 + "\n")
    
    # Calculate strict support within eliminated candidates
    letter_counts = {}
    for key, value in ballot_counts.items():
        i = 0
        newset = []
        if key and key[i] in candidates:
            while key[i] in elim_strings:
                newset.append(key[i])
                i=i+1
                if i >= len(key):
                    break

        new_key = ''.join(char for char in newset)
        for letter in new_key:
            if letter in letter_counts:
                letter_counts[letter] += value
            else:
                letter_counts[letter] = value

    # Print vote counts after elimination
    print("\n" + "-"*50)
    print(f"Total votes if {elim_cands} are eliminated:")
    print("-"*50)
    
    wins_during_elims = []
    for c in elec_cands:
        print(f"  {c}: {aggre_v_dict[c]}")
        if aggre_v_dict[c] >= Q:
            wins_during_elims.append(c)

    if len(wins_during_elims) > 0:
        print("\nWinners during elimination of lower group:")
        print(f"  {wins_during_elims}")
    print("-"*50 + "\n")

    # Print strict support
    print("-"*50)
    print(f"Strict support within {elim_cands}:")
    for letter, count in letter_counts.items():
        print(f"  {letter}: {count}")
    for candi in candidates:
        print(candi, letter_counts.get(candi, 0))
    
    best_c_irrelevant = max(letter_counts, key=letter_counts.get) if letter_counts else None
    if best_c_irrelevant:
        print(f"\nMax strict support is with: {best_c_irrelevant}")
    print("-"*50 + "\n")

    # Run STV to determine winners
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, subresults = utils.return_main_sub(rt)
    print("="*50)
    print(f"Overall winning order is: {results}")
    print("="*50 + "\n")

    # Check candidate removal if requested
    budget = budget_percent * sum(full_aggre_v_dict[cand] for cand in candidates) * 0.01
    print(budget)
    if check_removal_here:
        candidates_reduced, group_remaining, stop = remove_irrelevant(
            ballot_counts, rt, results[:keep_at_least], budget, ''.join(results), rigorous_check
        )
        
        print("-"*50)
        if stop:
            print(f"We can remove {group_remaining} and keep {candidates_reduced}")
            if zeros> 0:
                print('you are incorrectly insisting on first few losses')
        else: 
            print("We cannot remove any more candidates")
        print("-"*50 + "\n")

    # Check strategies if requested
    # , top_k = candidates[:4], bottom_m=candidates[-5:]
    if check_strats:
        print("="*50)
        print(f"Checking strategies for {elec_cands}")
        start = time.time()
        strats_frame = reach_any_winners_campaign_parallel(elec_cands, k, Q, filtered_data, budget, c_l,  zeros)
        end = time.time()
        print(f"Total time: {end - start:.2f} seconds")
        print("\nStrategy frame:")
        print(strats_frame)
        strats_frame_percent = convert_to_percentage(strats_frame, sum(ballot_counts.values()))
        print("\nStrategy frame in percentage:")
        print(strats_frame_percent)
        print("="*50)

def convert_to_percentage(item, total):
    if isinstance(item, (int, float)):
        return round((item / total) * 100,3)
    elif isinstance(item, list):
        return [convert_to_percentage(sub, total) for sub in item]
    elif isinstance(item, dict):
        return {k: convert_to_percentage(v, total) for k, v in item.items()}
    else:
        return item


def convert_combination_strats_to_candidate_strats(strats_frame, k, results):
    """
    Convert combination-based strategies to per-candidate strategies.

    For multi-winner STV (k > 1), reach_any_winners_campaign returns results
    keyed by winner COMBINATIONS (e.g., 'ABC', 'ABD', 'ACD'). Each combination
    represents a possible winning set, with cost = votes needed to achieve it.

    This function converts to PER-CANDIDATE format for webapp display:
    - Original winners (results[:k]) always get gap = 0
    - Non-winners get minimum cost across all combinations that include them

    Input Format (multi-winner, k=3):
        {
            'ABC': [0, {}],           # Current winners, cost=0
            'ABD': [150, {'D': 150}], # D joins winners, needs 150 votes
            'ACD': [200, {'C': 100, 'D': 100}], # Different combination
            'BCD': [300, {'D': 300}], # A loses, B/C/D win
        }

    Output Format:
        {
            'A': [0.0, {}],           # Winner, gap=0
            'B': [0.0, {}],           # Winner, gap=0
            'C': [0.0, {}],           # Winner, gap=0
            'D': [150, {'D': 150}],   # Non-winner, minimum cost to join
        }

    The minimum cost for D is 150 (from 'ABD' combination), not 200 or 300.

    IMPORTANT: Original winners are determined from the FULL election (results[:k]),
    not from the reduced state after candidate removal. This ensures:
    - A, B, C always show gap=0 even if reduced state has different winners
    - D's gap reflects actual votes needed to join the winning set

    Args:
        strats_frame : dict
            Raw output from reach_any_winners_campaign
            Keys are winner combination strings, values are [cost, additions]
        k : int
            Number of winners
        results : list
            Social choice order from full election (results[:k] = actual winners)

    Returns:
        dict
            Mapping individual candidates to [victory_gap, strategy_detail]
            Used directly for webapp display

    Paper Reference: Section 3.3 "Multi-Winner Strategy Interpretation"
    """
    if not strats_frame or k <= 1:
        # For single-winner, the format is already per-candidate
        return strats_frame

    # The original winners are ALWAYS results[:k] from the full election
    # These are the ACTUAL winners we want to preserve with gap=0
    original_winners = set(results[:k])

    # Find the cost=0 combination in strats_frame (winners in the reduced state)
    # This may differ from original_winners if candidate removal changed vote distributions
    reduced_winners = None
    for combo_key, value in strats_frame.items():
        if isinstance(value, (list, tuple)) and len(value) >= 1:
            if value[0] == 0:
                reduced_winners = set(combo_key)
                break

    candidate_strats = {}

    # Original winners (from full election) get gap = 0
    for winner in original_winners:
        candidate_strats[winner] = [0.0, {}]

    # Get all candidates that appear in any combination in strats_frame
    all_candidates_in_strats = set()
    for combo_key in strats_frame.keys():
        all_candidates_in_strats.update(combo_key)

    # For non-winners, find minimum cost to become a winner
    # Iterate through candidates in results order, but only if they appear in strats_frame
    for candidate in results:
        if candidate in original_winners:
            continue  # Already handled as winner with gap=0
        if candidate not in all_candidates_in_strats:
            continue  # Not in reduced set, skip

        min_cost = float('inf')
        best_additions = {}

        # Look through all combinations that include this candidate
        for combo_key, value in strats_frame.items():
            # Handle both list/tuple and other formats
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                cost = value[0]
                additions = value[1]
            else:
                continue

            combo_set = set(combo_key)

            # Skip if this is the original winners combination
            if combo_set == original_winners:
                continue

            # Skip if this is the reduced winners combination with cost=0
            # This prevents non-winners from getting 0% gap due to reduced state artifacts
            if reduced_winners and combo_set == reduced_winners and cost == 0:
                continue

            # For non-winners from the ORIGINAL election:
            # Only accept cost > 0 since they need votes to win
            if candidate in combo_key and cost > 0 and cost < min_cost:
                min_cost = cost
                # Convert additions to dict format if needed
                if isinstance(additions, dict):
                    best_additions = additions
                elif isinstance(additions, (list, tuple)):
                    # If it's a list, convert to dict with candidate as key
                    best_additions = {candidate: min_cost}
                else:
                    best_additions = {candidate: min_cost}

        if min_cost != float('inf'):
            candidate_strats[candidate] = [min_cost, best_additions]
        # If no valid combination found (all had cost=0 or cost=inf),
        # candidate gap cannot be reliably computed from this reduced state

    return candidate_strats
    
def process_ballot_counts_post_elim_no_print(ballot_counts, k, candidates, elim_cands,
                                    check_strats=False, budget_percent=0,
                                    check_removal_here=False, keep_at_least=8, rigorous_check=True,
                                    spl_check=False, allowed_length=None):
    """
    Main strategy computation function for RCV elections.

    This function computes victory gaps and optimal strategies for all candidates
    in a Ranked Choice Voting election. It handles both single-winner (k=1) and
    multi-winner (k>1) elections with sophisticated early winner handling.

    Parameters:
    -----------
    ballot_counts : dict
        Dictionary mapping ballot strings to vote counts.
        Example: {'ABC': 100, 'BAC': 80} means 100 voters ranked A>B>C
    k : int
        Number of winners (1 for IRV, >1 for STV)
    candidates : list
        List of candidate identifiers (e.g., ['A', 'B', 'C', 'D'])
    elim_cands : list
        Candidates to pre-eliminate (usually empty [])
    check_strats : bool
        If True, compute optimal strategies. If False, only compute results.
    budget_percent : float
        Maximum vote addition as percentage of total votes.
        Used for candidate removal and strategy bounds.
    check_removal_here : bool
        If True, use remove_irrelevant() to reduce candidate count for tractability.
    keep_at_least : int
        Minimum candidates to retain after removal (typically 7-8 for Portland k=3).
    rigorous_check : bool
        If True, use Theorem 4.3 for edge cases in candidate removal.
    spl_check : bool
        If True, use special check in permit_STV_removal() for multi-winner.
    allowed_length : int or None
        Maximum ballot chain length for strategy computation.

    Returns:
    --------
    dict with keys:
        - "num_candidates": Total candidate count
        - "total_votes": Total vote count
        - "quota": Droop quota Q
        - "candidates_removed": List of removed candidates
        - "candidates_retained": List of retained candidates
        - "winners_during_elims": Candidates exceeding Q after filtering
        - "Strategies": Dict mapping candidate -> [victory_gap, strategy_detail]
        - "overall_winning_order": Social choice order from STV

    Algorithm Overview (Multi-Winner k > 1):
    ----------------------------------------
    The key challenge is handling "early winners" - candidates who exceed quota Q
    after candidate removal. This can be an artifact of one-shot ballot filtering
    vs. actual STV round-by-round surplus transfers.

    We use the `collection` data structure from STV_optimal_result_simple():
    - collection[i] = ballot state after i STV events (eliminations + wins)
    - This respects proper Droop surplus transfers

    Logic Flow:
    1. Run STV to get results order and collection
    2. Call remove_irrelevant() to get tractable candidate set
    3. Filter ballots to retained candidates only
    4. Check for early winners (candidates >= Q after filtering)

    If early winner exists (multi-winner only):
        Step 0:  Check collection[small_num]. If no early winner there, use it directly.
        Step 0b: If collection[small_num] has winner, check one step back.
        Step A:  If collection has early winner, call permit_STV_removal().
                 If permitted, use "small election method":
                 - Get ballot state from collection[small_num]
                 - Compute strategies for remaining candidates with k-1 seats
                 - Add early winner back with gap = 0
        Step B:  If permit fails, try "loopy" - decrease keep_at_least until viable.
        Step C:  If loopy exhausted, reset - this budget doesn't work.

    If no early winner:
        Compute strategies directly on filtered_data with full k seats.

    Tractability Constraint:
    - Strategy computation is only tractable for < MAX_TRACTABLE_CANDIDATES
    - If len(candidates_retained) >= MAX_TRACTABLE_CANDIDATES, return empty strategies
    - This signals webapp's divide-and-conquer to try lower budget

    Paper References:
    - Theorem 4.1 (Candidate Removal): "Optimal Strategies in RCV"
    - Theorem 4.2 (Irrelevant Extension): "Optimal Strategies in RCV"
    - Small Election Method: Section 5.2 of the paper

    Example Usage:
    --------------
    >>> result = process_ballot_counts_post_elim_no_print(
    ...     ballot_counts={'ABC': 100, 'BAC': 80, 'CAB': 60},
    ...     k=1, candidates=['A', 'B', 'C'], elim_cands=[],
    ...     check_strats=True, budget_percent=10.0,
    ...     check_removal_here=False
    ... )
    >>> result['Strategies']
    {'A': [0.0, {}], 'B': [4.17, {'B': 4.17}], 'C': [12.5, {'C': 12.5}]}
    """
    # Create a filtered ballot count dictionary by removing eliminated candidates from keys.
    filtered_data = {}
    elim_strings = ''.join(elim_cands)
    for key, value in ballot_counts.items():
        new_key = ''.join(char for char in key if char not in elim_strings)
        filtered_data[new_key] = filtered_data.get(new_key, 0) + value
    filtered_data.pop('', None)

    # Remaining candidates after elimination.
    elec_cands = [cand for cand in candidates if cand not in elim_cands]

    # Calculate full and filtered vote counts.
    full_aggre_v_dict = utils.get_new_dict(ballot_counts)
    aggre_v_dict = utils.get_new_dict(filtered_data)

    # Total votes and quota (Q).
    total_votes = sum(full_aggre_v_dict.get(cand, 0) for cand in candidates)
    Q = round(total_votes / (k + 1) + 1, 3)
    if k==1:
        Q = Q*(k+1)

    # Calculate strict support for eliminated candidates.
    letter_counts = {}
    for key, value in ballot_counts.items():
        i = 0
        newset = []
        if key and key[i] in candidates:
            while i < len(key) and key[i] in elim_strings:
                newset.append(key[i])
                i += 1
        new_key = ''.join(newset)
        for letter in new_key:
            letter_counts[letter] = letter_counts.get(letter, 0) + value

    # Identify winners among the remaining candidates.
    wins_during_elims = [c for c in elec_cands if aggre_v_dict.get(c, 0) >= Q]

    # Run the STV algorithm to determine overall winning order.
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, subresults = utils.return_main_sub(rt)
    budget = budget_percent * total_votes * 0.01
    zeros, c_l = predict_losses(ballot_counts, results, k, Q, budget)

    # Check candidate removal if requested.
    candidates_removed = []
    candidates_retained = candidates
    group_remaining = ''

    if check_removal_here:
        candidates_reduced, group_remaining, stop = remove_irrelevant(
            ballot_counts, rt, results[:keep_at_least], budget, ''.join(results), rigorous_check
        )
        if stop:
            # Removal succeeded - use whatever candidates were retained
            candidates_retained = candidates_reduced
            candidates_removed = [c for c in results if c not in candidates_retained]
            #group_remaining = ''.join(candidates_removed)
        else:
            # Removal failed - keep all candidates
            candidates_removed = []
            candidates_retained = candidates

    strats_frame_percent = {}
    strats_frame = {}

    # Check strategies if requested
    if len(candidates) > 2:
        if len(candidates_retained) < 1:
            # Empty or single candidate - budget doesn't support keep_at_least
            # Return empty strategies so webapp's divide-and-conquer can try a lower budget
            strats_frame = {}
            strats_frame_percent = {}
        elif len(candidates_retained) >= 1 and len(candidates_retained) < len(candidates):
            # Some candidates were removed - filter ballots to only include retained candidates
            filtered_data = {}
            #elim_strings = ''.join(candidates_removed) if isinstance(candidates_removed, str) else ''.join(candidates_removed)
            for key, value in ballot_counts.items():
                new_key = ''.join(char for char in key if char not in group_remaining)
                filtered_data[new_key] = filtered_data.get(new_key, 0) + value
            filtered_data.pop('', None)

            agg_v_dict = utils.get_new_dict(filtered_data)

            # ============================================================
            # MULTI-WINNER EARLY WINNER HANDLING (k > 1 only)
            # ============================================================
            #
            # PROBLEM:
            # After candidate removal, filtered_data combines all vote transfers
            # at once (one-shot removal). A candidate may exceed quota Q here,
            # but in the actual STV (which does round-by-round surplus transfers),
            # that candidate may not have won yet at the corresponding round.
            #
            # SOLUTION: Use the `collection` data structure
            # - collection[i] = ballot state after i STV events
            # - small_num = len(candidates) - len(candidates_retained)
            # - collection[small_num] shows actual ballot state at that point
            #
            # KEY INSIGHT: filtered_data vs collection[small_num]
            # - filtered_data: One-shot removal, all votes transfer immediately
            # - collection[small_num]: Actual STV state with proper Droop transfers
            # Both have the SAME candidates retained, but DIFFERENT vote distributions
            #
            # LOGIC FLOW:
            # Step 0:  Check collection[small_num]. If no early winner, use directly.
            #          Example: Portland Dis 2 - 5 retained, no early winner in collection
            # Step 0b: If collection[small_num] has winner, try one step back.
            # Step A:  If collection has early winner, call permit_STV_removal().
            #          If permitted, use "SMALL ELECTION METHOD":
            #          - Treat early winner as already elected
            #          - Compute strategies for remaining k-1 seats
            #          - Add early winner back with gap = 0
            #          Example: Portland Dis 4 - A wins early, compute for B,C,D with k=2
            # Step B:  If permit fails, try "loopy" - smaller candidate pools
            # Step C:  If loopy exhausted, reset - this budget doesn't work
            #
            # TRACTABILITY CONSTRAINT:
            # Strategy computation only works for < MAX_TRACTABLE_CANDIDATES.
            # If len(ordered_test) >= MAX_TRACTABLE_CANDIDATES, skip computation and return empty.
            # This signals webapp's binary search to try a lower budget.
            #
            # Paper Reference: Theorem 4.2 "Irrelevant Extension"
            # ============================================================
            early_winner_handled = False

            if k > 1:  # Multi-winner only — k=1 skips entirely
                early_winners = [c for c in candidates_retained if agg_v_dict.get(c, 0) >= Q]

                if early_winners:
                    stv_viable = False

                    # --- Step 0: Check collection state for actual early winner ---
                    #
                    # The filtered_data combines all transfers at once, but in the
                    # actual STV the winner may not have been declared yet at this point.
                    #
                    # Calculation: small_num = len(candidates) - len(candidates_retained)
                    # This is the number of STV events (eliminations + wins) that have occurred
                    # to reach a state with len(candidates_retained) candidates.
                    #
                    # Note: Event N in STV might be a WIN (not elimination). If so,
                    # collection[N] is post-win state. We may need to check collection[N-1]
                    # (one step back, before the win).
                    small_num = len(candidates) - len(candidates_retained)
                    if small_num < len(collection):
                        ballot_counts_short = collection[small_num][0]
                        agg_collection = utils.get_new_dict(ballot_counts_short)
                        active_in_collection = set(key[0] for key in ballot_counts_short.keys() if key)
                        collection_early = [c for c in candidates_retained if agg_collection.get(c, 0) >= Q]
                        # Also check for retained candidates missing from collection
                        # (already elected and removed from the pool — still an early winner)
                        missing_from_collection = [c for c in candidates_retained if c not in active_in_collection]

                        if not collection_early and not missing_from_collection:
                            # No early winner in actual STV state — compute directly
                            stv_viable = True
                            if check_strats:
                                # Filter candidates_retained to only those in collection
                                valid_cands = [c for c in candidates_retained if c in active_in_collection]
                                ordered_test = sorted(valid_cands, key=lambda x: results.index(x))
                                # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                    strats_frame = reach_any_winners_campaign_parallel(
                                        ordered_test, k, Q, ballot_counts_short, budget,
                                        c_l=[], zeros=0, allowed_length=allowed_length
                                    )
                                    early_winner_handled = True

                        elif small_num > 0:
                            # Step 0b: collection[small_num] has early winner — check
                            # one step back (before the win event).
                            ballot_counts_back = collection[small_num - 1][0]
                            agg_back = utils.get_new_dict(ballot_counts_back)
                            active_back = sorted(set(key[0] for key in ballot_counts_back.keys() if key))
                            back_early = [c for c in active_back if agg_back.get(c, 0) >= Q]

                            if not back_early and all(c in active_back for c in candidates_retained):
                                # One step back: no early winner, tractable set
                                candidates_retained = active_back
                                candidates_removed = [c for c in results if c not in active_back]
                                stv_viable = True
                                if check_strats:
                                    # active_back is already filtered to valid candidates
                                    ordered_test = sorted(active_back, key=lambda x: results.index(x))
                                    # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                    if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                        strats_frame = reach_any_winners_campaign_parallel(
                                            ordered_test, k, Q, ballot_counts_back, budget,
                                            c_l=[], zeros=0, allowed_length=allowed_length
                                        )
                                        early_winner_handled = True

                    # --- Step A: Collection has early winner — try permit ---
                    #
                    # permit_STV_removal() checks if the early winner's position is
                    # robust against the removed candidates. If True, we can treat
                    # the early winner as already elected and compute for k-1 seats.
                    #
                    # SMALL ELECTION METHOD (when permit passes):
                    # 1. Use collection[small_num] - ballot state with early winner's
                    #    surplus already transferred via Droop method
                    # 2. Get remaining candidates from rt[small_num:] (elimination trace)
                    # 3. Compute strategies for these candidates competing for k-1 seats
                    #    (because the early winner already took one seat)
                    # 4. Add early winner back to strats_frame with gap = 0
                    #
                    # Example: Portland Dis 4 (k=3)
                    # - A wins early in round 7 (after 5 eliminations)
                    # - collection[26] has A's surplus transferred to B, C, D, F
                    # - We compute strategies for B, C, D with k-1=2 seats
                    # - A gets gap=0, D gets gap=1.09% (needs 835 votes to join top-3)
                    #
                    # WHY k-1? The early winner already occupies one seat. The remaining
                    # candidates compete for the remaining k-1 seats. Using full k would
                    # give wrong gaps (e.g., 0.325% instead of 1.09% for D).
                    #
                    # Guard: permit needs at least 2 retained candidates to be meaningful
                    if not stv_viable and len(candidates_retained) > 1:
                        for cw in early_winners:
                            permitted = permit_STV_removal(
                                cw, ballot_counts, Q, candidates_retained,
                                group_remaining, budget_percent, spl_check=spl_check
                            )
                            if permitted:
                                stv_viable = True
                                if check_strats:
                                    # Use collection[small_num] - state after early winner removed
                                    # Get candidates from rt (elimination trace)
                                    if small_num < len(collection):
                                        ballot_counts_short = collection[small_num][0]
                                        test = [rt[i][0] for i in range(small_num, len(rt))]
                                        ordered_test = sorted(test, key=lambda x: results.index(x))
                                        # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                        if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                            # CRITICAL: Use k-1 for remaining seats after early winner
                                            # The early winner already occupies one seat
                                            strats_frame = reach_any_winners_campaign_parallel(
                                                ordered_test, k - 1, Q, ballot_counts_short, budget,
                                                c_l=[], zeros=0, allowed_length=allowed_length
                                            )
                                            # Add early winner with 0 gap (they already won)
                                            strats_frame[cw] = [0, {}]
                                            early_winner_handled = True
                                break

                    # --- Step B: If permit failed, try loopy (shorter pools) ---
                    if not stv_viable:
                        current_keep = len(candidates_retained) - 1
                        while current_keep > k and not stv_viable:
                            cands_try, group_try, stop_try = remove_irrelevant(
                                ballot_counts, rt, results[:current_keep], budget,
                                ''.join(results), rigorous_check
                            )
                            if not stop_try:
                                current_keep -= 1
                                continue

                            # Check collection state at this level
                            sn_try = len(candidates) - len(cands_try)
                            if sn_try < len(collection):
                                bc_try = collection[sn_try][0]
                                agg_coll_try = utils.get_new_dict(bc_try)
                                coll_early_try = [c for c in cands_try if agg_coll_try.get(c, 0) >= Q]

                                if not coll_early_try:
                                    # No early winner in collection — compute directly
                                    candidates_retained = cands_try
                                    candidates_removed = [c for c in results if c not in cands_try]
                                    group_remaining = group_try
                                    stv_viable = True
                                    if check_strats:
                                        # Filter cands_try to only those present in bc_try
                                        active_try = set(key[0] for key in bc_try.keys() if key)
                                        valid_cands = [c for c in cands_try if c in active_try]
                                        ordered_test = sorted(valid_cands, key=lambda x: results.index(x))
                                        # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                        if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                            strats_frame = reach_any_winners_campaign_parallel(
                                                ordered_test, k, Q, bc_try, budget,
                                                c_l=[], zeros=0, allowed_length=allowed_length
                                            )
                                            early_winner_handled = True
                                    break

                                # Collection still has early winner — try permit
                                for cw in coll_early_try:
                                    permitted = permit_STV_removal(
                                        cw, ballot_counts, Q, cands_try, group_try,
                                        budget_percent, spl_check=spl_check
                                    )
                                    if permitted:
                                        candidates_retained = cands_try
                                        candidates_removed = [c for c in results if c not in cands_try]
                                        group_remaining = group_try
                                        stv_viable = True
                                        if check_strats:
                                            # Use collection[sn_try] - state after early winner
                                            if sn_try < len(collection):
                                                bc_small = collection[sn_try][0]
                                                test = [rt[i][0] for i in range(sn_try, len(rt))]
                                                ordered_test = sorted(test, key=lambda x: results.index(x))
                                                # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                                if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                                    # Use k-1 for remaining seats
                                                    strats_frame = reach_any_winners_campaign_parallel(
                                                        ordered_test, k - 1, Q, bc_small, budget,
                                                        c_l=[], zeros=0, allowed_length=allowed_length
                                                    )
                                                    strats_frame[cw] = [0, {}]
                                                    early_winner_handled = True
                                    break

                            if not stv_viable:
                                current_keep -= 1

                    # --- Step C: If loopy exhausted, reset — this budget doesn't work ---
                    if not stv_viable:
                        candidates_removed = []
                        candidates_retained = candidates
                        group_remaining = ''

            # If no early winner or single-winner, use standard approach
            if not early_winner_handled and check_strats:
                if all(agg_v_dict.get(cand, 0) < Q for cand in candidates_retained):
                    # No immediate winners, compute strategies for all retained candidates
                    # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                    if len(candidates_retained) < MAX_TRACTABLE_CANDIDATES:
                        strats_frame = reach_any_winners_campaign_parallel(
                            candidates_retained, k, Q, filtered_data, budget,
                            c_l=[], zeros=0, allowed_length=allowed_length
                        )
                # If early winner exists but loopy exhausted, strats_frame stays empty
                # This signals to webapp's binary search to try a lower budget

            # Convert combination-based strategies to per-candidate strategies for multi-winner
            if k > 1 and strats_frame:
                strats_frame = convert_combination_strats_to_candidate_strats(strats_frame, k, results)

            strats_frame_percent = convert_to_percentage(strats_frame, total_votes)
        else:
            # No candidates removed by remove_irrelevant at initial keep_at_least.
            # Try loopy: decrease keep_at_least until removal succeeds
            if check_strats and k > 1:
                current_keep = keep_at_least - 1
                loopy_success = False
                while current_keep > k and not loopy_success:
                    cands_try, group_try, stop_try = remove_irrelevant(
                        ballot_counts, rt, results[:current_keep], budget,
                        ''.join(results), rigorous_check
                    )
                    if stop_try:
                        # Removal succeeded at this level
                        sn_try = len(candidates) - len(cands_try)
                        if sn_try < len(collection):
                            bc_try = collection[sn_try][0]
                            agg_coll_try = utils.get_new_dict(bc_try)
                            coll_early_try = [c for c in cands_try if agg_coll_try.get(c, 0) >= Q]

                            if not coll_early_try:
                                # No early winner - compute directly
                                candidates_retained = cands_try
                                candidates_removed = [c for c in results if c not in cands_try]
                                active_try = set(key[0] for key in bc_try.keys() if key)
                                valid_cands = [c for c in cands_try if c in active_try]
                                ordered_test = sorted(valid_cands, key=lambda x: results.index(x))
                                # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                    strats_frame = reach_any_winners_campaign_parallel(
                                        ordered_test, k, Q, bc_try, budget,
                                        c_l=[], zeros=0, allowed_length=allowed_length
                                    )
                                    loopy_success = True
                            else:
                                # Early winner exists - try permit with small election
                                for cw in coll_early_try:
                                    permitted = permit_STV_removal(
                                        cw, ballot_counts, Q, cands_try, group_try,
                                        budget_percent, spl_check=spl_check
                                    )
                                    if permitted:
                                        candidates_retained = cands_try
                                        candidates_removed = [c for c in results if c not in cands_try]
                                        # Use collection[sn_try] - state after early winner
                                        if sn_try < len(collection):
                                            bc_small = collection[sn_try][0]
                                            test = [rt[i][0] for i in range(sn_try, len(rt))]
                                            ordered_test = sorted(test, key=lambda x: results.index(x))
                                            # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                            if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                                # Use k-1 for remaining seats
                                                strats_frame = reach_any_winners_campaign_parallel(
                                                    ordered_test, k - 1, Q, bc_small, budget,
                                                    c_l=[], zeros=0, allowed_length=allowed_length
                                                )
                                                strats_frame[cw] = [0, {}]
                                                loopy_success = True
                                        break
                    current_keep -= 1

                if loopy_success and k > 1 and strats_frame:
                    strats_frame = convert_combination_strats_to_candidate_strats(strats_frame, k, results)
                    strats_frame_percent = convert_to_percentage(strats_frame, total_votes)

            # Fallback for small elections or single-winner
            elif len(elec_cands) < 9 and check_strats:
                strats_frame = reach_any_winners_campaign_parallel(
                    elec_cands, k, Q, filtered_data, budget, c_l=[], zeros=0,
                    allowed_length=allowed_length
                )
                # Convert for multi-winner
                if k > 1 and strats_frame:
                    strats_frame = convert_combination_strats_to_candidate_strats(strats_frame, k, results)
                strats_frame_percent = convert_to_percentage(strats_frame, total_votes)

    if len(candidates) == 2:
        winner = results[0]
        loser = results[1]
        win_margin = filtered_data.get(winner, 0) - filtered_data.get(loser, 0) + 1
        strats_frame = {winner: [0.0, []], loser: [win_margin, {loser: win_margin}]}
        strats_frame_percent = convert_to_percentage(strats_frame, total_votes)

    return {
        "num_candidates": len(candidates),
        "total_votes": total_votes,
        "quota": Q,
        "candidates_removed": candidates_removed,
        "candidates_retained": candidates_retained,
        "winners_during_elims": wins_during_elims,
        "Strategies": strats_frame_percent,
        "overall_winning_order": results,
        "initial_zeros" : zeros,
        "def_losing": c_l
    }



# ============================================================================
# BOOTSTRAP FUNCTIONS
# ============================================================================

def generate_bootstrap_samples(data, n_samples=1000, save=False):
    """
    Generates bootstrap samples from the given dataset.

    Parameters:
    data (DataFrame): The original dataset containing RCV rankings.
    n_samples (int): The number of bootstrap samples to generate.
    save (bool): Whether to save the bootstrap samples.

    Returns:
    List[DataFrame]: A list containing the bootstrap samples.
    """
    bootstrap_samples = []
    n = len(data)
    for _ in range(n_samples):
        sample = data.sample(n, replace=True)  # Sampling with replacement
        bootstrap_samples.append(sample)

    if save:
        # Creating a directory to store all the bootstrap sample files
        output_dir = 'bootstrap_samples_new'
        os.makedirs(output_dir, exist_ok=True)

        # Saving each bootstrap sample as a separate CSV file
        for i, sample in enumerate(bootstrap_samples):
            sample_file_path = os.path.join(output_dir, f'bootstrap_sample_{i+1}.csv')
            sample.to_csv(sample_file_path, index=False)
        
    return bootstrap_samples

def process_bootstrap_samples(k, candidates_mapping, bootstrap_samples_dir, bootstrap_files, 
                              budget_percent, keep_at_least, iters=10, loopy_removal=False, 
                              want_strats=False, save=False, spl_check=False, allowed_length=None, rigorous_check=True):
    """
    Process multiple bootstrap samples to test algorithm efficiency.
    
    Parameters:
    k (int): Number of winners to select
    candidates_mapping (dict): Mapping from candidate names to identifiers
    bootstrap_samples_dir (str): Directory containing bootstrap samples
    bootstrap_files (list): List of bootstrap sample filenames
    budget_percent (float): Budget percentage for strategy calculation
    keep_at_least (int): Minimum number of candidates to keep
    iters (int): Maximum number of iterations
    loopy_removal (bool): Whether to try decreasing keep_at_least
    want_strats (bool): Whether to calculate strategies
    save (bool): Whether to save results
    spl_check (bool): Whether to do special check
    
    Returns:
    tuple: (algo_works, data_samples) - Algorithm success count and collected data
    """
    candidates = list(candidates_mapping.values())
    it = 0
    algo_works = 0
    data_samples = []
    
    for file in bootstrap_files:
        start = time.time()
        
        stop = False
        it += 1
    
        if it > iters:
            break

        #sample_file_path = os.path.join(bootstrap_samples_dir, file)
        df = bootstrap_samples_dir #pd.read_csv(sample_file_path)
    
        # Determine ballot counting method based on column names
        if 'rank1' in df.columns:
            ballot_counts = get_ballot_counts_df_republican_primary(candidates_mapping, df)
            Q = 800
      
            budget = budget_percent * sum(ballot_counts.values()) * 0.01
        else:
            ballot_counts = get_ballot_counts_df(candidates_mapping, df)
            Q = round(sum(ballot_counts.values())/(k+1)+1, 3) 
            budget = budget_percent * sum(ballot_counts.values()) * 0.01
        
    
        # Compute results using STV
    
        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
        results, subresults = utils.return_main_sub(rt)

        # Handle candidate removal based on approach
        if not loopy_removal:
            candidates_reduced, group, stop = remove_irrelevant(
                ballot_counts, rt, results[:keep_at_least], budget, ''.join(candidates), rigorous_check
            )
        else:
            # Start with original keep_at_least value and decrease until removal fails
            current_keep = keep_at_least
            while current_keep >= k:
                candidates_reduced, group, stop = remove_irrelevant(
                    ballot_counts, rt, results[:current_keep], budget, ''.join(candidates), rigorous_check
                )

                if not stop:
                    # If removal failed, revert to previous successful value
                    current_keep += 1
                    candidates_reduced, group, stop = remove_irrelevant(
                        ballot_counts, rt, results[:current_keep], budget, ''.join(candidates), rigorous_check
                    )
                    break
                current_keep -= 1
        
        print(f"Iteration {it}: Candidates = {candidates_reduced}")

        if stop:
            algo_works += 1

            if want_strats:
                # Initialize a dictionary for filtered data
                filtered_data = {}

                # Remove eliminated candidates while retaining the rest of the string
                for key, value in ballot_counts.items():
                    new_key = ''.join(char for char in key if char not in group)
                    filtered_data[new_key] = filtered_data.get(new_key, 0) + value
                filtered_data.pop('', None)

                agg_v_dict = utils.get_new_dict(filtered_data)

                # Check for immediate winners
                for cand_winner in candidates_reduced:
                    if agg_v_dict[cand_winner] >= Q and k > 1:   
                        print(cand_winner)
                        removal_permitted = permit_STV_removal(
                            cand_winner, ballot_counts, Q, candidates_reduced, 
                            group, budget_percent, spl_check=spl_check
                        )
                        
                        if not removal_permitted:
                            print(f"Error. Candidate {cand_winner} wins during elimination")
                        else:
                            # Use collection[small_election_number] - state after early winner
                            small_election_number = len(candidates) - len(candidates_reduced)
                            if small_election_number < len(collection):
                                bc_small = collection[small_election_number][0]
                                test = [rt[i][0] for i in range(small_election_number, len(rt))]
                                ordered_test = sorted(test, key=lambda x: results.index(x))
                                # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                                if len(ordered_test) < MAX_TRACTABLE_CANDIDATES:
                                    # Use k-1 for remaining seats
                                    strats_frame = reach_any_winners_campaign(
                                        ordered_test, k - 1, Q, bc_small, budget, allowed_length=allowed_length
                                    )
                                    strats_frame[cand_winner] = [0, {}]
                                    print("special removal permit is working")

                            break
                
                # If no immediate winners, check strategies for all remaining candidates
                if all(agg_v_dict[cand_winner] < Q for cand_winner in candidates_reduced):
                    # Only compute if tractable (< MAX_TRACTABLE_CANDIDATES)
                    if len(candidates_reduced) < MAX_TRACTABLE_CANDIDATES:
                        strats_frame = reach_any_winners_campaign(
                            candidates_reduced, k, Q, filtered_data, budget, allowed_length= allowed_length
                        )
                
                data_samples.append(strats_frame)

                # Save results if requested
                if save:
                    output_dir = 'strategy_optimization_results'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    save_path = os.path.join(output_dir, f"iteration_{it}.json")
                    with open(save_path, "w") as f:
                        json.dump({"iteration": it, "strats_frame": strats_frame}, f, indent=4)

                    print(f"Iteration {it}: strats_frame = {strats_frame}")
            
        end = time.time()    
        # print('Total time =', end - start)
    
    print(algo_works, it-1, ' Removal Efficiency is ', algo_works/(it-1)*100, '%')
    return algo_works, data_samples

# ============================================================================
# PERMITTING FUNCTIONS
# ============================================================================
#
# These functions implement Theorem 4.2 "Irrelevant Extension" from the paper.
#
# When we use remove_irrelevant() to eliminate candidates, a retained candidate
# may exceed quota Q in the filtered ballot state. This function checks whether
# this early winner's position is "robust" - meaning none of the removed
# candidates could have prevented their win, even with budget-sized additions.
#
# If permit returns True, we can safely use the "small election method":
# treat the early winner as already elected and compute strategies for k-1 seats.
# ============================================================================

def permit_STV_removal(cand_winner, ballot_counts, Q, candidates_reduced, group,
                      budget_percent, spl_check=False):
    """
    Check if early winner removal is permitted under STV rules.

    When a candidate exceeds quota Q after filtering, this function verifies
    the removal is valid under the "Irrelevant Extension" theorem (Theorem 4.2).

    The key insight: If candidate A exceeds Q after removal of group G, we check
    that no candidate in G could have prevented A's win by computing:
    1. Surplus transfers from A to candidates in G
    2. Vote redistributions that could occur
    3. Whether any candidate in G could have reached a position to block A

    Paper Reference: Theorem 4.2 "Irrelevant Extension" in "Optimal Strategies in RCV"

    Parameters:
    -----------
    cand_winner : str
        The candidate who exceeds Q after filtering (the "early winner")
    ballot_counts : dict
        Original ballot data (before any filtering)
    Q : float
        Droop quota
    candidates_reduced : list
        Candidates retained after removal
    group : str
        String of candidates that were removed
    budget_percent : float
        Budget threshold (% of total votes) for strategy computation
    spl_check : bool
        If True, also considers the weakest retained candidate in calculations.
        This makes the check more conservative (harder to pass).

    Returns:
    --------
    bool
        True if removal is valid and small election method can be used.
        False if the early winner might have been affected by removed candidates.

    Usage:
    ------
    If permit_STV_removal() returns True:
        - Use collection[small_num] for ballot state
        - Compute strategies with k-1 seats (early winner already elected)
        - Add early winner to strats_frame with gap = 0
    """
    surplusA = {}
    A_original = utils.get_new_dict(ballot_counts)[cand_winner]
    A_needs = Q - A_original
    candidates_still_in_elec = ''.join([c for c in candidates_reduced if c != cand_winner])
    
    if spl_check:
        probable_elec = min(candidates_still_in_elec, key=lambda x: utils.get_new_dict(ballot_counts)[x])
    else: 
        probable_elec = ''
        
    groupcopy = group + probable_elec
    groupcopy2 = deepcopy(groupcopy)

    # Calculate surplus for each candidate in group
    for cand in group:
        fil_data = {}
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in groupcopy.replace(cand, ""))
            fil_data[new_key] = fil_data.get(new_key, 0) + value
        fil_data.pop('', None)
        
        C_set = {}
        for cand_spl in candidates_still_in_elec.replace(probable_elec, ""):
            for key, value in ballot_counts.items():
                if key[0] in groupcopy2.replace(cand, ""):
                    if cand_spl in key:
                        if cand_winner in key and key.index(cand_winner) > key.index(cand_spl):
                            C_set[cand_spl] = C_set.get(cand_spl, 0) + value
        
        AtransfersL = 0 
        for key, value in fil_data.items():
            if key[0] == cand_winner:
                if len(key) > 1 and key[1] in groupcopy:
                    AtransfersL = AtransfersL + value

        SV0 = utils.get_new_dict(fil_data)[cand_winner] - Q 
        
        # Check if the list would be empty before trying to find the minimum
        candidates_after_replace = candidates_still_in_elec.replace(probable_elec, "")
        if candidates_after_replace:
            min_from_elec_cands = min([
                utils.get_new_dict(fil_data)[cande] + C_set.get(cande, 0) 
                for cande in candidates_after_replace
            ])
        else:
            # If there are no other candidates to compare against, use a default value
            # Infinity makes sense here since we're looking for the minimum
            min_from_elec_cands = 0
         
        # min_from_elec_cands = min([
        #     utils.get_new_dict(fil_data)[cande] + C_set.get(cande, 0) 
        #     for cande in candidates_still_in_elec.replace(probable_elec, "")
        # ])
        surplusA[cand] = SV0 * (AtransfersL) / (SV0 + A_original) - min_from_elec_cands + A_needs
    
    # Calculate final dictionaries for decision
    final_dict = {}
    another_dict = {}
    for cand in group:
        letter_counts = {}
        
        # Calculate strict support
        for key, value in ballot_counts.items():
            i = 0
            newset = []
            if key and key[0] not in [cand_winner]:
                while key[i] in groupcopy2 + cand_winner:
                    newset.append(key[i])
                    i = i + 1
                    if i >= len(key):
                        break

            new_key = ''.join(char for char in newset)
            if cand in new_key:
                if cand in letter_counts:
                    letter_counts[cand] += value
                else:
                    letter_counts[cand] = value
                    
        final_dict[cand] = -round(surplusA[cand] + letter_counts[cand], 2)
        another_dict[cand] = 2 * (
            letter_counts[cand] - strict_support(ballot_counts, group, '', cand)
        )
    
    # Make final decision
    maxdev = min(final_dict, key=final_dict.get)   
    dev_in_percent = round(final_dict[maxdev] / sum(ballot_counts.values()) * 100, 2)
    print(dev_in_percent)
    
    return dev_in_percent > budget_percent