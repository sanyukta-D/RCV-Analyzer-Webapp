"""
Single Transferable Vote (STV) and Instant Runoff Voting (IRV) Implementations
===============================================================================

Repository: https://github.com/sanyukta-D/Optimal_Strategies_in_RCV

This module implements core voting algorithms for Ranked Choice Voting elections.

Key Functions:
--------------
- IRV_optimal_result(): Single-winner IRV election simulation
- IRV_ballot_exhaust(): Track ballot exhaustion in single-winner IRV
- STV_ballot_exhaust(): Track ballot exhaustion in multi-winner STV
- STV_optimal_result_simple(): Multi-winner STV with collection tracking (MAIN FUNCTION)

Key Data Structures:
--------------------

ballot_counts : dict
    Maps ballot strings to vote counts.
    Example: {'ABC': 100, 'BAC': 80, 'CAB': 60}
    Meaning: 100 voters ranked A first, B second, C third

rt (result trace) : list of [candidate, is_winner] pairs
    Records each STV event in chronological order.
    is_winner = 1 if candidate won (reached quota Q), 0 if eliminated.

    Example: [['E', 0], ['D', 0], ['C', 0], ['A', 1], ['B', 1]]
    Meaning: E eliminated, D eliminated, C eliminated, A won, B won

    Usage: results, _ = return_main_sub(rt)
           results = ['A', 'B', 'C', 'D', 'E']  # Social choice order

dt (detailed trace) : list
    Similar to rt with additional metadata for debugging.

collection : list of (ballot_state, metadata) tuples
    CRITICAL DATA STRUCTURE for multi-winner strategy computation.

    collection[i] = ballot state after i STV events (eliminations + wins)
    collection[0] = initial state (all candidates active)
    collection[N] = state after N events

    Each ballot_state is a dict mapping ballot strings to vote counts.
    IMPORTANT: Vote counts may be fractional due to Droop surplus transfers.

    Example (Portland Dis 4, k=3):
    - 30 candidates total, A wins early in round 7
    - small_num = 30 - 4 = 26 events to reach 4 retained candidates
    - collection[26][0] = ballot state with A's surplus transferred to B, C, D, F
    - Use collection[26] for strategy computation via "small election method"

    Usage in case_study_helpers.py:
        small_num = len(candidates) - len(candidates_retained)
        ballot_counts_short = collection[small_num][0]
        # Compute strategies on ballot_counts_short with k-1 seats
        # (because early winner already occupies one seat)

Quota Calculation:
------------------
Droop Quota: Q = floor(total_votes / (k + 1)) + 1

For k=1 (single-winner): Q = total_votes / 2 + 1 (majority)
For k=3 (Portland): Q = total_votes / 4 + 1 (~25%)

When a candidate reaches Q, they win and their surplus (votes - Q) transfers
to next preferences proportionally.

Paper References:
-----------------
- "Optimal Strategies in Ranked Choice Voting" (STV algorithm, collection usage)
- "Simpler Than You Think" (Practical interpretation of results)
"""

from copy import deepcopy
from rcv_strategies.utils.helpers import get_new_dict


def IRV_optimal_result(cands, ballot_counts):
    """
    Produce optimal social choice order for Instant Runoff Voting (IRV) or Single-winner RCV.
    
    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        
    Returns:
        list: Ordered list of candidates from last eliminated to first.
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    
    # Eliminate candidates one by one
    for _ in range(len(candidates_remaining)):
        # Count first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }
        
        # Find candidate with fewest votes
        worst_candidate = min(vote_counts, key=vote_counts.get)
        
        # Remove candidate from remaining list and add to results
        candidates_remaining.remove(worst_candidate)
        results.insert(0, worst_candidate)
        
        # Redistribute votes from ballots containing eliminated candidates
        filtered_ballots = {}
        for ballot, count in ballot_counts.items():
            # Remove all eliminated candidates from ballots
            new_ballot = ''.join(char for char in ballot if char not in results)
            filtered_ballots[new_ballot] = filtered_ballots.get(new_ballot, 0) + count
        
        # Remove empty ballots
        filtered_ballots.pop('', None)
        
        # Recalculate aggregated votes
        aggregated_votes = get_new_dict(filtered_ballots)
        
    return results

def IRV_ballot_exhaust(cands, ballot_counts):
    """
    Produce ballot exhaustion for Instant Runoff Voting (IRV) or Single-winner RCV.
    
    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        
    Returns:
        list: Ordered list of candidates from last eliminated to first.
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    number_of_ballots = sum(ballot_counts.values())
    ballot_count_list = [number_of_ballots]    
    exhausted_ballots_list = []
    exhausted_ballots_dict = {}
    filtered_ballots = ballot_counts.copy()
    # Eliminate candidates one by one
    for _ in range(len(candidates_remaining)):
        # Count first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }
        
        # Find candidate with fewest votes
        worst_candidate = min(vote_counts, key=vote_counts.get)

        ballot_count_list.append(sum(filtered_ballots.values()))
        # Check if any ballots are exhausted
        exhausted_ballots_list.append(ballot_count_list[0] - ballot_count_list[-1])
        exhausted_ballots_dict[worst_candidate] = ballot_count_list[0] - ballot_count_list[-1]
        
        # Remove candidate from remaining list and add to results
        candidates_remaining.remove(worst_candidate)
        results.insert(0, worst_candidate)
        
        # Redistribute votes from ballots containing eliminated candidates
        filtered_ballots = {}
        for ballot, count in ballot_counts.items():
            # Remove all eliminated candidates from ballots
            new_ballot = ''.join(char for char in ballot if char not in results)
            filtered_ballots[new_ballot] = filtered_ballots.get(new_ballot, 0) + count
        
        # Remove empty ballots
        filtered_ballots.pop('', None)


        
        # Recalculate aggregated votes
        aggregated_votes = get_new_dict(filtered_ballots)
        
    return exhausted_ballots_list, exhausted_ballots_dict


def STV_ballot_exhaust(cands, ballot_counts, k, Q):
    """
    Calculate ballot exhaustion for STV elections.
    Replicates the proven approach from print_exhaustion_analysis
    that produced the paper's verified results.

    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        k (int): Number of winners.
        Q (float): Quota threshold for winning.

    Returns:
        tuple: Contains three elements:
            - List of incremental exhausted ballots at each round
            - Dictionary mapping each candidate to cumulative exhausted ballot count BEFORE their event
              (i.e., exhaustion that existed while the candidate was still active — matches IRV convention)
            - List of winners (for verification)
    """
    event_log, result_dict, round_history = STV_optimal_result_simple(cands, ballot_counts, k, Q)

    candidates_remaining = set(cands)
    used_ballots = {}       # ballots consumed by winners (winner-first)
    exhausted_ballots = {}  # cumulative exhausted ballots
    current_ballots = ballot_counts.copy()

    exhausted_ballots_list = []  # incremental exhaustion per round
    exhausted_ballots_dict = {}  # candidate -> cumulative exhaustion BEFORE their event
    winners = []
    prev_cumulative = 0

    for round_num, (candidate, is_winner) in enumerate(event_log, 1):
        if is_winner:
            winners.append(candidate)
            candidates_remaining.remove(candidate)
            # Mark ballots ranking this winner first as used
            for ballot, count in current_ballots.items():
                if ballot.startswith(candidate):
                    used_ballots[ballot] = used_ballots.get(ballot, 0) + count
        else:
            candidates_remaining.remove(candidate)

        # Count newly exhausted ballots (not used by winners, no remaining active candidates)
        new_exhausted = {}
        for ballot, count in current_ballots.items():
            if ballot not in used_ballots:
                if not any(c in candidates_remaining for c in ballot):
                    new_exhausted[ballot] = count

        for ballot, count in new_exhausted.items():
            exhausted_ballots[ballot] = exhausted_ballots.get(ballot, 0) + count

        cumulative = sum(exhausted_ballots.values())
        incremental = cumulative - prev_cumulative
        exhausted_ballots_list.append(incremental)
        exhausted_ballots_dict[candidate] = prev_cumulative  # BEFORE this event (matches IRV convention & paper)
        prev_cumulative = cumulative

        # Update current_ballots for next round from round_history
        if round_num < len(round_history):
            current_ballots = round_history[round_num][0]

    return exhausted_ballots_list, exhausted_ballots_dict, winners


def STV_optimal_result_simple(cands, ballot_counts, k, Q):
    """
    Compute STV election results with full state tracking (collection).

    This is the MAIN STV function used throughout the codebase. It runs a full
    STV election and returns not just the results, but a complete history of
    ballot states at each round (the "collection").

    Args:
        cands (list): List of candidate identifiers (e.g., ['A', 'B', 'C', ...])
        ballot_counts (dict): Ballot strings to vote counts
            Example: {'ABC': 100, 'BAC': 80} means 100 voters ranked A>B>C
        k (int): Number of winners
        Q (float): Droop quota (typically total_votes / (k+1) + 1)

    Returns:
        tuple: (rt, dt, collection)

        rt : list of [candidate, is_winner] pairs
            Result trace recording each event chronologically.
            is_winner = 1 if candidate won (reached Q), 0 if eliminated.

            Example: [['E', 0], ['D', 0], ['A', 1], ['C', 0], ['B', 1]]
            Meaning: E eliminated, D eliminated, A won, C eliminated, B won

            To get social choice order: results, _ = return_main_sub(rt)
            results = ['A', 'B', 'C', 'D', 'E'] (winner first, last eliminated last)

        dt : dict
            Detailed trace mapping candidates to win/loss status (1 or 0).

        collection : list of [ballot_state, round_number] pairs
            CRITICAL: Tracks ballot state after each event.

            collection[0] = [initial_ballot_counts, 0]
            collection[i] = [ballot_state_after_i_events, i]

            The ballot_state is a dict with potentially FRACTIONAL counts
            due to Droop surplus transfers.

            USAGE IN STRATEGY COMPUTATION:
            When using the "small election method" for multi-winner elections:

                # Number of events to reach retained candidate count
                small_num = len(candidates) - len(candidates_retained)

                # Get ballot state at that point
                ballot_counts_short = collection[small_num][0]

                # Compute strategies on this state with k-1 seats
                # (early winner already occupies one seat)
                strats = reach_any_winners_campaign(ordered_test, k-1, Q, ballot_counts_short, budget)

    Example:
        >>> rt, dt, collection = STV_optimal_result_simple(['A','B','C','D'], bc, k=2, Q=100)
        >>> len(collection)
        4  # Initial state + 3 events (2 winners, 1 eliminated, 1 final)
        >>> collection[2][0]  # Ballot state after 2 events
        {'BC': 45.5, 'CB': 30.2, ...}  # Note: fractional due to surplus transfer

    Paper Reference: Section 2 "STV Background" in "Optimal Strategies in RCV"
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    event_log = []
    result_dict = {candidate: 0 for candidate in cands}
   
    # Track ballot state in each round
    round_history = []
    current_round = 0
    round_history.append([ballot_counts, current_round])

    # Process candidates until all are either winners or eliminated
    while candidates_remaining:
        current_round += 1
        
        # Calculate first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }

        # Check if any candidate meets the quota
        winner = None
        for candidate, votes in vote_counts.items():
            if votes >= Q:
                # Select the candidate with the most votes who meets quota
                winner = max(vote_counts, key=vote_counts.get)
                
                # Record winner
                event_log.append([winner, 1])
                result_dict[winner] = 1
                candidates_remaining.remove(winner)
                results.append(winner)

                # Distribute surplus votes proportionally
                surplus = votes - Q
                if surplus > 0:
                    transfer_weight = surplus / votes
                    new_ballots = {}

                    # Redistribute votes
                    for ballot, count in ballot_counts.items():
                        if ballot.startswith(winner):
                            # Remove the winner from the ballot
                            new_ballot = ballot[1:]
                            new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count * transfer_weight
                        else:
                            if winner in ballot:
                                # Remove winner from anywhere in ballot
                                new_ballot = ''.join(char for char in ballot if char != winner)
                                new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
                            else:
                                # Keep ballot as is
                                new_ballots[ballot] = new_ballots.get(ballot, 0) + count
                    
                    # Clean up and update
                    new_ballots.pop('', None)
                    ballot_counts = new_ballots
                break

        # If no candidate meets the quota, eliminate the lowest
        if winner is None:
            if not vote_counts:  # Check if no candidates remain
                break
            
            # Find candidate with fewest votes
            loser = min(vote_counts, key=vote_counts.get)
            
            # Record elimination
            candidates_remaining.remove(loser)
            event_log.append([loser, 0])
            result_dict[loser] = 0
            results.append(loser)

            # Redistribute votes of the eliminated candidate
            new_ballots = {}
            for ballot, count in ballot_counts.items():
                # Remove the loser from anywhere in the ballot
                new_ballot = ''.join(char for char in ballot if char != loser)
                new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count

            # Clean up empty ballots
            new_ballots.pop('', None)
            ballot_counts = new_ballots

        # Recalculate votes and record round state
        aggregated_votes = get_new_dict(ballot_counts)
        round_history.append([ballot_counts, current_round])

    return event_log, result_dict, round_history


def create_STV_round_result_given_structure(structure, checked_candidates, remaining_candidates, ballot_counts, Q):
    """
    Compute the round-counts for a given structure (social choice order) using Single Transferable Vote (STV).

    Args:
        stdt (dict): Description of the structure.
        checked_candidates (list): List of candidates already checked.  
        remaining_candidates (list): List of remaining candidates.
        currentdict (dict): Dictionary with current vote counts.
        Q (float): Quota threshold for winning.

    Returns:
        new currentdict: Updated dictionary with current vote counts.
    """
    if len(checked_candidates) == 0:
        return ballot_counts
    currentdict = get_new_dict(ballot_counts)
    
    last_candidate = checked_candidates[-1]  # The most recently processed candidate
    new_ballot_counts = {last_candidate: currentdict[last_candidate] }
    if structure[last_candidate] == 0: 
        # Remove eliminated candidates while retaining the rest of the string
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char != last_candidate)
            new_ballot_counts[new_key] = new_ballot_counts.get(new_key, 0) + value
        new_ballot_counts.pop('', None)
        return new_ballot_counts
    
    else:
        # Distribute surplus votes proportionally
        surplus = currentdict[last_candidate] - Q
        if surplus > 0:
            transfer_weight = surplus / currentdict[last_candidate]

            # Redistribute votes
            for ballot, count in ballot_counts.items():
                if ballot.startswith(last_candidate):
                    # Remove the winner from the ballot
                    new_ballot = ballot[1:]
                    new_ballot_counts[new_ballot] = new_ballot_counts.get(new_ballot, 0) + count * transfer_weight
                else:
                    if last_candidate in ballot:
                        # Remove winner from anywhere in ballot
                        new_ballot = ''.join(char for char in ballot if char != last_candidate)
                        new_ballot_counts[new_ballot] = new_ballot_counts.get(new_ballot, 0) + count
                    else:
                        # Keep ballot as is
                        new_ballot_counts[ballot] = new_ballot_counts.get(ballot, 0) + count
            
            # Clean up and update
            new_ballot_counts.pop('', None)
            return new_ballot_counts
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in [last_candidate])
            new_ballot_counts[new_key] = new_ballot_counts.get(new_key, 0) + value
        new_ballot_counts.pop('', None)
        return new_ballot_counts


    