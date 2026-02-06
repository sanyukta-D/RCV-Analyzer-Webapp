"""
Candidate Removal Functions for Tractable Strategy Computation
===============================================================

Repository: https://github.com/sanyukta-D/Optimal_Strategies_in_RCV

PROBLEM:
Strategy computation is exponential in candidate count. For >= MAX_TRACTABLE_CANDIDATES,
direct computation is intractable (too slow and memory-intensive).

SOLUTION:
Remove candidates who provably cannot affect the election outcome within
the given budget. This reduces the candidate set to a tractable size.

Key Theorems (from "Optimal Strategies in RCV" paper):
------------------------------------------------------

Theorem 4.1 (Basic Removal):
    A candidate C in the removal group can be removed if:
    strict_support(C) < worst_retained_votes - budget

    Where strict_support(C) = votes ranking C first among the removal group.
    Intuition: Even if C gets all first-choice votes from the removal group,
    they still can't catch up to the worst retained candidate within budget.

Theorem 4.3 (Rigorous Check):
    Even if basic removal fails, we can still remove C if adding budget
    votes to C would not change who gets eliminated next. This handles
    edge cases where C has substantial support but still cannot change
    the winner order.

Functions:
----------
- strict_support(): Calculate votes where candidate is ranked first in removal group
- check_removal(): Verify if removal satisfies theorems (both basic and rigorous)
- remove_irrelevant(): Iteratively remove candidates until no more can be removed
- predict_losses(): Predict which candidates will definitely lose

Usage in Strategy Computation:
------------------------------
    # In case_study_helpers.py
    candidates_reduced, group, stop = remove_irrelevant(
        ballot_counts, rt, results[:keep_at_least], budget, ''.join(results), rigorous_check
    )

    if stop:
        # Removal succeeded - candidates_reduced is tractable (< MAX_TRACTABLE_CANDIDATES)
        # Compute strategies on this reduced set
        strats = reach_any_winners_campaign(candidates_reduced, k, Q, filtered_data, budget)

    # If stop=False, removal failed - try lower budget or smaller keep_at_least

TRACTABILITY CONSTRAINT:
    Strategy computation requires < MAX_TRACTABLE_CANDIDATES candidates.
    If len(candidates_reduced) >= MAX_TRACTABLE_CANDIDATES, the webapp's binary search
    should try a lower budget to achieve more aggressive removal.

Paper Reference: Section 4 "Candidate Removal" in "Optimal Strategies in RCV"
"""

from rcv_strategies.utils.helpers import get_new_dict
from rcv_strategies.constants import MAX_TRACTABLE_CANDIDATES
from operator import itemgetter

def strict_support(ballot_counts, lower_group, upper_group, candidate):
    """
    Calculate strict support for a candidate based on ballot preferences.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        lower_group: Group of candidates to consider for first choices
        upper_group: Group of candidates to exclude
        candidate: The specific candidate to calculate support for
        
    Returns:
        Integer count of strict support for the candidate
    """
    letter_counts = {}
    total_cands = lower_group + upper_group
    
    # Calculate strict support
    for key, value in ballot_counts.items():
        i = 0
        newset = []
        if key and key[i] in total_cands:
            while key[i] in lower_group and key[i] not in upper_group:
                newset.append(key[i])
                i += 1
                if i >= len(key):
                    break

        new_key = ''.join(char for char in newset)
        if candidate in new_key:
            if candidate in letter_counts:
                letter_counts[candidate] += value
            else:
                letter_counts[candidate] = value

    return letter_counts.get(candidate, 0)


def check_removal(candidates, group, ballot_counts, budget, rigorous_check=True):
    """
    Check if a group of candidates can be removed while retaining other candidates.
    
    Args:
        candidates: List of candidates to retain
        group: String of candidates to potentially remove
        ballot_counts: Dictionary of ballots and their counts
        budget: Available budget
        rigorous_check: Whether to perform the rigorous/advanced check
        
    Returns:
        Boolean indicating if removal is possible
    """
    # Calculate strict support for each candidate in group
    strict_support_dict = {}

    for key, value in ballot_counts.items():
        i = 0
        newset = []
        while i < len(key) and key[i] in group:
            newset.append(key[i])
            i += 1
        
        new_key = ''.join(char for char in newset)
        for letter in new_key:
            if letter in strict_support_dict:
                strict_support_dict[letter] += value
            else:
                strict_support_dict[letter] = value
    
    can_remove = False
   
    for best_c_irrelevant in strict_support_dict.keys():
        groupcopy = group
        mostly_irrelevant = groupcopy.replace(best_c_irrelevant, "")

        # Filter data by removing mostly_irrelevant letters
        filtered_data = {}
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in mostly_irrelevant)
            filtered_data[new_key] = filtered_data.get(new_key, 0) + value
        
        filtered_data.pop('', None)  # Remove empty key if exists
        aggre_v_dict = get_new_dict(filtered_data)
    
        # Get aggregated votes for relevant candidates
        relevant_aggre_dict = {c: aggre_v_dict[c] for c in candidates}
        worst_c_relevant = min(relevant_aggre_dict, key=relevant_aggre_dict.get)
        #print(int(relevant_aggre_dict[worst_c_relevant] - strict_support_dict[best_c_irrelevant]), worst_c_relevant, best_c_irrelevant)
        if int(relevant_aggre_dict[worst_c_relevant] - strict_support_dict[best_c_irrelevant]) + 1 >= budget:
            can_remove = True
        else:
            if not rigorous_check:  # Skip advanced check if disabled
                return False
            if len(candidates) < 3:
                return False
            # Check if addition changes who drops out after worst candidate removal
            last_three = sorted(relevant_aggre_dict.items(), key=itemgetter(1))[:3]
            
            # Budget not enough to benefit 2 candidates at the bottom
            if 2 * last_three[2][1] - last_three[1][1] - last_three[0][1] > budget:
                maybe_irrelevant = mostly_irrelevant + worst_c_relevant
                
                # Filter data again with new irrelevant set
                filtered_data = {}
                for key, value in ballot_counts.items():
                    new_key = ''.join(char for char in key if char not in maybe_irrelevant)
                    filtered_data[new_key] = filtered_data.get(new_key, 0) + value
                
                filtered_data.pop('', None)
                aggre_v_dict = get_new_dict(filtered_data)

                # Add budget votes to best_c_irrelevant
                aggre_v_dict[best_c_irrelevant] = aggre_v_dict[best_c_irrelevant] + budget

                # New candidate list that removes worst and adds best_c_irrelevant
                candidates_temp = candidates.copy()
                candidates_temp.remove(worst_c_relevant)
                candidates_temp.append(best_c_irrelevant)

                relevant_aggre_dict = {c: aggre_v_dict[c] for c in candidates_temp}
                new_best_c_irrelevant = min(relevant_aggre_dict, key=relevant_aggre_dict.get)

                if new_best_c_irrelevant == best_c_irrelevant:  # Still gets eliminated
                    can_remove = True
                else:
                    can_remove = False
                    return False
            else:
                return False
                
    return can_remove


def remove_irrelevant(ballot_counts, rt, startcandidates, budget, fullgroup, rigorous_check=True):
    """
    Iteratively remove candidates who cannot affect the election outcome.

    Starting from startcandidates (typically top-k from STV results), this
    function repeatedly tests if the remaining candidates can be removed.
    If removal fails at current size, it shrinks the retained set by one
    and retries, until either removal succeeds or no candidates remain.

    Algorithm:
    1. Start with startcandidates as the retained set
    2. group = all candidates not in retained set
    3. Call check_removal(retained, group, ballot_counts, budget)
    4. If check fails, remove last candidate from retained set and retry
    5. Continue until check passes or retained set is empty

    Args:
        ballot_counts : dict
            Dictionary mapping ballot strings to vote counts
        rt : list
            Result trace from STV (used for ordering, passed for compatibility)
        startcandidates : list
            Initial candidates to try retaining (e.g., results[:keep_at_least])
            Typically the top candidates from STV results
        budget : float
            Maximum vote addition in absolute votes (not percentage)
            Calculated as: budget = budget_percent * total_votes * 0.01
        fullgroup : str
            String containing all candidate identifiers
        rigorous_check : bool
            If True, use Theorem 4.3 (rigorous check) for edge cases.
            If False, only use Theorem 4.1 (basic check) - faster but less accurate.

    Returns:
        tuple: (candidates_retained, group_removed, stop)

        candidates_retained : list
            Candidates that must be kept for accurate strategy computation
        group_removed : str
            String of candidates that can be safely removed
        stop : bool
            True if removal succeeded (some candidates were removed)
            False if removal failed (couldn't remove anyone within budget)

    Example:
        >>> # Portland Dis 4: 30 candidates, k=3, budget=6%
        >>> candidates_retained, group, stop = remove_irrelevant(
        ...     ballot_counts, rt, results[:8], budget, ''.join(results), True
        ... )
        >>> stop
        True
        >>> candidates_retained
        ['A', 'B', 'C', 'D']  # Only 4 retained - tractable!
        >>> len(group)
        26  # 26 candidates can be removed

    Usage in case_study_helpers.py:
        if stop:
            # Removal succeeded - filter ballots and compute strategies
            filtered_data = filter_ballots(ballot_counts, group_removed)
            strats = reach_any_winners_campaign(candidates_retained, k, Q, filtered_data, budget)
        else:
            # Removal failed - try lower budget (webapp's binary search)
            return empty strategies to signal retry

    Paper Reference: Algorithm 1 "Iterative Candidate Removal"
    """
    candidatesnew = startcandidates
    group = ''.join(char for char in fullgroup if char not in candidatesnew)
   
    while not check_removal(candidatesnew, group, ballot_counts, budget, rigorous_check):
        candidatesnew = candidatesnew[:-1]
        group = ''.join(char for char in fullgroup if char not in candidatesnew)
        
        if len(candidatesnew) < 1:
            stop = False
            return candidatesnew, group, stop

    return candidatesnew, group, check_removal(candidatesnew, group, ballot_counts, budget, rigorous_check)


def predict_wins(ballot_counts, candidates, k, Q, budget):
    """
    Predict number of winning candidates.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        candidates: List of candidate identifiers
        k: Number of seats/positions
        Q: Quota value
        budget: Available budget
        
    Returns:
        Number of predicted winners (U_W)
    """
    C_W = []
    for cand in candidates:
        if budget + strict_support(ballot_counts, candidates, [], cand) > Q + budget/(k+1):
            C_W.append(cand)

    C_bar_W = [cand for cand in candidates if cand not in C_W]
    number_unique_ballots = 0
    
    for cand in C_W:
        # Call strict_support only for new unique ballots
        c_new = C_bar_W + [cand]
        support_count = strict_support(ballot_counts, c_new, [], cand)
        number_unique_ballots += support_count  # Accumulate only unique contribution

    U_W = min(k, int((budget + number_unique_ballots)/(Q + budget/(k+1))))
    return U_W


def predict_losses(ballot_counts, candidates, k, Q, budget):
    """
    Predict number of losing candidates in a row.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        candidates: List of candidate identifiers
        k: Number of seats/positions
        Q: Quota value
        budget: Available budget
        
    Returns:
        Number of predicted losers (i_L)
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    first_choice_votes = [aggre_v_dict[i] for i in candidates]
    first_choice_votes = sorted(first_choice_votes, reverse=True)

    C_L = []
    for cand in candidates:
        if budget + strict_support(ballot_counts, candidates, [], cand) <= first_choice_votes[k-1]:
            C_L.append(cand)

    T = []
    for c_i in C_L:
        t_i = 0
        for ballot, count in ballot_counts.items():
            # Check if this ballot starts with c_i
            if len(ballot) > 0 and ballot[0] == c_i:
                # Look at subsequent candidates in the ballot
                for cand in ballot[1:]:
                    if cand not in C_L:
                        t_i += count  # Found a candidate outside C_L -> add once
                        break         # Don't check further; we already transferred
        T.append(t_i)
    
    T = sorted(T, reverse=True)
    
    i_L = 1
    if len(T) == 0:
        return 0, []
    
    while i_L < len(T) and sum(T[:i_L]) + first_choice_votes[0] < Q + budget/(k+1):
        i_L += 1
        
    return i_L, C_L