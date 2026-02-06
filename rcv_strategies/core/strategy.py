"""
Strategy Computation for RCV Elections
=======================================

Repository: https://github.com/sanyukta-D/Optimal_Strategies_in_RCV

This module computes optimal vote addition strategies for candidates in
Ranked Choice Voting elections. Given a budget (max votes to add), it
finds the minimum votes each candidate needs to win.

Core Concept - Structures:
--------------------------
A "structure" specifies a complete election outcome:
- main: The winners in order (e.g., 'AB' for k=2 means A wins first, then B)
- sub: The elimination order for non-winners (e.g., 'DC' means D eliminated first)

Algorithm:
----------
1. Enumerate all possible (main, sub) structures (winner combinations)
2. For each structure, compute minimum vote additions needed to achieve it
3. For each candidate, find minimum cost across all structures where they win
4. Return mapping: candidate -> [minimum_cost, strategy_detail]

Key Functions:
--------------
- reach_any_winners_campaign(): Find minimum votes for any winning set
- reach_any_winners_campaign_parallel(): Parallelized version (faster)
- process_campaign_STV(): Compute votes needed for a specific structure
- add_campaign(): Calculate round-by-round vote additions

Output Format:
--------------
For single-winner (k=1):
    {'A': [0, {}], 'B': [150, {'B': 150}], 'C': [300, {'C': 300}]}

For multi-winner (k=3):
    {'ABC': [0, {}], 'ABD': [150, {'D': 150}], ...}
    (Use convert_combination_strats_to_candidate_strats for per-candidate format)

Tractability:
-------------
Strategy computation is exponential in candidate count.
CONSTRAINT: Only tractable for < MAX_TRACTABLE_CANDIDATES candidates.

For larger elections, use remove_irrelevant() (candidate_removal.py) to reduce
the candidate set before calling strategy functions.

Paper Reference: Section 3 "Strategy Computation Algorithm"
"""

from copy import deepcopy
from rcv_strategies.constants import MAX_TRACTABLE_CANDIDATES
from rcv_strategies.utils.helpers import (
    get_new_dict,
    clean_aggre_dict_diff,
    create_structure,
    give_winners,
    str_for_given_winners,
    str_for_given_winners_losers,
    return_main_sub,
    main_structures,
    sub_structures,
    sub_structures_at_most_k_ones_fixed_last,
    campaign_addition_dict_simple
)
from rcv_strategies.core.optimization import (
    roundupdate,
    decode_dict,
    STV_optimal_result)
from rcv_strategies.core.stv_irv import (
    STV_optimal_result_simple,
    create_STV_round_result_given_structure)
from itertools import combinations, permutations
import math
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import atexit


# Set multiprocessing start method BEFORE your custom imports
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

# Global reference to active pool — allows cleanup from atexit/Streamlit rerun
_active_pool = None

def _cleanup_pool():
    """Kill any lingering pool workers. Registered with atexit."""
    global _active_pool
    if _active_pool is not None:
        try:
            _active_pool.terminate()
            _active_pool.join(timeout=5)
        except Exception:
            pass
        _active_pool = None

atexit.register(_cleanup_pool)

def add_campaign(log_campaign_list, main_st, remaining_candidates, decoded_dict, Q, k, t, stdt, budget):
    """
    Returns the campaign investment needed for a round, given budget.
    
    Args:
        log_campaign_list: List of campaign dictionaries for previous rounds
        main_st: Main structure
        remaining_candidates: List of candidates still in the race
        decoded_dict: Dictionary with current vote counts
        Q: Quota value
        k: Number of seats
        t: Current candidate being evaluated
        stdt: Structure dictionary indicating win/loss status
        budget: Available budget
        
    Returns:
        Tuple of (updated campaign list, boolean indicating success)
    """
    current_campaign_dict = {j: 0 for j in main_st} 
          
    if stdt[t] == 1:  # If candidate is a winner
        v = []
        for candidate in remaining_candidates:
            v.append(decoded_dict[candidate])
        t_update = max(0, (max(v) - decoded_dict[t]) + 1, (Q - decoded_dict[t]) + 1)
        if t_update > budget:
            return {}, False
        current_campaign_dict[t] = t_update
    else:  # If candidate is a loser
        for candidate in remaining_candidates:
            if decoded_dict[candidate] >= Q:
                return {}, False
            else:
                can_update = max(0, decoded_dict[t] - decoded_dict[candidate] + 1)
                if can_update > budget:
                    return {}, False
                current_campaign_dict[candidate] = can_update 
            
    log_campaign_list.append(deepcopy(current_campaign_dict))
    return log_campaign_list, True


def process_campaign_STV(candidates, main, sub, k, Q, aggre_v_dict, budget):
    """
    Given a structure and voter data and an updated Q, this gives a round-wise campaign 
    allocation requirement to make the structure feasible.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q: Quota value
        aggre_v_dict: Dictionary of aggregate votes
        budget: Available budget
        
    Returns:
        Tuple containing (decoded dictionaries, structure, campaign list, status list)
    """
    strt, stdt = create_structure(main, sub)
    currentdict = {can: [[can]] for can in candidates}

    remaining_candidates = list(stdt.keys())
    checked_candidates = []
    
    CheckDicts = []  # List of additions each round should have
    DecodedDicts = []  # List of how votes look in each round
    status_list = []
    log_campaign_list = []

    for i in range(len(candidates) - 1):
        # Update according to current round of elimination/win
        currentdict = roundupdate(stdt, checked_candidates, remaining_candidates, currentdict)

        # Convert into numerical data
        decoded_dict = decode_dict(currentdict, candidates, Q, aggre_v_dict)

        # Perform the next elimination/win
        t = remaining_candidates.pop(0) 
        checked_candidates.append(t)
        
        # Check if the next elimination/win is alright and calculate required additions
        log_campaign_list, status = add_campaign(
            log_campaign_list, main, remaining_candidates, decoded_dict, Q, k, t, stdt, budget
        )
        if status == False:
            return {}, [], [], [False]
            
        status_list.append(status)
        DecodedDicts.append(decoded_dict)
        
    return DecodedDicts, strt, log_campaign_list, status_list


def process_campaign_STV_simple(candidates, main, sub, k, Q, ballot_counts, budget):
    """
    Simplified version of process_campaign_STV that uses pre-calculated ballot counts.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        collections: Pre-calculated ballot collections
        
    Returns:
        Tuple containing (decoded dictionaries, structure, campaign list, status list)
    """
    strt, stdt = create_structure(main, sub)
    ballot_counts_new = deepcopy(ballot_counts)

    remaining_candidates = list(stdt.keys())
    checked_candidates = []
    DecodedDicts = []
    status_list = []
    log_campaign_list = []

    for i in range(len(candidates) - 1):

        # Update according to structre-specified elimination/win
        ballot_counts_new = create_STV_round_result_given_structure(stdt, checked_candidates, remaining_candidates, ballot_counts_new, Q)
        aggredict = get_new_dict(ballot_counts_new)
        currentdict = {can: aggredict[can] for can in remaining_candidates}
    
        # Perform the next elimination/win
        t = remaining_candidates.pop(0) 
        checked_candidates.append(t)
        
        
        # Check and calculate required campaign additions
        log_campaign_list, status = add_campaign(
            log_campaign_list, main, remaining_candidates, currentdict, Q, k, t, stdt, budget
        )
        if status == False:
            return {}, [], [], [False]
            
        status_list.append(status)
        DecodedDicts.append(currentdict)
        
    return DecodedDicts, strt, log_campaign_list, status_list


def smart_campaign(candidates, log_campaign_list, strt, stdt, Q, DecodedDicts, budget, allowed_length=None):
    """
    Takes the round-wise requirement dict and returns the optimized investment strategy.
    
    Args:
        candidates: List of candidate identifiers
        log_campaign_list: List of campaign dictionaries by round
        strt: Structure list
        stdt: Structure dictionary
        Q: Quota value
        DecodedDicts: List of decoded vote dictionaries by round
        budget: Available budget
        
    Returns:
        Tuple of (investment dictionary, amount spent)
    """
    if allowed_length is None:
        allowed_length = len(candidates)
    log = deepcopy(log_campaign_list)
    dict1 = log[0]  # First round allocations
    invest_dict = {can: 0 for can in candidates}  # Current round investments
    total_investment_dict = {can: 0 for can in candidates}  # Total investments
    amount_spent = 2*budget
    # Available funds from previous investments
    avl_dict = {can: 0 for can in candidates} 
    
    # First round: everyone gets what's needed
    for can in dict1.keys():
        if dict1[can] > 0:
            invest_dict[can] = dict1[can]
            total_investment_dict[can] = dict1[can]
             
    t = strt[0][0]  # First candidate for elimination or win
    decoded_dict = DecodedDicts[0]
    
    # Process subsequent rounds with reallocation
    for rd in range(len(log) - 1):
        # Update log of checked candidates and dissipate investment
        if stdt[t] == 1 and invest_dict[t] > 0:
            dee_invest = deepcopy(invest_dict)
            avl_dict[t] = math.floor(
                (decoded_dict[t] + dee_invest[t] - Q) * 
                (dee_invest[t] / (decoded_dict[t] + dee_invest[t]))
            )
            invest_dict[t] = 0
            
        if stdt[t] == 0 and invest_dict[t] > 0:
            for rich_can in total_investment_dict.keys():
                if rich_can[len(rich_can) - 1] == t:  # Investments owned by t
                    avl_dict[rich_can] = deepcopy(total_investment_dict)[rich_can]
            invest_dict[t] = 0
        
        # Process the next round
        dict_rd = log[rd + 1]
        decoded_dict = DecodedDicts[rd + 1]
        
        for can in dict_rd.keys():
            if dict_rd[can] > invest_dict[can]:
                needed_extra = dict_rd[can] - invest_dict[can]
                invest_dict[can] = dict_rd[can]
                
                # Check if previously checked candidates can help
                for rich_can in avl_dict.keys():
                    if avl_dict[rich_can] > 0 and len(rich_can) < allowed_length:
                        rich_can_gives = min(avl_dict[rich_can], needed_extra)
                        avl_dict[rich_can] -= rich_can_gives
                        total_investment_dict[rich_can] -= rich_can_gives
                        # Ballots with multiple choices
                        total_investment_dict[str(rich_can) + str(can)] = rich_can_gives
                        needed_extra -= rich_can_gives
                        
                if needed_extra > 0:
                    total_investment_dict[can] += needed_extra
                    
        t = strt[rd + 1][0]
        if total_investment_dict:
            amount_spent = sum(total_investment_dict[key] for key in total_investment_dict.keys())
        
        if amount_spent > budget+1:
            return {}, budget*2
                    
    return total_investment_dict, amount_spent



def reach_a_structure_check(candidates, main, sub, k, Q_new, ballot_counts, budget, allowed_length=None):
    """
    Given main and sub structure specifications, finds optimal investment to reach that structure.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q_new: Updated quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Tuple of (success boolean, updated ballot counts, amount spent)
    """
    amount_check = 1
    aggre_v_dict = get_new_dict(ballot_counts)
    check_aggre_v_dict = deepcopy(aggre_v_dict)
    check_ballot_counts = deepcopy(ballot_counts)
    strt, stdt = create_structure(main, sub)
    
    # Iterate until no more investment is needed
    while amount_check > 0:
        DecodedDicts, strt, log_campaign_list, status_list = process_campaign_STV(
            candidates, main, sub, k, Q_new, check_aggre_v_dict, budget
        )

        if all(status_list) == True:  # If campaigning is feasible
            total_investment_dict, amount_check = smart_campaign(
                candidates, log_campaign_list, strt, stdt, Q_new, DecodedDicts, budget, allowed_length= allowed_length
            )
            
            # Update ballot counts with new investment
            check_ballot_counts = campaign_addition_dict_simple(
                total_investment_dict, candidates, check_ballot_counts
            )
            
            if amount_check > budget or not total_investment_dict:
                return False, {}, 0
        else:
            return False, {}, 0
            
        check_aggre_v_dict = get_new_dict(check_ballot_counts)
        
    # Calculate total amount spent
    amount_spent = sum(check_aggre_v_dict.get(candidate, 0) for candidate in candidates) - sum(aggre_v_dict.get(candidate, 0) for candidate in candidates)
    
    return amount_spent < budget+1, check_ballot_counts, amount_spent

def reach_a_structure_check_memoization(candidates, main, sub, k, Q_new, ballot_counts, budget, failed_partial_orders=None, allowed_length=None):
    """
    Given main and sub structure specifications, finds optimal investment to reach that structure.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q_new: Updated quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        collection: Pre-calculated ballot collections
        failed_partial_orders: Dictionary of already failed partial orders
        allowed_length: Maximum length of ballot chains to consider (default: None)
        
    Returns:
        Tuple of (success boolean, updated ballot counts, amount spent)
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    check_aggre_v_dict = deepcopy(aggre_v_dict)
    check_ballot_counts = deepcopy(ballot_counts)
    strt, stdt = create_structure(main, sub)
    
    # First, check if this structure is already known to fail at some level
    if failed_partial_orders:
        for i in range(1, len(main)):
            partial_order = tuple(main[:i])
            if partial_order in failed_partial_orders:
                return False, {}, 0, i
    
    amount_check = 1
    while amount_check > 0:
        DecodedDicts, strt, log_campaign_list, status_list = process_campaign_STV_simple(
            candidates, main, sub, k, Q_new, check_ballot_counts, budget
        )

        # If process_campaign_STV fails, identify at which level it failed
        if status_list and False in status_list:
            failure_level = status_list.index(False) + 1
            return False, {}, 0, failure_level
            
        if all(status_list) == True:
            total_investment_dict, amount_check = smart_campaign(
                candidates, log_campaign_list, strt, stdt, Q_new, DecodedDicts, budget, allowed_length
            )
            
            if amount_check > budget or not total_investment_dict:
                # This is a budget failure, likely at the last level
                return False, {}, 0, len(main)
                
            check_ballot_counts = campaign_addition_dict_simple(
                total_investment_dict, candidates, check_ballot_counts
            )
        else:
            return False, {}, 0, len(main)  # Generic failure
            
        check_aggre_v_dict = get_new_dict(check_ballot_counts)
        
    amount_spent = sum(check_aggre_v_dict.get(candidate, 0) for candidate in candidates) - sum(aggre_v_dict.get(candidate, 0) for candidate in candidates)
    
    return amount_spent < budget+1, check_ballot_counts, amount_spent, None  # None indicates success


def flip_order_campaign(candidates, k, Q, ballot_counts, budget):
    """
    Finds minimum investment to flip the current election order.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Tuple of (updated counts dictionary, new quota, difference dictionary)
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    budget_list_flip = []
    campaigned_dict_list = []
    
    # Try each possible sub-structure
    for sub in sub_structures(candidates):
        status_final, check_aggre_v_dict, amount_spent = reach_a_structure_check(
            candidates, list(reversed(original_main)), sub, k, Q_new, aggre_v_dict, budget
        )
        
        if status_final == True:
            campaigned_dict_list.append(check_aggre_v_dict)
            budget_list_flip.append(amount_spent)
    
    if len(budget_list_flip) == 0:
        print('increase the budget')
        return {}, 0, {}
    else:        
        min_budget = min(budget_list_flip)
        print(budget_list_flip)
        min_index = budget_list_flip.index(min_budget)

        check_aggre_v_dict = campaigned_dict_list[min_index]
        new_ballot_counts = clean_aggre_dict_diff(check_aggre_v_dict)
        
        strt_new, stdt_new, collection = STV_optimal_result_simple(
            candidates, new_ballot_counts, 2, Q_new
        )
        new_main, new_sub = return_main_sub(strt_new)
        
        # Calculate vote difference
        C = {x: check_aggre_v_dict[x] - aggre_v_dict[x] for x in check_aggre_v_dict if x in aggre_v_dict}
        diff = {x: y for x, y in C.items() if y != 0}
        
        print('New votes to be added = ', clean_aggre_dict_diff(diff))
        print('original order = ', original_main, original_sub)
        print('new order = ', new_main, new_sub)
        print('budget used = ', min_budget)
        return check_aggre_v_dict, Q_new, clean_aggre_dict_diff(diff)


def reach_any_winners_campaign(candidates, k, Q, ballot_counts, budget, c_l =[], zeros = 0, allowed_length=None):
    """
    Finds minimum investment required for each possible combination of winners.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Dictionary mapping winner combinations to [budget, additions]
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    # Try each possible combination of k winners
    for comb in combinations(candidates, k):
        if set(comb) != set(og_winners) and not any(x in c_l for x in set(comb)):
            main_set = str_for_given_winners(comb, candidates)
            
            for current_main in main_set:
                budget_list_flip = []
                campaigned_dict_list = []
                
                for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k, zeros):
                    status_final, check_ballot_counts, amount_spent = reach_a_structure_check(
                        candidates, current_main, sub, k, Q_new, ballot_counts, budget, allowed_length= allowed_length
                    )

                    if status_final == True:
                        campaigned_dict_list.append(check_ballot_counts)
                        budget_list_flip.append(amount_spent)

                if len(budget_list_flip) > 0:
                    min_budget = min(budget_list_flip)
                    min_index = budget_list_flip.index(min_budget)

                    check_ballot_counts = campaigned_dict_list[min_index]
                    check_aggre_v_dict = get_new_dict(check_ballot_counts)
                    
                    strt_new, stdt_new, collection = STV_optimal_result_simple(
                        candidates, check_ballot_counts, k, Q_new
                    )
                    
                    new_main, new_sub = return_main_sub(strt_new)
                    
                    # Verify winners
                    if set(give_winners(new_main, k)) != set(comb):
                        strt_new, stdt_new = STV_optimal_result(
                        candidates, k, Q_new, check_aggre_v_dict)
                        new_main, new_sub = return_main_sub(strt_new)
                        if set(give_winners(new_main, k)) != set(comb):
                            print(min_budget)
                            print(set(give_winners(new_main, k)), set(comb))
                            print('error!')
    
                        
                    # Calculate vote difference
                    C = {
                        x: check_aggre_v_dict[x] - aggre_v_dict[x] 
                        for x in check_aggre_v_dict if x in aggre_v_dict
                    }
                    diff = {x: y for x, y in C.items() if y > 0}
                    
                    if strats_frame.get(''.join(comb), [budget, {}])[0] > min_budget:
                        strats_frame[''.join(comb)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x: y for x, y in strats_frame.items() if y[0] >= 0}


def reach_any_winners_campaign_memoization(candidates, k, Q, ballot_counts, budget, c_l =[], zeros = 0, allowed_length=None):
    """
    Finds minimum investment required for each possible combination of winners.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Dictionary mapping winner combinations to [budget, additions]
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    # Create a cache to store results of bottom portion checks
    failed_partial_orders = {}

    # Modified approach
    for comb in combinations(candidates, k):
        if set(comb) != set(og_winners) and not any(x in c_l for x in set(comb)):
            main_set = str_for_given_winners(comb, candidates)
            
            for current_main in main_set:
                # Check if any bottom portion of this order is already known to fail
                should_skip = False
                for i in range(1, len(current_main)):
                    partial_order = tuple(current_main[:i])  # Bottom portion
                    if partial_order in failed_partial_orders:
                        should_skip = True
                        break
                
                if should_skip:
                    continue
                budget_list_flip = []
                campaigned_dict_list = []
                
                for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k, zeros):
                    status_final, check_ballot_counts, amount_spent, failed_at = reach_a_structure_check_memoization(
                    candidates, current_main, sub, k, Q_new, ballot_counts, budget, allowed_length=allowed_length,)
                
                    if status_final == True:
                        campaigned_dict_list.append(check_ballot_counts)
                        budget_list_flip.append(amount_spent)
                    else:
                        # Store the point where this order failed for future reference
                        failed_partial_orders[tuple(current_main[:failed_at])] = True

                if len(budget_list_flip) > 0:
                    min_budget = min(budget_list_flip)
                    min_index = budget_list_flip.index(min_budget)

                    check_ballot_counts = campaigned_dict_list[min_index]
                    check_aggre_v_dict = get_new_dict(check_ballot_counts)
                    
                    strt_new, stdt_new, collection = STV_optimal_result_simple(
                        candidates, check_ballot_counts, k, Q_new
                    )
                    
                    new_main, new_sub = return_main_sub(strt_new)
                    
                    # Verify winners
                    if set(give_winners(new_main, k)) != set(comb):
                        strt_new, stdt_new = STV_optimal_result(
                        candidates, k, Q_new, check_aggre_v_dict)
                        new_main, new_sub = return_main_sub(strt_new)
                        if set(give_winners(new_main, k)) != set(comb):
                            print(min_budget)
                            print(set(give_winners(new_main, k)), set(comb))
                            print('error!')
    
                        
                    # Calculate vote difference
                    C = {
                        x: check_aggre_v_dict[x] - aggre_v_dict[x] 
                        for x in check_aggre_v_dict if x in aggre_v_dict
                    }
                    diff = {x: y for x, y in C.items() if y > 0}
                    
                    if strats_frame.get(''.join(comb), [budget, {}])[0] > min_budget:
                        strats_frame[''.join(comb)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x: y for x, y in strats_frame.items() if y[0] >= 0}



def reach_any_order_campaign(candidates, k, Q, ballot_counts, budget, zeros= 0):
    """
    Finds minimum investment needed to reach any possible structure.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Dictionary mapping orders to [budget, additions]
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og = STV_optimal_result(candidates, k, Q, aggre_v_dict)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    # Try each possible main structure
    main_set = main_structures(candidates)
    
    for current_main in main_set:
        budget_list_flip = []
        campaigned_dict_list = []
        
        for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k, zeros):
            status_final, check_ballot_counts, amount_spent = reach_a_structure_check(
                candidates, current_main, sub, k, Q_new, ballot_counts, budget
            )

            if status_final == True:
                campaigned_dict_list.append(check_ballot_counts)
                budget_list_flip.append(amount_spent)

        if len(budget_list_flip) > 0:
            min_budget = min(budget_list_flip)
            min_index = budget_list_flip.index(min_budget)

            check_ballot_counts = campaigned_dict_list[min_index]
            check_aggre_v_dict = get_new_dict(check_ballot_counts)
            
            strt_new, stdt_new = STV_optimal_result(candidates, k, Q_new, check_aggre_v_dict)
            new_main, new_sub = return_main_sub(strt_new)
            
            if current_main != new_main:
                print('error')
           
            # Calculate vote difference
            C = {
                x: check_aggre_v_dict[x] - aggre_v_dict[x] 
                for x in check_aggre_v_dict if x in aggre_v_dict
            }
            diff = {x: y for x, y in C.items() if y > 0}
            
            if strats_frame.get(''.join(new_main), [budget, {}])[0] > min_budget:
                strats_frame[''.join(new_main)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x: y for x, y in strats_frame.items() if y[0] >= 0}




def str_for_given_winners_with_position_constraints(winners, candidates, top_k, bottom_m):
    """
    Return a list of main structures for given winners with position constraints.
    
    Args:
        winners: List of winning candidates
        candidates: List of all candidates
        top_k: List of candidates that cannot appear in the bottom m positions
        bottom_m: List of candidates that cannot appear in the top k positions
    """
    # Check if winners contain any bottom_m candidates (invalid combination)
    if any(winner in bottom_m for winner in winners):
        return []
        
    losers = [item for item in candidates if item not in winners]
    potential_sets = []
    
    k = len(top_k)  # Number of top positions to constrain
    m = len(bottom_m)  # Number of bottom positions to constrain

    for perm_w in permutations(winners, len(winners)):
        # Check if any bottom_m candidates appear in the top k positions
        # (assuming len(perm_w) >= k, otherwise we just check what we have)
        top_positions = perm_w[:min(k, len(perm_w))]
        if any(candidate in bottom_m for candidate in top_positions):
            continue
            
        for perm_l in permutations(losers, len(losers)):
            # Check if any top_k candidates appear in the last m positions
            # (assuming len(perm_l) >= m, otherwise we just check what we have)
            bottom_positions = perm_l[-min(m, len(perm_l)):]
            if any(candidate in top_k for candidate in bottom_positions):
                continue
                
            potential_sets.append(list(perm_w) + list(perm_l))
    
    return potential_sets




def process_combination(comb, candidates, k, Q_new, ballot_counts, budget, og_winners, c_l, zeros, aggre_v_dict, top_k=None, bottom_m=None, allowed_length=None):
    """Process a single combination in parallel with optional position constraints"""
    # Skip if combination contains any bottom_m candidates (if constraints are provided)
    if bottom_m and any(c in bottom_m for c in comb):
        return None
        
    if set(comb) != set(og_winners) and not any(x in c_l for x in set(comb)):
        # Use different function based on whether constraints are provided
        if top_k and bottom_m:
            main_set = str_for_given_winners_with_position_constraints(comb, candidates, top_k, bottom_m)
        else:
            main_set = str_for_given_winners(comb, candidates)
        
        # If no valid orders, return early
        if not main_set:
            return None
            
        min_budget = budget + 1
        best_diff = {}
        
        # Local failed_partial_orders for this process only
        local_failed_orders = {}
        
        # Rest of your existing code remains the same
        for current_main in main_set:
            # Check local failed orders
            should_skip = False
            for i in range(1, len(current_main)):
                partial_order = tuple(current_main[:i])
                if partial_order in local_failed_orders:
                    should_skip = True
                    break
            
            if should_skip:
                continue
                
            budget_list_flip = []
            campaigned_dict_list = []
            
            for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k, zeros):
                status_final, check_ballot_counts, amount_spent, failed_at = reach_a_structure_check_memoization(
                    candidates, current_main, sub, k, Q_new, ballot_counts, budget,  
                    failed_partial_orders=local_failed_orders, allowed_length=allowed_length
                )
                
                if status_final == True:
                    campaigned_dict_list.append(check_ballot_counts)
                    budget_list_flip.append(amount_spent)
                else:
                    # Store locally - but only if partial order has more than k elements
                    # (single-element orders are too broad and incorrectly prune valid paths)
                    if failed_at is not None and failed_at > k:
                        local_failed_orders[tuple(current_main[:failed_at])] = True

            if len(budget_list_flip) > 0:
                current_min_budget = min(budget_list_flip)
                min_index = budget_list_flip.index(current_min_budget)

                check_ballot_counts = campaigned_dict_list[min_index]
                check_aggre_v_dict = get_new_dict(check_ballot_counts)
                
                # Verification code as in your original function
                strt_new, stdt_new, collection = STV_optimal_result_simple(
                    candidates, check_ballot_counts, k, Q_new
                )
                
                new_main, new_sub = return_main_sub(strt_new)
                
                # Verify winners
                if set(give_winners(new_main, k)) != set(comb):
                    strt_new, stdt_new = STV_optimal_result(
                        candidates, k, Q_new, check_aggre_v_dict)
                    new_main, new_sub = return_main_sub(strt_new)
                    if set(give_winners(new_main, k)) != set(comb):
                        print(current_min_budget)
                        print(set(give_winners(new_main, k)), set(comb))
                        print('error!')
                    
                # Calculate vote difference
                C = {
                    x: check_aggre_v_dict[x] - aggre_v_dict[x] 
                    for x in check_aggre_v_dict if x in aggre_v_dict
                }
                diff = {x: y for x, y in C.items() if y > 0}
                
                if current_min_budget < min_budget:
                    min_budget = current_min_budget
                    best_diff = clean_aggre_dict_diff(diff)
        
        if min_budget <= budget:
            return (''.join(comb), [min_budget, best_diff])
    
    return None

def reach_any_winners_campaign_parallel(candidates, k, Q, ballot_counts, budget, c_l=[], zeros=0, top_k=None, bottom_m=None, allowed_length=None):
    """
    Parallelized version with optional position constraints
    
    Args:
        candidates: List of all candidates
        k: Number of winners
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        c_l: List of excluded candidates
        zeros: Parameter for sub-structures
        top_k: List of candidates that cannot appear in bottom m positions (optional)
        bottom_m: List of candidates that cannot appear in top k positions (optional)
        allowed_length: Maximum length of ballot chains to consider (default: None)
    """
    if __name__ != '__main__':
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
    
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    # Generate combinations based on constraints if provided
    if bottom_m:
        # Filter combinations that contain bottom_m candidates
        all_combinations = [comb for comb in combinations(candidates, k) 
                           if not any(c in bottom_m for c in comb) 
                           and not any(x in c_l for x in set(comb))]
    else:
        # Use all combinations if no constraints
        all_combinations = list(combinations(candidates, k))
    
    # Setup partial function with constraints if provided
    process_func = partial(
        process_combination,
        candidates=candidates,
        k=k,
        Q_new=Q_new, 
        ballot_counts=deepcopy(ballot_counts),
        budget=budget,
        og_winners=og_winners,
        c_l=c_l,
        zeros=zeros,
        aggre_v_dict=deepcopy(aggre_v_dict), 
        top_k=top_k,
        bottom_m=bottom_m,
        allowed_length=allowed_length
    )
    
    # Process in parallel with robust cleanup for Streamlit environment.
    # Uses global _active_pool so atexit handler can kill workers if the
    # main thread is interrupted (Streamlit rerun, page navigation, etc.)
    global _active_pool

    # Kill any pool left over from a previous interrupted run
    _cleanup_pool()

    try:
        _active_pool = Pool(processes=None)
        all_results = _active_pool.map(process_func, all_combinations)

        # Process results
        for result in all_results:
            if result is not None:
                comb_key, value = result
                strats_frame[comb_key] = value

        return {x: y for x, y in strats_frame.items() if y[0] >= 0}
    finally:
        # Normal cleanup path — also covers exceptions/StopException
        _cleanup_pool()