from itertools import permutations, product, combinations
from copy import deepcopy
from collections import Counter
import math

###########################################
# HELPER FUNCTIONS
###########################################

def common_member(a, b): 
    """Check if two lists have any common elements."""
    a_set = set(a)
    b_set = set(b)
    return len(a_set.intersection(b_set)) > 0

###########################################
# PERMUTATION AND STRUCTURE GENERATION
###########################################

def main_structures(candidates):
    """Generate all possible orderings of candidates - n! in total."""
    return [list(perm) for perm in permutations(candidates, len(candidates))]

def str_for_given_winners(winners, candidates):
    """Return a list of main structures for given list of winners."""
    losers = [item for item in candidates if item not in winners]
    potential_sets = []

    for perm_w in permutations(winners, len(winners)):
        for perm_l in permutations(losers, len(losers)):
            potential_sets.append(list(perm_w) + list(perm_l))
    return potential_sets

def str_for_given_losers(losers, candidates):
    """Return a list of main structures for given list of losers."""
    winners = [item for item in candidates if item not in losers]
    potential_sets = []

    for perm_w in permutations(winners, len(winners)):
        for perm_l in permutations(losers, len(losers)):
            potential_sets.append(list(perm_w) + list(perm_l))
    return potential_sets
    
def str_for_given_winners_losers(winners, candidates, losers):
    """Return a list of main structures for given lists of winners and losers."""
    middle = [item for item in candidates if item not in winners and item not in losers]
    potential_sets = []

    for perm_w in permutations(winners, len(winners)):
        for perm_m in permutations(middle, len(middle)):
            for perm_l in permutations(losers, len(losers)):
                potential_sets.append(list(perm_w) + list(perm_m) + list(perm_l))
    return potential_sets

def sub_structures(candidates):
    """Generate all possible W/L round result types - 2^(n-1) in total."""
    l = len(candidates) - 1
    results = list(product([0, 1], repeat=l))
    return [list(ele) + [1] for ele in results]

def sub_structures_specific(candidates, z):
    """Generate combinations with z zeros at beginning and (l-z) zeros in middle."""
    l = len(candidates) - 1
    
    prefix = [0] * z
    middle = list(product([0, 1], repeat=l - z))
    suffix = [1]
    
    return [prefix + list(ele) + suffix for ele in middle]

def sub_structures_at_most_k_ones_fixed_last(candidates, k, z=0):
    """
    Return all binary sequences of length len(candidates)
    whose last bit is forced to 1 and which have at most k ones.
    If z is provided, only include sequences where the first z bits are zeros.

    This efficient version generates only the valid free portions of the sequence.
    """
    n = len(candidates)
    # There are n-1 positions free to choose (last bit is forced to 1)
    # and the first z of these (or overall) are forced to 0.
    if z > n - 1:
        return []  # Not enough positions for forced zeros if z exceeds n-1.
    if k ==1:
        return [[0] * (len(candidates)-1)+ [1]]

    free_length = n - 1 - z   # positions available for free selection after z forced zeros
    # Since the last bit is 1, free positions can contribute at most (k-1) ones.
    max_free_ones = max(0, k - 1)

    results = []
    # Iterate over possible numbers of ones in the free region from 0 to max_free_ones
    for ones_count in range(max_free_ones + 1):
        # If ones_count exceeds the available free positions, skip.
        if ones_count > free_length:
            continue
        # Choose positions within the free region to place ones.
        for ones_positions in combinations(range(free_length), ones_count):
            free_bits = [0] * free_length
            for pos in ones_positions:
                free_bits[pos] = 1
            # Build the full sequence: forced zeros at the beginning, then free_bits, then the final 1.
            sequence = [0] * z + free_bits + [1]
            results.append(sequence)
    return results

###########################################
# STRUCTURE OPERATIONS
###########################################

def create_structure(main, sub):
    """
    Create a structure showing who wins/loses in each round.
    Returns both a list and a dictionary representation.
    """
    maincopy = deepcopy(main)
    results = []
    redict = {}  # Mapping candidates to their results in round order
    
    for i in sub:
        if i == 0:
            res = maincopy.pop(-1)
            results.append([res, i])
        else:
            res = maincopy.pop(0)
            results.append([res, i])
        redict[res] = i
           
    return results, redict

def return_main_sub(results_list):
    """
    Convert a list showing who wins/loses in each round back to main and sub structure.
    """
    strt = deepcopy(results_list)
    sub = []
    win = []
    lose = []
    
    for i in strt:
        if i[1] > 0:
            win = win + [i[0]]
        else:
            lose = [i[0]] + lose
        sub.append(i[1])
        
    main = win + lose
    return main, sub

def give_winners(main_st, k):
    """Return the list of winners, given main structure."""
    return main_st[0:k]

###########################################
# PERMUTATION AND VOTE PATTERN GENERATION
###########################################

def make_perms_transfer(candidates):
    """
    Generate all permutations of checked candidates to add paths between transfers.
    """
    perms_candidates = []
    for l in range(len(candidates)):
        perms = [list(perm) for perm in permutations(candidates, l)]
        perms_candidates.extend(perms)
    return perms_candidates

def make_perms(candidates, min_ballot_length=0):
    """
    Generate all possible vote patterns of specified length for given candidates.
    """
    perms_candidates = []
    for l in range(min_ballot_length, len(candidates)):
        perms = [list(perm) for perm in permutations(candidates, l+1)]
        perms_candidates.extend(perms)
    return perms_candidates

def dict_perm_k(candidates, k):
    """Initialize a dictionary of all possible winner types."""
    dict_new = {}
    for perm in permutations(candidates, k):
        dict_new[''.join(list(perm))] = 0
    return dict_new

###########################################
# DICTIONARY OPERATIONS
###########################################

def non_agg_dict(ballot_types, num_voters):
    """
    Convert data in (type, num) format to a non-aggregated dictionary.
    """
    string_ballots = [''.join(ballot) for ballot in ballot_types]
    return dict(zip(string_ballots, num_voters))

def agg_dict(candidates, non_aggre_voterdata):
    """
    Create a complete aggregated dictionary from non-aggregated data.
    V_{AB} includes total votes with first and second choices as A and B.
    """
    aggre_v_dict = Counter()
    permset = make_perms(candidates)

    # Calculate the aggregation using Counter
    for ballot in permset:
        for i in range(len(ballot)):
            v = ''.join(ballot[0:i+1])
            aggre_v_dict[v] += non_aggre_voterdata.get(''.join(ballot), 0)

    # Filter out keys with zero counts
    return {x: y for x, y in aggre_v_dict.items() if y > 0}

def get_new_dict(my_dict):
    """Convert any investment or ballot counts into aggregated investment."""
    new_dict = {}
    
    # Iterate through the keys and values in my_dict
    for key, value in my_dict.items():
        # Generate all possible substrings of the key
        substrings = [key[:i+1] for i in range(len(key))]
        
        # Update the new_dict with the aggregated values for each substring
        for substring in substrings:
            new_dict[substring] = new_dict.get(substring, 0) + value
    
    # Filter out keys with zero counts
    return {x: y for x, y in new_dict.items() if y > 0}

def campaign_addition_dict(my_total_investment_dict, candidates, aggre_v_dict):
    """
    Given investment and original aggregated dict, return the final aggregated dict.
    """
    my_total_investment_dict_d = deepcopy(my_total_investment_dict)
    aggre_v_dict_d = deepcopy(aggre_v_dict)
  
    agg_total_investment_dict = get_new_dict(my_total_investment_dict_d)

    for key in agg_total_investment_dict.keys():
        aggre_v_dict_d[key] = aggre_v_dict_d.get(key, 0) + math.ceil(agg_total_investment_dict.get(key, 0))
    
    return aggre_v_dict_d

def campaign_addition_dict_simple(my_total_investment_count, candidates, ballot_counts):
    """
    Simplified version of campaign_addition_dict without aggregation step.
    """
    my_total_investment_count_d = deepcopy(my_total_investment_count)
    ballot_counts_d = deepcopy(ballot_counts)

    for key in my_total_investment_count_d.keys():
        ballot_counts_d[key] = ballot_counts_d.get(key, 0) + math.ceil(my_total_investment_count_d.get(key, 0))
    
    return ballot_counts_d

def clean_aggre_dict_diff(anydict):
    """
    Reverse of aggregation operation. Convert aggregated dict to non-aggregated dict.
    """
    cleandict = deepcopy(anydict)
    for key1 in cleandict.keys():
        for key2 in cleandict.keys():
            if key2[0:len(key1)] == key1 and key1 != key2:
                cleandict[key1] = cleandict[key1] - cleandict[key2]
    
    return {x: y for x, y in cleandict.items() if y > 0}