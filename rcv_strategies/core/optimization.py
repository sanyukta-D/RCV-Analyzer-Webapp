from copy import deepcopy
from rcv_strategies.utils.helpers import common_member, make_perms_transfer

###########################################
# ROUND UPDATE FUNCTIONS
###########################################

def roundupdate(structure, checked_candidates, remaining_candidates, olddict):
    """
    After a round result (W/L) is reflected in checked_candidates,
    this function updates the remaining vote banks by transferring the unused votes.
    
    Args:
        structure: Dictionary mapping candidates to their win/lose status
        checked_candidates: List of candidates that have been processed
        remaining_candidates: List of candidates still in the election
        olddict: Previous vote bank dictionary
        
    Returns:
        Updated vote bank dictionary
    """
    if len(checked_candidates) == 0:
        return olddict
        
    currentdict = deepcopy(olddict)
    last_candidate = checked_candidates[-1]  # The most recently processed candidate
    
    perms_checked_candidates = make_perms_transfer(checked_candidates)
    
    stvotelist = currentdict[last_candidate]  # Last candidate's total vote bank
    for I_c in stvotelist:  # Each set of votes that has a separate path to the last candidate
        I = deepcopy(I_c)  # Copy to prevent modifying the original list
        q = []
        
        # Remove lists with [1,..] from vote banks (these are fractions)
        while I and I[-1][0] == 1:  
            q1 = I.pop()
            q.append(q1)  # Store removed fractions for later
            
        for j in remaining_candidates:  # Update vote banks of remaining candidates
            for l in perms_checked_candidates:  # For each permutation of checked candidates
                if not common_member(I, l):  # Ensure permutation candidates and sets are distinct
                    if structure[last_candidate] == 0:  # Last round was an elimination
                        Inew = I + l + [j] + q 
                    else:  # Last round was a win
                        Inew = I + l + [j] + q + [[1, 'Q', stvotelist]]  # Append list for surplus fraction
                    currentdict[j].append(Inew)
                    
    return currentdict

###########################################
# DECODING FUNCTIONS
###########################################

def decode_list(pathcopy, Q, aggre_v_dict):
    """
    Convert a coded path into a number representing the votes this path contributes.
    
    Args:
        pathcopy: Path to decode
        Q: Quota value
        aggre_v_dict: Aggregated voter data dictionary
        
    Returns:
        Number of votes this path contributes
    """
    path = deepcopy(pathcopy)
    
    # If the last element isn't a fraction indicator
    if not path or path[-1][0] != 1:
        value = aggre_v_dict.get(''.join(path), 0)  
        return value
        
    Qlist = []
    path2 = deepcopy(path)
    
    # Extract all fraction indicators
    while path2 and path2[-1][0] == 1:
        q_item = path2.pop()
        Qlist.append(q_item)
        
    # Get value of the base path without fractions
    tot = decode_list(path2, Q, aggre_v_dict)
    
    # Apply each fraction modifier
    for q_item in Qlist:
        value = 0
        for part in q_item[-1]:
            value += decode_list(part, Q, aggre_v_dict)
        tot = tot * max(0, (1 - Q/(value + 0.001)))
        
    return tot

def decode_dict(currentdict, candidates, Q, aggre_v_dict):
    """
    Convert a dictionary with complete variable descriptions to numerical data.
    
    Args:
        currentdict: Dictionary with coded path descriptions
        candidates: List of all candidates
        Q: Quota value
        aggre_v_dict: Aggregated voter data dictionary
        
    Returns:
        Dictionary mapping candidates to their vote counts
    """
    newcurrentdict = {}
    
    for candidate in candidates:
        currentdict2 = deepcopy(currentdict)
        v_list = currentdict2[candidate]
        # Sum the decoded values for all paths for this candidate
        newcurrentdict[candidate] = round(sum(decode_list(path, Q, aggre_v_dict) for path in v_list), 3)

    return newcurrentdict

###########################################
# OPTIMAL STV RESULT FUNCTIONS
###########################################

def give_next_candidate(decoded_dict, remaining_candidates, Q, rt, dt):
    """
    Determine the next candidate to be eliminated or declared a winner.
    
    Args:
        decoded_dict: Dictionary with decoded numerical values
        remaining_candidates: List of candidates still in the election
        Q: Quota value
        rt: Results tracking list
        dt: Results tracking dictionary
        
    Returns:
        Tuple of (selected candidate, updated rt, updated dt)
    """
    # Create a dictionary with only remaining candidates
    short_decoded_dict = {can: decoded_dict[can] for can in remaining_candidates}
        
    # Find candidates with max and min votes
    t1 = max(short_decoded_dict, key=short_decoded_dict.get)
    t2 = min(short_decoded_dict, key=short_decoded_dict.get)
    
    # Determine if next action is a win or elimination
    if decoded_dict[t1] >= Q:
        t = t1
        rt.append([t, 1])  # Mark as winner (1)
        dt[t] = 1
    else:
        t = t2
        rt.append([t, 0])  # Mark as eliminated (0)
        dt[t] = 0
        
    return t, rt, dt

def STV_optimal_result(candidates, k, Q, aggre_v_dict):
    """
    Produce the resulting optimal structure for given voter data.
    
    Args:
        candidates: List of all candidates
        k: Number of winners to select
        Q: Quota value
        aggre_v_dict: Aggregated voter data dictionary
        
    Returns:
        Tuple of (results list, results dictionary)
    """
    # Initialize each candidate with their own vote bank
    currentdict = {can: [[can]] for can in candidates}

    remaining_candidates = deepcopy(candidates)
    checked_candidates = []
    dt = {}  # Dictionary to track results
    rt = []  # List to track results
    
    # Process each round
    for i in range(len(candidates)):
        # Update vote banks according to current round
        currentdict = roundupdate(dt, checked_candidates, remaining_candidates, currentdict)
        
        # Convert variable descriptions to numerical data
        decoded_dict = decode_dict(currentdict, candidates, Q, aggre_v_dict)
        
        # Get next candidate to process
        t, rt, dt = give_next_candidate(decoded_dict, remaining_candidates, Q, rt, dt)
        
        # Update candidate lists
        remaining_candidates.remove(t) 
        checked_candidates.append(t)
        
    return rt, dt