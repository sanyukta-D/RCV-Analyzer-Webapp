# Paper figures produced by this script:
#   - Fig 5 right: nyc_models_by_ratio.pdf    (Fig \ref{fig:exhaustion_nyc})
#   - Fig 6 right: alaska_models_by_ratio.pdf  (Fig \ref{fig:exhaustion_alaska})
#
# Output data (used by heatmap.py and webapp):
#   - model_comparison_results/all_elections_analysis.csv
#
# Input data (paper's source of truth for single-winner exhaustion):
#   - summary_table_nyc_final.xlsx   (also at results/tables/)
#   - summary_table_alska_lite.xlsx  (also at results/tables/)
#
# Raw ballot data for bootstrap models:
#   - case_studies/nyc/data/     (NYC 2021 DEM primaries)
#   - case_studies/alaska/data/  (Alaska 2024 elections)

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # Force Agg backend for better plot saving
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import os
from string import ascii_uppercase
import math
import argparse

# Core RCV imports
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.utils import helpers as utils

# Optional import for NYC data processing (only needed for standalone analysis)
try:
    from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
    NYC_CONVERT_AVAILABLE = True
except ImportError:
    NYC_CONVERT_AVAILABLE = False
    process_single_file = None
    create_candidate_mapping = None

# Constants and configuration settings
MAX_RANKINGS_NYC = 5  # NYC allows max 6 rankings per ballot

def extract_strategy_dict(strategy_str):
    """Extract strategy dictionary from string"""
    try:
        strategy_dict = ast.literal_eval(strategy_str)
        if not isinstance(strategy_dict, dict):
            return {}
        
        # Extract main percentage for each candidate
        result = {}
        for candidate, data_list in strategy_dict.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                result[candidate] = data_list[0]
        return result
    except:
        return {}

def extract_exhaust_dict(exhaust_str):
    """Extract exhaust dictionary from string"""
    try:
        exhaust_dict = ast.literal_eval(exhaust_str)
        if not isinstance(exhaust_dict, dict):
            return {}
        return exhaust_dict
    except:
        return {}

def beta_parameters(gap_to_win_pct):
    """Calculate Beta parameters based on gap size"""
    # Base parameters for a perfectly tied race (centered at 0.5)
    base_param = 50.0
    
    # Maximum shift based on gap (capped at 40)
    max_shift = min(gap_to_win_pct * 0.5, 40)
    
    # Calculate parameters
    a = max(base_param - max_shift, 10.0)  # Parameter for B (challenger), minimum 10
    b = max(base_param + max_shift, 10.0)  # Parameter for A (leader), minimum 10
    
    return a, b

def beta_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using Beta distribution"""
    # Ensure required_preference_pct is within bounds
    required_preference_pct = max(0, min(required_preference_pct, 100))
    
    # Convert from percentage to proportion
    required_proportion = required_preference_pct / 100
    
    a, b = beta_parameters(gap_to_win_pct)
    return 1 - stats.beta.cdf(required_proportion, a, b)

def direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots, gap_to_win_pct):
    """Calculate probability using a Beta distribution fitted to observed data.
    
    This model uses conditional evidence from observed ballot patterns by first preference,
    similar to the category-based bootstrap, but applies a Beta distribution to model uncertainty.
    It follows the logic:
    
    1. Categorizes exhausted ballots by first preference
    2. For each category, analyzes non-exhausted ballots to determine A>B vs B>A preferences
    3. Calculates expected completions for each category and aggregates them
    4. Uses Beta distribution with parameters derived from B>A vs A>B percentages
       to calculate win probability directly
    """
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    
    # Calculate the gap in votes directly from the gap percentage
    gap_votes = int(gap_to_win_pct * total_ballots / 100)
    
    # Votes needed for B to win: gap + 1
    votes_needed_for_b = gap_votes + 1
    
    # Categorize exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Get total exhausted ballots
    total_exhausted = sum(exhausted_ballots.values())
    
    if total_exhausted == 0:
        # No exhausted ballots to analyze
        return 0.5  # Return neutral probability when no data
    
    # Identify non-exhausted ballots that express preferences for A and/or B
    # group them by first preference to match the exhausted ballot categories
    complete_by_first_pref = {}
    
    # Only consider ballots that rank BOTH A and B for determining preference percentages
    # This aligns better with what the bootstrap methods do
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
            
        # Include only ballots that rank both A and B to get true preference distribution
        has_a = 'A' in ballot
        has_b = 'B' in ballot
        
        if has_a and has_b:
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {'b_over_a': 0, 'a_over_b': 0, 'total': 0}
            
            complete_by_first_pref[first_pref]['total'] += count
            
            # Record A>B vs B>A preference based on actual ordering
            if ballot.index('B') < ballot.index('A'):
                complete_by_first_pref[first_pref]['b_over_a'] += count
            else:
                complete_by_first_pref[first_pref]['a_over_b'] += count
    
    # Track expected completions for reporting
    total_expected_b_over_a = 0
    total_expected_a_over_b = 0
    
    # Process each category of exhausted ballots to get expected completions
    for first_pref, count in exh_by_first_pref.items():
        # Check if we have data on preferences for this first preference
        if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
            # Get preference distribution for this category
            category_data = complete_by_first_pref[first_pref]
            prob_b_over_a = category_data['b_over_a'] / category_data['total']
            
            # Calculate expected completions for this category
            expected_b_over_a = count * prob_b_over_a
            expected_a_over_b = count * (1 - prob_b_over_a)
            
            # Add to totals
            total_expected_b_over_a += expected_b_over_a
            total_expected_a_over_b += expected_a_over_b
        else:
            # If no data for this category, use the overall distribution
            total_complete = sum(data['total'] for data in complete_by_first_pref.values())
            total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
            
            if total_complete > 0:
                overall_prob_b_over_a = total_b_over_a / total_complete
                
                # Calculate expected completions for this category
                expected_b_over_a = count * overall_prob_b_over_a
                expected_a_over_b = count * (1 - overall_prob_b_over_a)
                
                # Add to totals
                total_expected_b_over_a += expected_b_over_a
                total_expected_a_over_b += expected_a_over_b
    
    # Calculate probability using Beta distribution with A>B and B>A percentages
    if total_expected_b_over_a + total_expected_a_over_b > 0:
        # Get total completions
        total_completions = total_expected_b_over_a + total_expected_a_over_b
        
        # Calculate B>A preference percentage (0-100 scale)
        b_over_a_pct = 100 * total_expected_b_over_a / total_completions
        a_over_b_pct = 100 * total_expected_a_over_b / total_completions
        
        # Calculate expected net votes for B (for reporting only)
        expected_net_for_b = total_expected_b_over_a - total_expected_a_over_b
        
        # Check if expected completions would be enough for B to win (for reporting only)
        expected_b_wins = expected_net_for_b >= votes_needed_for_b
        
        # Use the percentages directly as Beta parameters
        # This is consistent with beta_parameters which also returns values in percentage scale
        alpha = b_over_a_pct
        beta = a_over_b_pct
        
        # Required preference as proportion (0-1 scale for Beta CDF)
        required_proportion = required_preference_pct / 100
        
        # Calculate probability using Beta distribution
        probability = 1 - stats.beta.cdf(required_proportion, alpha, beta)
    else:
        # No matching preference data found
        probability = 0.5  # Neutral when no evidence
        b_over_a_pct = 50
        a_over_b_pct = 50
        expected_net_for_b = 0
        expected_b_wins = False
        alpha = 50  # Neutral parameters (in percentage scale)
        beta = 50
    
    # Print diagnostics
    print(f"[Direct Posterior Beta] " +
          f"Expected B>A: {total_expected_b_over_a:.2f} ({b_over_a_pct:.2f}%), " +
          f"A>B: {total_expected_a_over_b:.2f} ({a_over_b_pct:.2f}%), " +
          f"required: {required_preference_pct:.2f}%, " +
          f"beta params: alpha={alpha:.2f}, beta={beta:.2f}, " +
          f"expected net votes: {expected_net_for_b:.2f}, needed: {votes_needed_for_b}, " +
          f"expected win: {expected_b_wins}, probability: {probability:.4f}")
    
    return probability

def extract_first_preferences(ballot_counts, candidates):
    """Extract first preferences from ballot counts"""
    first_prefs = {}
    for ballot, count in ballot_counts.items():
        if ballot:  # Non-empty ballot
            first_pref = ballot[0]
            if first_pref not in first_prefs:
                first_prefs[first_pref] = 0
            first_prefs[first_pref] += count
    return first_prefs

def calculate_preference_distributions(ballot_counts, candidates, first_pref):
    """Calculate B>A vs A>B distributions for ballots with given first preference"""
    b_over_a = 0
    a_over_b = 0
    total = 0
    
    for ballot, count in ballot_counts.items():
        if not ballot or ballot[0] != first_pref:
            continue
            
        # Check combinations of A and B in ballot
        has_a = 'A' in ballot
        has_b = 'B' in ballot
        
        if has_a and has_b:
            # Both A and B are ranked - use exact ordering
            total += count
            if ballot.index('B') < ballot.index('A'):
                b_over_a += count
            else:
                a_over_b += count
        elif has_a and not has_b:
            # Only A is ranked - count as A>B
            total += count
            a_over_b += count
        elif has_b and not has_a:
            # Only B is ranked - count as B>A
            total += count
            b_over_a += count
                
    return b_over_a, a_over_b, total 

def category_based_bootstrap(ballot_counts, candidates, exhausted_ballots, gap_to_win_pct, exhaust_pct, required_preference_pct=None, n_bootstrap=1000):
    """
    Category-based bootstrap simulation for RCV elections:
    1. Finds votes needed for candidate B to win based on the gap percentage
    2. Categorizes exhausted ballots (that don't rank A or B) by first choice
    3. For each category, analyzes non-exhausted ballots to determine A>B vs B>A preferences
    4. Uses bootstrap sampling to complete exhausted ballots based on observed preferences
    5. Checks if B would win with the completed preferences
    6. Repeats multiple times to calculate win probability
    """
    print(f"Running category-based bootstrap with {n_bootstrap} iterations...")
    
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    print(f"Total ballots in election: {total_ballots}")
    
    # Calculate the gap in votes directly from the gap percentage
    gap_votes = int(gap_to_win_pct * total_ballots / 100)
    print(f"Gap in votes (A leads B by): {gap_votes}")
    
    # Votes needed for B to win: gap + 1
    votes_needed_for_b = gap_votes + 1
    print(f"B needs {votes_needed_for_b} additional votes to win")
    
    # Get total exhausted ballots
    total_exhausted = sum(exhausted_ballots.values())
    print(f"Total exhausted ballots: {total_exhausted}")
    
    # Categorize exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Print first preference distribution in exhausted ballots
    print("\nExhausted ballots by first preference:")
    for pref, count in sorted(exh_by_first_pref.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pref}: {count} ({100*count/total_exhausted:.2f}%)")
    
    # Identify non-exhausted ballots that have both A and B
    complete_by_first_pref = {}
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
        if 'A' in ballot and 'B' in ballot:
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {'ballots': {}, 'b_over_a': 0, 'a_over_b': 0, 'total': 0}
            
            complete_by_first_pref[first_pref]['ballots'][ballot] = count
            complete_by_first_pref[first_pref]['total'] += count
            
            # Record A>B vs B>A preference
            if ballot.index('B') < ballot.index('A'):
                complete_by_first_pref[first_pref]['b_over_a'] += count
            else:
                complete_by_first_pref[first_pref]['a_over_b'] += count
    
    # Calculate A>B and B>A percentages for each first preference category
    print("\nA>B vs B>A preferences by first preference category:")
    for pref, data in sorted(complete_by_first_pref.items(), key=lambda x: x[1]['total'], reverse=True):
        if data['total'] > 0:
            b_over_a_pct = 100 * data['b_over_a'] / data['total']
            print(f"  {pref}: Total={data['total']}, B>A={data['b_over_a']} ({b_over_a_pct:.2f}%), " +
                  f"A>B={data['a_over_b']} ({100-b_over_a_pct:.2f}%)")
    
    # Run bootstrap iterations
    bootstrap_results = []
    b_win_counts = 0
    
    # Results for each iteration
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
        
        # Track net votes gained by B (B>A minus A>B)
        net_votes_for_b = 0
        
        # Track completions
        b_over_a_completions = 0
        a_over_b_completions = 0
        
        # Process each category of exhausted ballots
        for first_pref, count in exh_by_first_pref.items():
            # Check if we have data on preferences for this first preference
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                # Get preference distribution for this category
                category_data = complete_by_first_pref[first_pref]
                prob_b_over_a = category_data['b_over_a'] / category_data['total']
                
                # Sample B>A vs A>B preferences for this category
                b_completions = np.random.binomial(count, prob_b_over_a)
                a_completions = count - b_completions
                
                # Add to totals
                b_over_a_completions += b_completions
                a_over_b_completions += a_completions
                
                # Update net votes for B
                net_votes_for_b += (b_completions - a_completions)
            else:
                # If no data for this category, use the overall distribution
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
                
                if total_complete > 0:
                    overall_prob_b_over_a = total_b_over_a / total_complete
                    
                    # Sample using overall distribution
                    b_completions = np.random.binomial(count, overall_prob_b_over_a)
                    a_completions = count - b_completions
                    
                    # Add to totals
                    b_over_a_completions += b_completions
                    a_over_b_completions += a_completions
                    
                    # Update net votes for B
                    net_votes_for_b += (b_completions - a_completions)
        
        # Check if B wins: net_votes_for_b must exceed the gap
        b_wins = net_votes_for_b >= votes_needed_for_b
        
        # Record result
        bootstrap_results.append(b_wins)
        if b_wins:
            b_win_counts += 1
        
        # Record detailed stats for first few iterations
        if i < 5:
            total_completions = b_over_a_completions + a_over_b_completions
            b_over_a_pct = 100 * b_over_a_completions / total_completions if total_completions > 0 else 0
            
            print(f"\nIteration {i} details:")
            print(f"  Completed {total_completions} ballots")
            print(f"  B>A: {b_over_a_completions}, A>B: {a_over_b_completions}, B>A%: {b_over_a_pct:.2f}%")
            print(f"  Net votes gained by B: {net_votes_for_b}")
            print(f"  Votes needed for B to win: {votes_needed_for_b}")
            print(f"  B wins: {b_wins}")
    
    # Calculate probability and confidence interval
    b_win_probability = b_win_counts / n_bootstrap
    
    # 95% confidence interval
    alpha = 0.05
    se = np.sqrt((b_win_probability * (1 - b_win_probability)) / n_bootstrap)
    ci_lower = max(0, b_win_probability - 1.96 * se)
    ci_upper = min(1, b_win_probability + 1.96 * se)
    
    # Report results
    print("\nCategory-Based Bootstrap Results:")
    print(f"Probability B wins: {b_win_probability:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"B won in {b_win_counts} out of {n_bootstrap} iterations")
    if required_preference_pct is not None:
        print(f"Required preference: {required_preference_pct:.2f}%")
    
    return b_win_probability, (ci_lower, ci_upper), bootstrap_results

def limited_ranking_bootstrap(ballot_counts, candidates, exhausted_ballots, gap_to_win_pct, exhaust_pct, required_preference_pct=None, n_bootstrap=1000, max_rankings=6):
    """
    Bootstrap simulation that respects NYC's 6-candidate ranking limit.
    
    This function:
    1. Only considers partially filled exhausted ballots (fewer than 6 rankings)
    2. For each first preference category, uses preferences observed in unexhausted ballots
       with that same first preference to inform sampling
    3. Only completes ballots that have room for more rankings (< max_rankings)
    """
    print(f"Running limited ranking bootstrap with {n_bootstrap} iterations (max rankings: {max_rankings})...")
    
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    print(f"Total ballots in election: {total_ballots}")
    
    # Calculate the gap in votes directly from the gap percentage
    gap_votes = int(gap_to_win_pct * total_ballots / 100)
    print(f"Gap in votes (A leads B by): {gap_votes}")
    
    # Votes needed for B to win: gap + 1
    votes_needed_for_b = gap_votes + 1
    print(f"B needs {votes_needed_for_b} additional votes to win")
    
    # Filter exhausted ballots to only include those with fewer than max_rankings
    partial_exhausted_ballots = {ballot: count for ballot, count in exhausted_ballots.items() 
                               if ballot and len(ballot) < max_rankings}
    
    # Count all exhausted ballots and partial ones
    total_exhausted = sum(exhausted_ballots.values())
    total_partial_exhausted = sum(partial_exhausted_ballots.values())
    
    print(f"Total exhausted ballots: {total_exhausted}")
    print(f"Total partially filled exhausted ballots (< {max_rankings} rankings): {total_partial_exhausted} ({100*total_partial_exhausted/total_exhausted:.2f}%)")
    
    # Categorize partially filled exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in partial_exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Print first preference distribution in partially filled exhausted ballots
    print(f"\nPartially filled exhausted ballots (< {max_rankings} rankings) by first preference:")
    for pref, count in sorted(exh_by_first_pref.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pref}: {count} ({100*count/total_partial_exhausted:.2f}%)")
    
    # Identify non-exhausted ballots that express preferences for A and B
    # group them by first preference to match the exhausted ballot categories
    complete_by_first_pref = {}
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
        if 'A' in ballot and 'B' in ballot:  # Only ballots that explicitly rank both A and B
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {'ballots': {}, 'b_over_a': 0, 'a_over_b': 0, 'total': 0}
            
            complete_by_first_pref[first_pref]['ballots'][ballot] = count
            complete_by_first_pref[first_pref]['total'] += count
            
            # Record A>B vs B>A preference
            if ballot.index('B') < ballot.index('A'):
                complete_by_first_pref[first_pref]['b_over_a'] += count
            else:
                complete_by_first_pref[first_pref]['a_over_b'] += count
    
    # Calculate A>B and B>A percentages for each first preference category
    print("\nA>B vs B>A preferences by first preference category:")
    for pref, data in sorted(complete_by_first_pref.items(), key=lambda x: x[1]['total'], reverse=True):
        if data['total'] > 0:
            b_over_a_pct = 100 * data['b_over_a'] / data['total']
            print(f"  {pref}: Total={data['total']}, B>A={data['b_over_a']} ({b_over_a_pct:.2f}%), " +
                  f"A>B={data['a_over_b']} ({100-b_over_a_pct:.2f}%)")
    
    # Run bootstrap iterations
    bootstrap_results = []
    b_win_counts = 0
    
    # Results for each iteration
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
        
        # Track net votes gained by B (B>A minus A>B)
        net_votes_for_b = 0
        
        # Track completions
        b_over_a_completions = 0
        a_over_b_completions = 0
        
        # Process each category of partially filled exhausted ballots
        for first_pref, count in exh_by_first_pref.items():
            # Check if we have data on preferences for this first preference
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                # Get preference distribution for this category
                category_data = complete_by_first_pref[first_pref]
                prob_b_over_a = category_data['b_over_a'] / category_data['total']
                
                # Sample B>A vs A>B preferences for this category
                # Using preferences from only the same first preference category
                b_completions = np.random.binomial(count, prob_b_over_a)
                a_completions = count - b_completions
                
                # Add to totals
                b_over_a_completions += b_completions
                a_over_b_completions += a_completions
                
                # Update net votes for B
                net_votes_for_b += (b_completions - a_completions)
            else:
                # If no data for this category, use the overall distribution from all categories
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
                
                if total_complete > 0:
                    overall_prob_b_over_a = total_b_over_a / total_complete
                    
                    # Sample using overall distribution
                    b_completions = np.random.binomial(count, overall_prob_b_over_a)
                    a_completions = count - b_completions
                    
                    # Add to totals
                    b_over_a_completions += b_completions
                    a_over_b_completions += a_completions
                    
                    # Update net votes for B
                    net_votes_for_b += (b_completions - a_completions)
        
        # Check if B wins: net_votes_for_b must exceed the gap
        b_wins = net_votes_for_b >= votes_needed_for_b
        
        # Record result
        bootstrap_results.append(b_wins)
        if b_wins:
            b_win_counts += 1
        
        # Record detailed stats for first few iterations
        if i < 5:
            total_completions = b_over_a_completions + a_over_b_completions
            b_over_a_pct = 100 * b_over_a_completions / total_completions if total_completions > 0 else 0
            
            print(f"\nIteration {i} details:")
            print(f"  Completed {total_completions} partially filled ballots")
            print(f"  B>A: {b_over_a_completions}, A>B: {a_over_b_completions}, B>A%: {b_over_a_pct:.2f}%")
            print(f"  Net votes gained by B: {net_votes_for_b}")
            print(f"  Votes needed for B to win: {votes_needed_for_b}")
            print(f"  B wins: {b_wins}")
    
    # Calculate probability and confidence interval
    b_win_probability = b_win_counts / n_bootstrap
    
    # 95% confidence interval
    alpha = 0.05
    se = np.sqrt((b_win_probability * (1 - b_win_probability)) / n_bootstrap)
    ci_lower = max(0, b_win_probability - 1.96 * se)
    ci_upper = min(1, b_win_probability + 1.96 * se)
    
    # Report results
    print(f"\nLimited Ranking Bootstrap Results (max {max_rankings} rankings):")
    print(f"Probability B wins: {b_win_probability:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"B won in {b_win_counts} out of {n_bootstrap} iterations")
    if required_preference_pct is not None:
        print(f"Required preference: {required_preference_pct:.2f}%")
    
    return b_win_probability, (ci_lower, ci_upper), bootstrap_results 

def unconditional_bootstrap(ballot_counts, candidates, exhausted_ballots, gap_to_win_pct, exhaust_pct, required_preference_pct=None, n_bootstrap=1000, max_rankings=6):
    """
    Bootstrap simulation without conditioning on first preferences.
    
    This function:
    1. Takes all exhausted ballots with partial ranks (less than 6 choices) and no mark on A or B
    2. Samples completions from ALL non-exhausted ballots that ranked either A or B or both,
       regardless of their first preference
    3. Computes the probability of B winning
    """
    print(f"Running unconditional bootstrap with {n_bootstrap} iterations (max rankings: {max_rankings})...")
    
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    print(f"Total ballots in election: {total_ballots}")
    
    # Calculate the gap in votes directly from the gap percentage
    gap_votes = int(gap_to_win_pct * total_ballots / 100)
    print(f"Gap in votes (A leads B by): {gap_votes}")
    
    # Votes needed for B to win: gap + 1
    votes_needed_for_b = gap_votes + 1
    print(f"B needs {votes_needed_for_b} additional votes to win")
    
    # Filter exhausted ballots to only include those with fewer than max_rankings
    # and don't rank either A or B
    partial_exhausted_ballots = {}
    for ballot, count in exhausted_ballots.items():
        if ballot and len(ballot) < max_rankings and 'A' not in ballot and 'B' not in ballot:
            partial_exhausted_ballots[ballot] = count
    
    # Count all exhausted ballots and partial ones
    total_exhausted = sum(exhausted_ballots.values())
    total_partial_exhausted = sum(partial_exhausted_ballots.values())
    
    print(f"Total exhausted ballots: {total_exhausted}")
    print(f"Total partially filled exhausted ballots without A or B (< {max_rankings} rankings): {total_partial_exhausted} ({100*total_partial_exhausted/total_exhausted:.2f}%)")
    
    # Categorize partially filled exhausted ballots by first preference (just for reporting)
    exh_by_first_pref = {}
    for ballot, count in partial_exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Print first preference distribution in partially filled exhausted ballots
    print(f"\nPartially filled exhausted ballots (< {max_rankings} rankings) by first preference:")
    for pref, count in sorted(exh_by_first_pref.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pref}: {count} ({100*count/total_partial_exhausted:.2f}%)")
    
    # Identify ALL non-exhausted ballots that express preferences for A and/or B
    # (regardless of first preference)
    all_a_b_ballots = {}
    b_over_a_total = 0
    a_over_b_total = 0
    total_with_both = 0
    only_a_count = 0
    only_b_count = 0
    
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
            
        has_a = 'A' in ballot
        has_b = 'B' in ballot
        
        if has_a or has_b:  # Include ballots that rank either A or B or both
            all_a_b_ballots[ballot] = count
            
            if has_a and has_b:
                total_with_both += count
                # Record A>B vs B>A preference
                if ballot.index('B') < ballot.index('A'):
                    b_over_a_total += count
                else:
                    a_over_b_total += count
            elif has_a:
                only_a_count += count
                a_over_b_total += count  # Assume A>B if only A is ranked
            elif has_b:
                only_b_count += count
                b_over_a_total += count  # Assume B>A if only B is ranked
    
    # Calculate overall A>B and B>A percentages
    total_ranked = only_a_count + only_b_count + total_with_both
    
    print("\nOverall A/B preference statistics:")
    print(f"  Total ballots ranking A and/or B: {total_ranked}")
    print(f"  Only A: {only_a_count} ({100*only_a_count/total_ranked:.2f}%)")
    print(f"  Only B: {only_b_count} ({100*only_b_count/total_ranked:.2f}%)")
    print(f"  Both A and B: {total_with_both} ({100*total_with_both/total_ranked:.2f}%)")
    
    if b_over_a_total + a_over_b_total > 0:
        overall_b_over_a_pct = 100 * b_over_a_total / (b_over_a_total + a_over_b_total)
        print(f"  Overall preference: B>A: {b_over_a_total} ({overall_b_over_a_pct:.2f}%), A>B: {a_over_b_total} ({100-overall_b_over_a_pct:.2f}%)")
    
    # Run bootstrap iterations
    bootstrap_results = []
    b_win_counts = 0
    
    # Calculate probability that B is preferred over A across all ballots
    overall_prob_b_over_a = b_over_a_total / (b_over_a_total + a_over_b_total) if (b_over_a_total + a_over_b_total) > 0 else 0.5
    
    # Results for each iteration
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
        
        # Track net votes gained by B (B>A minus A>B)
        net_votes_for_b = 0
        
        # Track completions
        b_over_a_completions = 0
        a_over_b_completions = 0
        
        # Sample using overall distribution
        b_completions = np.random.binomial(total_partial_exhausted, overall_prob_b_over_a)
        a_completions = total_partial_exhausted - b_completions
        
        # Add to totals
        b_over_a_completions += b_completions
        a_over_b_completions += a_completions
        
        # Update net votes for B
        net_votes_for_b += (b_completions - a_completions)
        
        # Check if B wins: net_votes_for_b must exceed the gap
        b_wins = net_votes_for_b >= votes_needed_for_b
        
        # Record result
        bootstrap_results.append(b_wins)
        if b_wins:
            b_win_counts += 1
        
        # Record detailed stats for first few iterations
        if i < 5:
            total_completions = b_over_a_completions + a_over_b_completions
            b_over_a_pct = 100 * b_over_a_completions / total_completions if total_completions > 0 else 0
            
            print(f"\nIteration {i} details:")
            print(f"  Completed {total_completions} partially filled ballots")
            print(f"  B>A: {b_over_a_completions}, A>B: {a_over_b_completions}, B>A%: {b_over_a_pct:.2f}%")
            print(f"  Net votes gained by B: {net_votes_for_b}")
            print(f"  Votes needed for B to win: {votes_needed_for_b}")
            print(f"  B wins: {b_wins}")
    
    # Calculate probability and confidence interval
    b_win_probability = b_win_counts / n_bootstrap
    
    # 95% confidence interval
    alpha = 0.05
    se = np.sqrt((b_win_probability * (1 - b_win_probability)) / n_bootstrap)
    ci_lower = max(0, b_win_probability - 1.96 * se)
    ci_upper = min(1, b_win_probability + 1.96 * se)
    
    # Report results
    print(f"\nUnconditional Bootstrap Results:")
    print(f"Probability B wins: {b_win_probability:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"B won in {b_win_counts} out of {n_bootstrap} iterations")
    if required_preference_pct is not None:
        print(f"Required preference: {required_preference_pct:.2f}%")
    
    # Calculate expected net votes more directly based on probability
    expected_b_over_a = total_partial_exhausted * overall_prob_b_over_a
    expected_a_over_b = total_partial_exhausted * (1 - overall_prob_b_over_a)
    expected_net_for_b = expected_b_over_a - expected_a_over_b
    
    print(f"\nExpected completions based on overall probabilities:")
    print(f"  Expected B>A: {expected_b_over_a:.2f}")
    print(f"  Expected A>B: {expected_a_over_b:.2f}")
    print(f"  Expected net for B: {expected_net_for_b:.2f}")
    print(f"  B would win based on expectations: {expected_net_for_b >= votes_needed_for_b}")
    
    return b_win_probability, (ci_lower, ci_upper), bootstrap_results 

def process_election_data():
    """Process NYC and Alaska election data, filtering for exhaust > strategy only"""
    print("Loading and processing election data...")
    
    # Create output directory
    os.makedirs('bootstrap_figures', exist_ok=True)
    
    try:
        # Load NYC data
        nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
        nyc_df = pd.read_excel(nyc_path)
        nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
        print(f"Loaded NYC data: {len(nyc_df)} rows")
        
        # Load NYC ballot data
        nyc_ballot_data = {}
        nyc_folder = "Case_Studies/NewYork_City_RCV/nyc_files"
        for input_file in os.listdir(nyc_folder):
            if input_file.endswith('.csv'):
                full_path = os.path.join(nyc_folder, input_file)
                df, processed_file = process_single_file(full_path)
                candidates_mapping = create_candidate_mapping(processed_file)
                ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
                nyc_ballot_data[input_file] = {
                    'ballot_counts': ballot_counts,
                    'candidates': list(candidates_mapping.values()),
                    'df': df,
                    'candidates_mapping': candidates_mapping
                }
    except Exception as e:
        print(f"Error loading NYC data: {e}")
        nyc_df = pd.DataFrame()
        nyc_ballot_data = {}
    
    try:
        # Load Alaska data
        alaska_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_alska_lite.xlsx"
        alaska_df = pd.read_excel(alaska_path)
        print(f"Loaded Alaska data: {len(alaska_df)} rows")
        
        # Load Alaska ballot data
        alaska_ballot_data = {}
        alaska_folder = "Case_Studies/Alaska_RCV/alaska_files"
        for input_file in os.listdir(alaska_folder):
            if input_file.endswith('.csv'):
                full_path = os.path.join(alaska_folder, input_file)
                df, processed_file = process_single_file(full_path)
                candidates_mapping = create_candidate_mapping(processed_file)
                ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
                alaska_ballot_data[input_file] = {
                    'ballot_counts': ballot_counts,
                    'candidates': list(candidates_mapping.values()),
                    'df': df,
                    'candidates_mapping': candidates_mapping
                }
    except Exception as e:
        print(f"Error loading Alaska data: {e}")
        alaska_df = pd.DataFrame()
        alaska_ballot_data = {}
    
    # Process NYC data - ONLY where exhaust > strategy
    nyc_data = []
    for idx, row in nyc_df.iterrows():
        try:
            strategy_dict = extract_strategy_dict(row['Strategies'])
            exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
            
            # Skip if no data
            if not strategy_dict or not exhaust_dict:
                print(f"Skipping NYC row {idx}: Missing strategy or exhaust data")
                continue
            
            # Process each letter (excluding A)
            for letter, strategy_val in strategy_dict.items():
                if letter == 'A':
                    continue
                
                if letter in exhaust_dict:
                    exhaust_val = exhaust_dict[letter]
                    
                    # ONLY include cases where exhaust > strategy
                    if exhaust_val > strategy_val:
                        diff = exhaust_val - strategy_val
                        
                        # Get ballot data for this election
                        ballot_data = nyc_ballot_data.get(row['file_name'], {})
                        ballot_counts = ballot_data.get('ballot_counts', {})
                        candidates = ballot_data.get('candidates', [])
                        df = ballot_data.get('df', None)
                        candidates_mapping = ballot_data.get('candidates_mapping', {})
                        
                        if df is None or not candidates_mapping:
                            print(f"Missing data for NYC file: {row['file_name']}")
                            continue
                        
                        # Create final mapping and updated ballot counts
                        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, 1, sum(ballot_counts.values())/(1+1))
                        results, subresults = utils.return_main_sub(rt)
                        reverse_mapping = {code: name for name, code in candidates_mapping.items()}
                        ordered_candidate_names = [reverse_mapping[code] for code in results]
                        final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
                        updated_ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
                        
                        # Extract exhausted ballots
                        exhausted_ballots = {ballot: count for ballot, count in updated_ballot_counts.items() if 'A' not in ballot and 'B' not in ballot}
                        
                        nyc_data.append({
                            'letter': letter,
                            'diff': diff,
                            'strategy': strategy_val,
                            'exhaust': exhaust_val,
                            'region': 'NYC',
                            'election_id': row.get('file_name', f"NYC_{idx}"),
                            'ballot_counts': updated_ballot_counts,
                            'candidates': list(final_mapping.values()),
                            'exhausted_ballots': exhausted_ballots
                        })
                        print(f"Processed NYC row {idx}, letter {letter}: exhaust={exhaust_val}, strategy={strategy_val}, diff={diff}")
        except Exception as e:
            print(f"Error processing NYC row {idx}: {e}")
    
    # Process Alaska data - ONLY where exhaust > strategy
    alaska_data = []
    for idx, row in alaska_df.iterrows():
        try:
            strategy_dict = extract_strategy_dict(row['Strategies'])
            exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
            
            # Skip if no data
            if not strategy_dict or not exhaust_dict:
                print(f"Skipping Alaska row {idx}: Missing strategy or exhaust data")
                continue
            
            # Process each letter (excluding A)
            for letter, strategy_val in strategy_dict.items():
                if letter == 'A':
                    continue
                
                if letter in exhaust_dict:
                    exhaust_val = exhaust_dict[letter]
                    
                    # ONLY include cases where exhaust > strategy
                    if exhaust_val > strategy_val:
                        diff = exhaust_val - strategy_val
                        
                        # Get ballot data for this election
                        ballot_data = alaska_ballot_data.get(row['file_name'], {})
                        ballot_counts = ballot_data.get('ballot_counts', {})
                        candidates = ballot_data.get('candidates', [])
                        df = ballot_data.get('df', None)
                        candidates_mapping = ballot_data.get('candidates_mapping', {})
                        
                        if df is None or not candidates_mapping:
                            print(f"Missing data for Alaska file: {row['file_name']}")
                            continue
                        
                        # Create final mapping and updated ballot counts
                        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, 1, sum(ballot_counts.values())/(1+1))
                        results, subresults = utils.return_main_sub(rt)
                        reverse_mapping = {code: name for name, code in candidates_mapping.items()}
                        ordered_candidate_names = [reverse_mapping[code] for code in results]
                        final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
                        updated_ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
                        
                        # Extract exhausted ballots
                        exhausted_ballots = {ballot: count for ballot, count in updated_ballot_counts.items() if 'A' not in ballot and 'B' not in ballot}
                        
                        alaska_data.append({
                            'letter': letter,
                            'diff': diff,
                            'strategy': strategy_val,
                            'exhaust': exhaust_val,
                            'region': 'Alaska',
                            'election_id': row.get('file_name', f"Alaska_{idx}"),
                            'ballot_counts': updated_ballot_counts,
                            'candidates': list(final_mapping.values()),
                            'exhausted_ballots': exhausted_ballots
                        })
                        print(f"Processed Alaska row {idx}, letter {letter}: exhaust={exhaust_val}, strategy={strategy_val}, diff={diff}")
        except Exception as e:
            print(f"Error processing Alaska row {idx}: {e}")
    
    # Create DataFrames
    nyc_df_analysis = pd.DataFrame(nyc_data)
    alaska_df_analysis = pd.DataFrame(alaska_data)
    
    print(f"NYC analysis DataFrame (exhaust > strategy only): {len(nyc_df_analysis)} rows")
    print(f"Alaska analysis DataFrame (exhaust > strategy only): {len(alaska_df_analysis)} rows")
    
    # Save preprocessed data
    nyc_df_analysis.to_csv('nyc_bootstrap_analysis.csv', index=False)
    alaska_df_analysis.to_csv('alaska_bootstrap_analysis.csv', index=False)
    
    return nyc_df_analysis, alaska_df_analysis

def analyze_nyc_elections(max_elections=None, bootstrap_iters=500, run_models=False):
    """
    Run bootstrap methods on NYC elections and create visualizations
    
    Args:
        max_elections: Maximum number of elections to analyze. If None, analyzes all elections.
        bootstrap_iters: Number of bootstrap iterations to run for each method
        run_models: Whether to run theoretical and Bayesian models
    """
    try:
        # Try to load preprocessed data
        if os.path.exists('nyc_bootstrap_analysis.csv'):
            nyc_df = pd.read_csv('nyc_bootstrap_analysis.csv')
            
            # Convert string representations of dictionaries back to actual dictionaries
            for col in ['ballot_counts', 'candidates', 'exhausted_ballots']:
                if col in nyc_df.columns:
                    nyc_df[col] = nyc_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    
            print(f"Loaded {len(nyc_df)} NYC elections from CSV file")
        else:
            # If no preprocessed data, generate it
            nyc_df = process_election_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create directory for results
    os.makedirs('bootstrap_results', exist_ok=True)
    
    # Store results
    bootstrap_results = []
    
    # Focus on the most competitive elections
    nyc_df_sorted = nyc_df.sort_values('strategy')
    
    # Create result columns
    nyc_df_sorted['category_bootstrap_prob'] = np.nan
    nyc_df_sorted['limited_bootstrap_prob'] = np.nan
    nyc_df_sorted['unconditional_bootstrap_prob'] = np.nan  # New column for unconditional bootstrap
    if run_models:
        nyc_df_sorted['beta_model_prob'] = np.nan
        nyc_df_sorted['posterior_beta_prob'] = np.nan
        nyc_df_sorted['prior_posterior_beta_prob'] = np.nan  # NEW
    
    # Process elections
    elections_to_process = nyc_df_sorted if max_elections is None else nyc_df_sorted.head(max_elections)
    
    print(f"\nAnalyzing {len(elections_to_process)} NYC elections with {bootstrap_iters} bootstrap iterations...")
    
    # Process elections
    for idx, election in elections_to_process.iterrows():
        election_id = election['election_id']
        letter = election['letter']
        gap_to_win_pct = election['strategy']
        exhaust_pct = election['exhaust']
        
        # Calculate required preference percentage
        required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
        required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
        
        print(f"\nAnalyzing election: {election_id}")
        print(f"Candidate: {letter}")
        print(f"Gap to win: {gap_to_win_pct:.2f}%")
        print(f"Exhaust percent: {exhaust_pct:.2f}%")
        print(f"Required preference percentage: {required_preference_pct:.2f}%")
        
        # Get ballot data
        ballot_counts = election['ballot_counts']
        candidates = election['candidates']
        exhausted_ballots = election['exhausted_ballots']
        
        # Get total ballots
        total_ballots = sum(ballot_counts.values())
        
        # Calculate gap votes (votes needed for B to win)
        gap_votes = int(gap_to_win_pct * total_ballots / 100)
        votes_needed_for_b = gap_votes + 1
        
        # Run theoretical models if requested
        if run_models:
            # Run beta model (theoretical)
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            print(f"\n=== THEORETICAL BETA MODEL ===")
            print(f"Beta parameters: a={beta_parameters(gap_to_win_pct)[0]}, b={beta_parameters(gap_to_win_pct)[1]}")
            print(f"Probability B wins: {beta_prob:.4f}")
            
            # Run direct posterior beta model (empirical)
            print(f"\n=== DIRECT POSTERIOR BETA MODEL ===")
            posterior_beta_prob = direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots, gap_to_win_pct)
            print(f"Probability B wins: {posterior_beta_prob:.4f}")
            
            # Run prior-posterior beta model (new, intermediate)
            print(f"\n=== PRIOR-POSTERIOR BETA MODEL ===")
            prior_posterior_beta_prob = prior_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots, gap_to_win_pct)
            print(f"Probability B wins: {prior_posterior_beta_prob:.4f}")
        
        # Filter exhausted ballots to only include those with fewer than max_rankings
        partial_exhausted_ballots = {ballot: count for ballot, count in exhausted_ballots.items() 
                                  if ballot and len(ballot) < MAX_RANKINGS_NYC}
        
        # Count total exhausted and partially exhausted ballots
        total_exhausted = sum(exhausted_ballots.values())
        total_partial_exhausted = sum(partial_exhausted_ballots.values())
        
        # Calculate expected completions based on category probabilities
        expected_b_over_a = 0
        expected_a_over_b = 0
        expected_completions_by_category = {}
        
        # Categorize partially filled exhausted ballots by first preference
        exh_by_first_pref = {}
        for ballot, count in partial_exhausted_ballots.items():
            if not ballot:  # Skip empty ballots
                continue
            first_pref = ballot[0]
            if first_pref not in exh_by_first_pref:
                exh_by_first_pref[first_pref] = 0
            exh_by_first_pref[first_pref] += count
        
        # Identify non-exhausted ballots that have both A and B
        complete_by_first_pref = {}
        for ballot, count in ballot_counts.items():
            if not ballot:
                continue
            if 'A' in ballot and 'B' in ballot:
                first_pref = ballot[0]
                if first_pref not in complete_by_first_pref:
                    complete_by_first_pref[first_pref] = {'ballots': {}, 'b_over_a': 0, 'a_over_b': 0, 'total': 0}
                
                complete_by_first_pref[first_pref]['ballots'][ballot] = count
                complete_by_first_pref[first_pref]['total'] += count
                
                # Record A>B vs B>A preference
                if ballot.index('B') < ballot.index('A'):
                    complete_by_first_pref[first_pref]['b_over_a'] += count
                else:
                    complete_by_first_pref[first_pref]['a_over_b'] += count
        
        # Calculate expected completions for each category
        for first_pref, count in exh_by_first_pref.items():
            # Initialize category data in our tracking dict
            expected_completions_by_category[first_pref] = {
                'total_partial': count,
                'expected_b_over_a': 0,
                'expected_a_over_b': 0
            }
            
            # Check if we have data on preferences for this first preference
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                # Get preference distribution for this category
                category_data = complete_by_first_pref[first_pref]
                prob_b_over_a = category_data['b_over_a'] / category_data['total']
                
                # Calculate expected completions for this category
                expected_b_for_category = count * prob_b_over_a
                expected_a_for_category = count * (1 - prob_b_over_a)
                
                # Update expectation tracking
                expected_completions_by_category[first_pref]['expected_b_over_a'] = expected_b_for_category
                expected_completions_by_category[first_pref]['expected_a_over_b'] = expected_a_for_category
                
                # Add to total expectations
                expected_b_over_a += expected_b_for_category
                expected_a_over_b += expected_a_for_category
            else:
                # If no data for this category, use the overall distribution
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
                
                if total_complete > 0:
                    overall_prob_b_over_a = total_b_over_a / total_complete
                    
                    # Calculate expected completions for this category
                    expected_b_for_category = count * overall_prob_b_over_a
                    expected_a_for_category = count * (1 - overall_prob_b_over_a)
                    
                    # Update expectation tracking
                    expected_completions_by_category[first_pref]['expected_b_over_a'] = expected_b_for_category
                    expected_completions_by_category[first_pref]['expected_a_over_b'] = expected_a_for_category
                    
                    # Add to total expectations
                    expected_b_over_a += expected_b_for_category
                    expected_a_over_b += expected_a_for_category
        
        # Run category-based bootstrap
        cat_bootstrap_prob, cat_bootstrap_ci, _ = category_based_bootstrap(
            ballot_counts, candidates, exhausted_ballots, 
            gap_to_win_pct=gap_to_win_pct,
            exhaust_pct=exhaust_pct,
            required_preference_pct=required_preference_pct,
            n_bootstrap=bootstrap_iters
        )
        
        # Run limited ranking bootstrap (NYC 6-candidate limit)
        limited_bootstrap_prob, limited_bootstrap_ci, _ = limited_ranking_bootstrap(
            ballot_counts, candidates, exhausted_ballots, 
            gap_to_win_pct=gap_to_win_pct,
            exhaust_pct=exhaust_pct,
            required_preference_pct=required_preference_pct,
            n_bootstrap=bootstrap_iters,
            max_rankings=MAX_RANKINGS_NYC
        )
        
        # Run unconditional bootstrap (new method)
        unconditional_bootstrap_prob, unconditional_bootstrap_ci, _ = unconditional_bootstrap(
            ballot_counts, candidates, exhausted_ballots, 
            gap_to_win_pct=gap_to_win_pct,
            exhaust_pct=exhaust_pct,
            required_preference_pct=required_preference_pct,
            n_bootstrap=bootstrap_iters,
            max_rankings=MAX_RANKINGS_NYC
        )
        
        # Record results
        result = {
            'election_id': election_id,
            'letter': letter,
            'gap_to_win_pct': gap_to_win_pct,
            'exhaust_pct': exhaust_pct,
            'required_preference_pct': required_preference_pct,
            'strategy_exhaust_ratio': gap_to_win_pct / exhaust_pct,
            'category_bootstrap_prob': cat_bootstrap_prob,
            'category_bootstrap_ci_lower': cat_bootstrap_ci[0],
            'category_bootstrap_ci_upper': cat_bootstrap_ci[1],
            'limited_bootstrap_prob': limited_bootstrap_prob,
            'limited_bootstrap_ci_lower': limited_bootstrap_ci[0],
            'limited_bootstrap_ci_upper': limited_bootstrap_ci[1],
            'unconditional_bootstrap_prob': unconditional_bootstrap_prob,
            'unconditional_bootstrap_ci_lower': unconditional_bootstrap_ci[0],
            'unconditional_bootstrap_ci_upper': unconditional_bootstrap_ci[1],
            # Add detailed statistics about expected completions
            'total_ballots': total_ballots,
            'gap_votes': gap_votes,
            'votes_needed_for_b': votes_needed_for_b,
            'total_exhausted': total_exhausted,
            'total_partial_exhausted': total_partial_exhausted,
            'expected_b_over_a': expected_b_over_a,
            'expected_a_over_b': expected_a_over_b,
            'expected_net_for_b': expected_b_over_a - expected_a_over_b,
            'category_completions': expected_completions_by_category
        }
        
        # Add model results if they were calculated
        if run_models:
            result['beta_model_prob'] = beta_prob
            result['posterior_beta_prob'] = posterior_beta_prob
            result['prior_posterior_beta_prob'] = prior_posterior_beta_prob  # NEW
        
        # Print expected completion statistics
        print(f"\nDetailed Completion Statistics:")
        print(f"Total ballots: {total_ballots}")
        print(f"Gap votes (A leads B by): {gap_votes}")
        print(f"Votes needed for B to win: {votes_needed_for_b}")
        print(f"Total exhausted ballots: {total_exhausted}")
        print(f"Total partially filled exhausted ballots: {total_partial_exhausted}")
        print(f"Expected B>A completions: {expected_b_over_a:.2f}")
        print(f"Expected A>B completions: {expected_a_over_b:.2f}")
        print(f"Expected net votes for B: {expected_b_over_a - expected_a_over_b:.2f}")
        
        # Check if expected completions would give B enough votes to win
        would_b_win = (expected_b_over_a - expected_a_over_b) >= votes_needed_for_b
        print(f"Would B win based on expected completions? {would_b_win}")
        
        # Print category-by-category breakdown
        print("\nCategory-by-category completion breakdown:")
        for category, data in expected_completions_by_category.items():
            print(f"  Category {category}: {data['total_partial']} ballots")
            print(f"    Expected B>A: {data['expected_b_over_a']:.2f}, A>B: {data['expected_a_over_b']:.2f}")
            print(f"    Net for B: {data['expected_b_over_a'] - data['expected_a_over_b']:.2f}")
        
        bootstrap_results.append(result)
        
        # Print summary of bootstrap probabilities
        print("\nBootstrap Probability Summary:")
        print(f"  Category-based bootstrap: {cat_bootstrap_prob:.4f}")
        print(f"  Limited ranking bootstrap: {limited_bootstrap_prob:.4f}")
        print(f"  Unconditional bootstrap: {unconditional_bootstrap_prob:.4f}")
        if run_models:
            print(f"  Beta model: {beta_prob:.4f}")
            print(f"  Posterior Beta model: {posterior_beta_prob:.4f}")
            print(f"  Prior-Posterior Beta model: {prior_posterior_beta_prob:.4f}")
    
    # Create DataFrame with results - handle the nested dictionary for category_completions
    results_df = pd.DataFrame(bootstrap_results)
    
    # Save results to CSV, but exclude the complex nested categories
    simple_results_df = results_df.copy()
    if 'category_completions' in simple_results_df.columns:
        simple_results_df = simple_results_df.drop(columns=['category_completions'])
    simple_results_df.to_csv('bootstrap_results/nyc_analysis_comparison.csv', index=False)
    
    # Save detailed category breakdown to a separate file
    with open('bootstrap_results/nyc_analysis_detailed.txt', 'w') as f:
        for idx, row in results_df.iterrows():
            f.write(f"Election: {row['election_id']}, Candidate: {row['letter']}\n")
            f.write(f"Gap votes: {row['gap_votes']}, Votes needed for B: {row['votes_needed_for_b']}\n")
            f.write(f"Total exhausted: {row['total_exhausted']}, Partially exhausted: {row['total_partial_exhausted']}\n")
            f.write(f"Expected B>A: {row['expected_b_over_a']:.2f}, Expected A>B: {row['expected_a_over_b']:.2f}\n")
            f.write(f"Expected net for B: {row['expected_net_for_b']:.2f}\n")
            
            f.write(f"Category bootstrap probability: {row['category_bootstrap_prob']:.4f}\n")
            f.write(f"Limited ranking bootstrap probability: {row['limited_bootstrap_prob']:.4f}\n")
            f.write(f"Unconditional bootstrap probability: {row['unconditional_bootstrap_prob']:.4f}\n")
            
            if run_models:
                f.write(f"Beta model probability: {row.get('beta_model_prob', 'N/A')}\n")
                f.write(f"Posterior Beta probability: {row.get('posterior_beta_prob', 'N/A')}\n")
                f.write(f"Prior-Posterior Beta probability: {row.get('prior_posterior_beta_prob', 'N/A')}\n")
            
            f.write("\n")
            
            if 'category_completions' in row:
                f.write("Category breakdown:\n")
                for category, data in row['category_completions'].items():
                    f.write(f"  Category {category}: {data['total_partial']} ballots\n")
                    f.write(f"    Expected B>A: {data['expected_b_over_a']:.2f}, A>B: {data['expected_a_over_b']:.2f}\n")
                    f.write(f"    Net for B: {data['expected_b_over_a'] - data['expected_a_over_b']:.2f}\n")
                f.write("\n---\n\n")
    
    return results_df

def analyze_specific_election(election_id, letter=None, max_rankings=MAX_RANKINGS_NYC, bootstrap_iters=1000, run_models=False):
    """
    Analyze a specific election with bootstrap methods and optional theoretical models
    
    Args:
        election_id: The ID of the election to analyze
        letter: The candidate letter (if None, will use the first available candidate)
        max_rankings: Maximum number of rankings allowed (default: NYC's limit of 6)
        bootstrap_iters: Number of bootstrap iterations to run for each method
        run_models: Whether to run theoretical and Bayesian models
    """
    try:
        # Load preprocessed data
        if os.path.exists('nyc_bootstrap_analysis.csv'):
            nyc_df = pd.read_csv('nyc_bootstrap_analysis.csv')
            
            # Convert string representations of dictionaries back to actual dictionaries
            for col in ['ballot_counts', 'candidates', 'exhausted_ballots']:
                if col in nyc_df.columns:
                    nyc_df[col] = nyc_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        else:
            print("Preprocessed data not found. Please run process_election_data() first.")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Filter to the specific election
    election_data = nyc_df[nyc_df['election_id'] == election_id]
    
    if election_data.empty:
        print(f"Election with ID {election_id} not found.")
        return
    
    # If letter is provided, filter to that candidate
    if letter is not None:
        election_data = election_data[election_data['letter'] == letter]
        
        if election_data.empty:
            print(f"Candidate {letter} not found for election {election_id}.")
            return
    
    # Use the first matching election
    election = election_data.iloc[0]
    letter = election['letter']
    
    # Calculate required preference percentage
    gap_to_win_pct = election['strategy']
    exhaust_pct = election['exhaust']
    required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
    required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
    
    print(f"\nAnalyzing election: {election_id}")
    print(f"Candidate: {letter}")
    print(f"Gap to win: {gap_to_win_pct:.2f}%")
    print(f"Exhaust percent: {exhaust_pct:.2f}%")
    print(f"Required preference percentage: {required_preference_pct:.2f}%")
    print(f"Bootstrap iterations: {bootstrap_iters}")
    
    # Get ballot data
    ballot_counts = election['ballot_counts']
    candidates = election['candidates']
    exhausted_ballots = election['exhausted_ballots']
    
    # Run theoretical models if requested
    if run_models:
        # Run beta model (theoretical)
        beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
        print(f"\n=== THEORETICAL BETA MODEL ===")
        print(f"Beta parameters: a={beta_parameters(gap_to_win_pct)[0]}, b={beta_parameters(gap_to_win_pct)[1]}")
        print(f"Probability B wins: {beta_prob:.4f}")
        
        # Run direct posterior beta model (empirical)
        print(f"\n=== DIRECT POSTERIOR BETA MODEL ===")
        posterior_beta_prob = direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots, gap_to_win_pct)
        print(f"Probability B wins: {posterior_beta_prob:.4f}")
        
        # Run prior-posterior beta model (new, intermediate)
        print(f"\n=== PRIOR-POSTERIOR BETA MODEL ===")
        prior_posterior_beta_prob = prior_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots, gap_to_win_pct)
        print(f"Probability B wins: {prior_posterior_beta_prob:.4f}")
    
    # Run category-based bootstrap with detailed output
    print("\n=== CATEGORY-BASED BOOTSTRAP ANALYSIS ===")
    cat_bootstrap_prob, cat_bootstrap_ci, cat_results = category_based_bootstrap(
        ballot_counts, candidates, exhausted_ballots, 
        gap_to_win_pct=gap_to_win_pct,
        exhaust_pct=exhaust_pct,
        required_preference_pct=required_preference_pct,
        n_bootstrap=bootstrap_iters
    )
    
    # Run limited ranking bootstrap with detailed output
    print(f"\n=== LIMITED RANKING BOOTSTRAP ANALYSIS (MAX {max_rankings} RANKINGS) ===")
    limited_bootstrap_prob, limited_bootstrap_ci, limited_results = limited_ranking_bootstrap(
        ballot_counts, candidates, exhausted_ballots, 
        gap_to_win_pct=gap_to_win_pct,
        exhaust_pct=exhaust_pct,
        required_preference_pct=required_preference_pct,
        n_bootstrap=bootstrap_iters,
        max_rankings=max_rankings
    )
    
    # Run unconditional bootstrap with detailed output
    print(f"\n=== UNCONDITIONAL BOOTSTRAP ANALYSIS ===")
    unconditional_bootstrap_prob, unconditional_bootstrap_ci, unconditional_results = unconditional_bootstrap(
        ballot_counts, candidates, exhausted_ballots, 
        gap_to_win_pct=gap_to_win_pct,
        exhaust_pct=exhaust_pct,
        required_preference_pct=required_preference_pct,
        n_bootstrap=bootstrap_iters,
        max_rankings=max_rankings
    )
    
    # Compare results
    print("\n=== METHOD COMPARISON ===")
    print(f"Category Bootstrap: {cat_bootstrap_prob:.4f} ({cat_bootstrap_ci[0]:.4f}, {cat_bootstrap_ci[1]:.4f})")
    print(f"Limited Ranking Bootstrap: {limited_bootstrap_prob:.4f} ({limited_bootstrap_ci[0]:.4f}, {limited_bootstrap_ci[1]:.4f})")
    print(f"Unconditional Bootstrap: {unconditional_bootstrap_prob:.4f} ({unconditional_bootstrap_ci[0]:.4f}, {unconditional_bootstrap_ci[1]:.4f})")
    
    if run_models:
        print(f"Beta Model: {beta_prob:.4f}")
        print(f"Direct Posterior Beta: {posterior_beta_prob:.4f}")
        print(f"Prior-Posterior Beta: {prior_posterior_beta_prob:.4f}")
    
    # Removed individual election plots
    
    return {
        'election_id': election_id,
        'letter': letter,
        'gap_to_win_pct': gap_to_win_pct,
        'exhaust_pct': exhaust_pct,
        'required_preference_pct': required_preference_pct,
        'category_bootstrap_prob': cat_bootstrap_prob,
        'category_bootstrap_ci': cat_bootstrap_ci,
        'limited_bootstrap_prob': limited_bootstrap_prob,
        'limited_bootstrap_ci': limited_bootstrap_ci,
        'unconditional_bootstrap_prob': unconditional_bootstrap_prob, 
        'unconditional_bootstrap_ci': unconditional_bootstrap_ci,
        'beta_model_prob': beta_prob if run_models else None,
        'posterior_beta_prob': posterior_beta_prob if run_models else None,
        'prior_posterior_beta_prob': prior_posterior_beta_prob if run_models else None
    }

def create_bootstrap_comparison_plots(results_df):
    """
    Create comprehensive comparison plots for both bootstrap methods
    showing results across all elections together.
    
    Args:
        results_df: DataFrame containing bootstrap results for all elections
    """
    if results_df.empty:
        print("No results available to create comparison plots.")
        return
    
    print("\nCreating comprehensive bootstrap comparison plots...")
    
    # Create output directory
    os.makedirs('bootstrap_results', exist_ok=True)
    
    # 1. Bootstrap probability comparison - Bar chart for all elections together
    plt.figure(figsize=(16, 8))
    
    # Sort elections by gap_to_win_pct for better visualization
    sorted_df = results_df.sort_values('gap_to_win_pct')
    
    # Create x positions for grouped bars
    x = np.arange(len(sorted_df))
    width = 0.35
    
    # Plot bars for each method
    plt.bar(x - width/2, sorted_df['category_bootstrap_prob'], width, 
           color='darkgreen', alpha=0.7, label='Category Bootstrap')
    plt.bar(x + width/2, sorted_df['limited_bootstrap_prob'], width, 
           color='purple', alpha=0.7, label='Limited Ranking Bootstrap')
    
    # Add model probabilities if they exist
    if 'beta_model_prob' in sorted_df.columns:
        plt.plot(x, sorted_df['beta_model_prob'], 'o-', color='blue', linewidth=2, 
                markersize=8, label='Beta Model')
    if 'posterior_beta_prob' in sorted_df.columns:
        plt.plot(x, sorted_df['posterior_beta_prob'], 's-', color='orange', linewidth=2, 
                markersize=8, label='Posterior Beta')
    if 'prior_posterior_beta_prob' in sorted_df.columns:
        plt.plot(x, sorted_df['prior_posterior_beta_prob'], 'd-', color='cyan', linewidth=2, 
                markersize=8, label='Prior-Posterior Beta')
    
    # Add election labels
    plt.xticks(x, [f"{row['election_id'].split('/')[-1][:15]}...\n({row['letter']})" 
                  for _, row in sorted_df.iterrows()], rotation=45, ha='right', fontsize=10)
    
    # Add horizontal line at 0.5 probability
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title('Comparison of Methods Across All NYC Elections', fontsize=16)
    plt.ylabel('Probability of Candidate B Winning', fontsize=14)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('bootstrap_results/all_elections_methods_comparison.png', dpi=300)
    plt.close()
    
    # 2. Create scatterplot of both methods vs strategy/exhaust ratio
    plt.figure(figsize=(14, 10))
    
    # Category bootstrap
    plt.scatter(sorted_df['strategy_exhaust_ratio'], sorted_df['category_bootstrap_prob'], 
               s=150, marker='o', color='darkgreen', alpha=0.7, label='Category Bootstrap')
    
    # Limited ranking bootstrap
    plt.scatter(sorted_df['strategy_exhaust_ratio'], sorted_df['limited_bootstrap_prob'], 
               s=150, marker='s', color='purple', alpha=0.7, label='Limited Ranking Bootstrap')
    
    # Add model probabilities if they exist
    if 'beta_model_prob' in sorted_df.columns:
        plt.scatter(sorted_df['strategy_exhaust_ratio'], sorted_df['beta_model_prob'],
                   s=150, marker='^', color='blue', alpha=0.7, label='Beta Model')
    if 'posterior_beta_prob' in sorted_df.columns:
        plt.scatter(sorted_df['strategy_exhaust_ratio'], sorted_df['posterior_beta_prob'],
                   s=150, marker='d', color='orange', alpha=0.7, label='Posterior Beta')
    if 'prior_posterior_beta_prob' in sorted_df.columns:
        plt.scatter(sorted_df['strategy_exhaust_ratio'], sorted_df['prior_posterior_beta_prob'],
                   s=150, marker='*', color='cyan', alpha=0.7, label='Prior-Posterior Beta')
    
    # Connect points from same election with dashed lines
    for i, row in sorted_df.iterrows():
        methods = [row['category_bootstrap_prob'], row['limited_bootstrap_prob']]
        if 'beta_model_prob' in sorted_df.columns:
            methods.append(row['beta_model_prob'])
        if 'posterior_beta_prob' in sorted_df.columns:
            methods.append(row['posterior_beta_prob'])
        if 'prior_posterior_beta_prob' in sorted_df.columns:
            methods.append(row['prior_posterior_beta_prob'])
        
        plt.plot([row['strategy_exhaust_ratio']] * len(methods), methods, 
                'k--', alpha=0.4, linewidth=1)
    
    # Add horizontal line at 0.5 probability
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    
    # Add annotations for key elections
    for i, row in sorted_df.iterrows():
        # Annotate all points
            plt.annotate(f"{row['election_id'].split('/')[-1][:10]}...({row['letter']})",
                       (row['strategy_exhaust_ratio'], 
                    row['category_bootstrap_prob']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add title and labels
    plt.title('Method Probabilities vs Strategy/Exhaust Ratio', fontsize=16)
    plt.xlabel('Strategy/Exhaust Ratio', fontsize=14)
    plt.ylabel('Probability of Candidate B Winning', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('bootstrap_results/strategy_exhaust_ratio_all_methods.png', dpi=300)
    plt.close()
    
    # 3. Create a simpler observed vs. required preferences plot
    # Instead of calculating from scratch, just plot the required percentage
    plt.figure(figsize=(14, 8))
    
    # Plot bars for required preferences
    plt.bar(x, sorted_df['required_preference_pct'], width=0.6, color='red', alpha=0.7, 
           label='Required B>A Preference')
        
    # Add horizontal line at 50% preference
    plt.axhline(50, color='black', linestyle='--', alpha=0.5, label='Neutral (50%)')
    
    # Add election labels
    plt.xticks(x, [f"{row['election_id'].split('/')[-1][:15]}...\n({row['letter']})" 
              for _, row in sorted_df.iterrows()], rotation=45, ha='right', fontsize=10)
    
    # Add title and labels
    plt.title('Required Preferences for Candidate B to Win', fontsize=16)
    plt.ylabel('B>A Preference Percentage', fontsize=14)
    plt.ylim(0, 110)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('bootstrap_results/required_preferences.png', dpi=300)
    plt.close()
    
    # 4. Plot expected net votes vs votes needed
    plt.figure(figsize=(14, 8))
    
    # Calculate points where expected completions equal votes needed
    sorted_df['expected_completions_enough'] = sorted_df['expected_net_for_b'] >= sorted_df['votes_needed_for_b']
    
    # Plot with different colors depending on whether expected completions are enough
    enough = sorted_df[sorted_df['expected_completions_enough']]
    not_enough = sorted_df[~sorted_df['expected_completions_enough']]
    
    plt.scatter(not_enough['votes_needed_for_b'], not_enough['expected_net_for_b'], 
               s=150, alpha=0.7, color='red', marker='o', label='Not Enough for B to Win')
    
    if not enough.empty:
        plt.scatter(enough['votes_needed_for_b'], enough['expected_net_for_b'], 
                   s=150, alpha=0.7, color='green', marker='o', label='Enough for B to Win')
    
    # Add diagonal line where expected net = votes needed
    max_votes = max(sorted_df['votes_needed_for_b'].max(), 
                   sorted_df['expected_net_for_b'].max() if sorted_df['expected_net_for_b'].max() > 0 else 0)
    min_votes = min(sorted_df['expected_net_for_b'].min(), 0)
    plt.plot([0, max_votes * 1.1], [0, max_votes * 1.1], 'k--', alpha=0.5)
    
    # Add annotations for all elections
    for idx, row in sorted_df.iterrows():
        plt.annotate(f"{row['election_id'].split('/')[-1][:10]}...({row['letter']})",
                   (row['votes_needed_for_b'], row['expected_net_for_b']),
                   xytext=(10, 0), textcoords='offset points', fontsize=10)
    
    plt.xlabel('Votes Needed for B to Win', fontsize=14)
    plt.ylabel('Expected Net Votes for B from Completions', fontsize=14)
    plt.title('Expected Net Votes vs. Votes Needed Across All Elections', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set y-axis to include zero
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    y_min = min(sorted_df['expected_net_for_b'].min() * 1.1, -100)
    y_max = max(max_votes * 0.5, 500)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('bootstrap_results/expected_votes_vs_needed_all_elections.png', dpi=300)
    plt.close()
    
    # 5. Summary statistics - gap, exhaust, ratio histogram
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gap histogram
    axs[0].hist(sorted_df['gap_to_win_pct'], bins=5, color='blue', alpha=0.7)
    axs[0].set_title('Gap to Win Distribution', fontsize=14)
    axs[0].set_xlabel('Gap to Win (%)', fontsize=12)
    axs[0].set_ylabel('Count', fontsize=12)
    axs[0].grid(alpha=0.3)
    
    # Exhaust histogram
    axs[1].hist(sorted_df['exhaust_pct'], bins=5, color='green', alpha=0.7)
    axs[1].set_title('Exhaust Percentage Distribution', fontsize=14)
    axs[1].set_xlabel('Exhaust (%)', fontsize=12)
    axs[1].set_ylabel('Count', fontsize=12)
    axs[1].grid(alpha=0.3)
    
    # Ratio histogram
    axs[2].hist(sorted_df['strategy_exhaust_ratio'], bins=5, color='purple', alpha=0.7)
    axs[2].set_title('Strategy/Exhaust Ratio Distribution', fontsize=14)
    axs[2].set_xlabel('Ratio', fontsize=12)
    axs[2].set_ylabel('Count', fontsize=12)
    axs[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bootstrap_results/election_statistics_distributions.png', dpi=300)
    plt.close()
    
    print("Comprehensive comparison plots for all elections together saved to the bootstrap_results directory.")

def create_model_comparison_plot(results_df):
    """
    Create a plot of all probability models vs required preference percentage,
    similar to the style shown in the reference image.
    
    Args:
        results_df: DataFrame containing results from all models
    """
    if results_df.empty:
        print("No results available to create model comparison plot.")
        return
    
    # Create output directory
    os.makedirs('bootstrap_results', exist_ok=True)
    
    # Sort results by required preference percentage for better visualization
    sorted_df = results_df.sort_values('required_preference_pct')
    
    # Create two subplots - one with linear scale, one with log scale
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18))
    
    # Function to add data series to both plots
    def add_data_series(x_data, y_data, marker, linestyle, color, label, linewidth=2):
        ax1.plot(x_data, y_data, marker=marker, linestyle=linestyle, color=color, label=label, linewidth=linewidth)
        
        # Replace zeros with small values for log plot
        y_data_log = np.array([max(y, 1e-15) for y in y_data])
        ax2.plot(x_data, y_data_log, marker=marker, linestyle=linestyle, color=color, label=label, linewidth=linewidth)
    
    # Add data series for each model type
    # Bootstrap models
    if 'limited_bootstrap_prob' in sorted_df.columns:
        add_data_series(sorted_df['required_preference_pct'], sorted_df['limited_bootstrap_prob'], 
                's', '-', 'purple', 'Limited Ranking Bootstrap')
    
    if 'unconditional_bootstrap_prob' in sorted_df.columns:
        add_data_series(sorted_df['required_preference_pct'], sorted_df['unconditional_bootstrap_prob'], 
                '^', '-', 'red', 'Unconditional Bootstrap')
    
    # Theoretical models
    if 'beta_model_prob' in sorted_df.columns:
        add_data_series(sorted_df['required_preference_pct'], sorted_df['beta_model_prob'], 
                'd', '-', 'blue', 'Beta Model')
    
    if 'posterior_beta_prob' in sorted_df.columns:
        add_data_series(sorted_df['required_preference_pct'], sorted_df['posterior_beta_prob'], 
                '*', '-', 'cyan', 'Direct Posterior')
    if 'prior_posterior_beta_prob' in sorted_df.columns:
        add_data_series(sorted_df['required_preference_pct'], sorted_df['prior_posterior_beta_prob'], 
                'o', '-', 'magenta', 'Prior-Posterior Beta')

def analyze_all_elections(bootstrap_iters=1000):
    """
    Run analysis on both NYC and Alaska elections with all five probability models,
    create comprehensive visualizations comparing models, and output detailed CSV.
    
    Args:
        bootstrap_iters: Number of bootstrap iterations for bootstrap models (default: 1000)
    """
    # Create results directory
    os.makedirs('model_comparison_results', exist_ok=True)
    
    # Process NYC elections
    try:
        # Try to load preprocessed NYC data
        if os.path.exists('nyc_bootstrap_analysis.csv'):
            nyc_df = pd.read_csv('nyc_bootstrap_analysis.csv')
            
            # Convert string representations of dictionaries back to actual dictionaries
            for col in ['ballot_counts', 'candidates', 'exhausted_ballots']:
                if col in nyc_df.columns:
                    nyc_df[col] = nyc_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    
            print(f"Loaded {len(nyc_df)} NYC elections from CSV file")
        else:
            # If no preprocessed data, generate it
            nyc_df = process_election_data()
    except Exception as e:
        print(f"Error loading NYC data: {e}")
        nyc_df = pd.DataFrame()
    
    # Process Alaska elections
    try:
        # Try to load preprocessed Alaska data
        if os.path.exists('alaska_bootstrap_analysis.csv'):
            alaska_df = pd.read_csv('alaska_bootstrap_analysis.csv')
            
            # Convert string representations of dictionaries back to actual dictionaries
            for col in ['ballot_counts', 'candidates', 'exhausted_ballots']:
                if col in alaska_df.columns:
                    alaska_df[col] = alaska_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    
            print(f"Loaded {len(alaska_df)} Alaska elections from CSV file")
        else:
            # If no preprocessed data, we'll need a different approach
            # For now, we'll just log this and continue with NYC data
            print("No Alaska preprocessed data found. Processing only NYC elections.")
            alaska_df = pd.DataFrame()
    except Exception as e:
        print(f"Error loading Alaska data: {e}")
        alaska_df = pd.DataFrame()
    
    # Combine NYC and Alaska data if both are available
    all_elections = []
    if not nyc_df.empty:
        all_elections.append(nyc_df)
    if not alaska_df.empty:
        all_elections.append(alaska_df)
    
    if not all_elections:
        print("No election data available for analysis.")
        return
    
    combined_df = pd.concat(all_elections, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} elections")
    
    # Store results
    all_results = []
    
    # Create result columns
    combined_df['category_bootstrap_prob'] = np.nan
    combined_df['limited_bootstrap_prob'] = np.nan
    combined_df['unconditional_bootstrap_prob'] = np.nan
    combined_df['beta_model_prob'] = np.nan
    combined_df['posterior_beta_prob'] = np.nan
    combined_df['prior_posterior_beta_prob'] = np.nan  # NEW COLUMN
    
    # Process all elections
    print(f"\nAnalyzing {len(combined_df)} elections with {bootstrap_iters} bootstrap iterations...")
    
    for idx, election in combined_df.iterrows():
        election_id = election['election_id']
        region = election['region']
        letter = election['letter']
        gap_to_win_pct = election['strategy']
        exhaust_pct = election['exhaust']
        
        # Calculate required preference percentage
        required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
        required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
        
        print(f"\nAnalyzing {region} election: {election_id}")
        print(f"Candidate: {letter}")
        print(f"Gap to win: {gap_to_win_pct:.2f}%")
        print(f"Exhaust percent: {exhaust_pct:.2f}%")
        print(f"Required preference percentage: {required_preference_pct:.2f}%")
        
        # Get ballot data
        ballot_counts = election['ballot_counts']
        candidates = election['candidates']
        exhausted_ballots = election['exhausted_ballots']
        
        # Get total ballots
        total_ballots = sum(ballot_counts.values())
        
        # Calculate gap votes (votes needed for B to win)
        gap_votes = int(gap_to_win_pct * total_ballots / 100)
        votes_needed_for_b = gap_votes + 1
        
        # Run beta model (theoretical)
        beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
        print(f"Beta model probability: {beta_prob:.4f}")
        
        # Run direct posterior beta model (empirical)
        posterior_beta_prob = direct_posterior_beta(required_preference_pct, ballot_counts, 
                                               candidates, exhausted_ballots, gap_to_win_pct)
        print(f"Posterior beta probability: {posterior_beta_prob:.4f}")
        
        # Run prior-posterior beta model (new, intermediate)
        prior_posterior_beta_prob = prior_posterior_beta(required_preference_pct, ballot_counts, 
                                                        candidates, exhausted_ballots, gap_to_win_pct)
        print(f"Prior-Posterior beta probability: {prior_posterior_beta_prob:.4f}")
        
        # Run category-based bootstrap
        cat_bootstrap_prob, cat_bootstrap_ci, _ = category_based_bootstrap(
            ballot_counts, candidates, exhausted_ballots, 
            gap_to_win_pct=gap_to_win_pct,
            exhaust_pct=exhaust_pct,
            required_preference_pct=required_preference_pct,
            n_bootstrap=bootstrap_iters
        )
        print(f"Category bootstrap probability: {cat_bootstrap_prob:.4f}")
        
        # Run limited ranking bootstrap (max 6 rankings) - only for NYC
        if election['region'] == 'Alaska':
            # Skip limited ranking bootstrap for Alaska elections
            limited_bootstrap_prob = np.nan
            limited_bootstrap_ci = (np.nan, np.nan)
            print(f"Skipping limited ranking bootstrap for Alaska")
        else:
            # Run limited ranking bootstrap for NYC elections
            limited_bootstrap_prob, limited_bootstrap_ci, _ = limited_ranking_bootstrap(
                ballot_counts, candidates, exhausted_ballots, 
                gap_to_win_pct=gap_to_win_pct,
                exhaust_pct=exhaust_pct,
                required_preference_pct=required_preference_pct,
                n_bootstrap=bootstrap_iters,
                max_rankings=MAX_RANKINGS_NYC
            )
            print(f"Limited bootstrap probability: {limited_bootstrap_prob:.4f}")
        
        # Run unconditional bootstrap
        unconditional_bootstrap_prob, unconditional_bootstrap_ci, _ = unconditional_bootstrap(
            ballot_counts, candidates, exhausted_ballots, 
            gap_to_win_pct=gap_to_win_pct,
            exhaust_pct=exhaust_pct,
            required_preference_pct=required_preference_pct,
            n_bootstrap=bootstrap_iters,
            max_rankings=MAX_RANKINGS_NYC
        )
        print(f"Unconditional bootstrap probability: {unconditional_bootstrap_prob:.4f}")
        
        # Store the results
        result = {
            'region': region,
            'election_id': election_id,
            'letter': letter,
            'gap_to_win_pct': gap_to_win_pct,
            'exhaust_pct': exhaust_pct,
            'required_preference_pct': required_preference_pct,
            'strategy_exhaust_ratio': gap_to_win_pct / exhaust_pct,
            
            # Model probabilities
            'beta_model_prob': beta_prob,
            'posterior_beta_prob': posterior_beta_prob,
            'prior_posterior_beta_prob': prior_posterior_beta_prob,  # NEW
            'category_bootstrap_prob': cat_bootstrap_prob,
            'limited_bootstrap_prob': limited_bootstrap_prob,
            'unconditional_bootstrap_prob': unconditional_bootstrap_prob,
            
            # Confidence intervals for bootstrap methods
            'category_bootstrap_ci_lower': cat_bootstrap_ci[0],
            'category_bootstrap_ci_upper': cat_bootstrap_ci[1],
            'limited_bootstrap_ci_lower': limited_bootstrap_ci[0],
            'limited_bootstrap_ci_upper': limited_bootstrap_ci[1],
            'unconditional_bootstrap_ci_lower': unconditional_bootstrap_ci[0],
            'unconditional_bootstrap_ci_upper': unconditional_bootstrap_ci[1],
            
            # Additional stats
            'total_ballots': total_ballots,
            'gap_votes': gap_votes,
            'votes_needed_for_b': votes_needed_for_b
        }
        
        # Update the combined dataframe
        combined_df.loc[idx, 'beta_model_prob'] = beta_prob
        combined_df.loc[idx, 'posterior_beta_prob'] = posterior_beta_prob
        combined_df.loc[idx, 'prior_posterior_beta_prob'] = prior_posterior_beta_prob  # NEW
        combined_df.loc[idx, 'category_bootstrap_prob'] = cat_bootstrap_prob
        combined_df.loc[idx, 'limited_bootstrap_prob'] = limited_bootstrap_prob
        combined_df.loc[idx, 'unconditional_bootstrap_prob'] = unconditional_bootstrap_prob
        
        all_results.append(result)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results/all_elections_analysis.csv', index=False)
    print(f"\nSaved detailed results to model_comparison_results/all_elections_analysis.csv")
    
    # Create visualizations
    create_comparative_visualizations(results_df)
    
    # Create Alaska-specific visualizations
    create_alaska_model_comparison(results_df)
    
    # Create NYC-specific visualizations
    create_nyc_model_comparison(results_df)
    
    # Create region-specific heatmaps
    create_region_heatmaps(results_df)
    
    return results_df

def create_comparative_visualizations(results_df):
    """
    Create comprehensive visualizations comparing all probability models
    across all elections.
    
    Args:
        results_df: DataFrame containing results for all elections and models
    """
    os.makedirs('model_comparison_results/figures', exist_ok=True)
    
    # Make sure we have data to plot
    if results_df.empty:
        print("No results available for visualizations.")
        return
    
    # Updated model names per user request
    model_names = {
        'beta_model_prob': 'Gap-Based Beta',
        'posterior_beta_prob': 'Similarity Beta',
        'prior_posterior_beta_prob': 'Prior-Posterior Beta',
        'unconditional_bootstrap_prob': 'Unconditional Bootstrap',
        'category_bootstrap_prob': 'Similarity Bootstrap',
        'limited_bootstrap_prob': 'Rank-Restricted Bootstrap'
    }
    
    # Updated model colors
    model_colors = {
        'beta_model_prob': 'blue',
        'posterior_beta_prob': 'green',
        'prior_posterior_beta_prob': 'magenta',
        'unconditional_bootstrap_prob': 'orange',
        'category_bootstrap_prob': 'red',
        'limited_bootstrap_prob': 'purple'
    }
    
    # Updated model markers
    model_markers = {
        'beta_model_prob': 'o',
        'posterior_beta_prob': '*',
        'prior_posterior_beta_prob': 's',
        'unconditional_bootstrap_prob': 'X',
        'category_bootstrap_prob': '^',
        'limited_bootstrap_prob': 'D'
    }
    
    # Updated model linestyles
    model_linestyles = {
        'beta_model_prob': '-',
        'posterior_beta_prob': '-',
        'prior_posterior_beta_prob': '-',
        'unconditional_bootstrap_prob': ':',
        'category_bootstrap_prob': ':',
        'limited_bootstrap_prob': ':'
    }
    
    # Specify the order of models to plot
    model_order = [
        'beta_model_prob',          # 1. Gap-based beta
        'posterior_beta_prob',      # 2. Similarity beta
        'prior_posterior_beta_prob', # 3. Prior-posterior beta
        'unconditional_bootstrap_prob', # 4. Unconditional bootstrap
        'category_bootstrap_prob',   # 5. Similarity bootstrap
        'limited_bootstrap_prob'     # 6. Rank-restricted bootstrap
    ]
    
    # 1. Comparison of all models across all elections - Scatter plot
    plt.figure(figsize=(14, 10))
    
    # Sort by required preference percentage
    sorted_df = results_df.sort_values('required_preference_pct')
    
    # Add jitter for each model
    n_models = len(model_order)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_order)}
    
    # Plot each model in specified order
    for model in model_order:
        if model in model_names and model in sorted_df.columns:
            x_jittered = sorted_df['required_preference_pct'] + model_jitter[model]
            plt.plot(x_jittered, sorted_df[model],
                    marker=model_markers[model], 
                    linestyle=model_linestyles[model], 
                    color=model_colors[model], 
                    label=model_names[model],
                    markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Probability Models vs Required Preference Percentage', fontsize=16)
    plt.xlabel('Required Preference Percentage', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/all_models_by_preference.png', dpi=300)
    plt.close()
    
    # 2. Model comparison by region - Box plots
    plt.figure(figsize=(16, 10))
    
    # Create a long-form dataframe for easier plotting
    models_df = pd.melt(results_df, 
                       id_vars=['region', 'election_id', 'letter', 'required_preference_pct'],
                       value_vars=model_order,
                       var_name='model', value_name='probability')
    
    # Map model names to more readable versions
    models_df['model'] = models_df['model'].map(lambda x: model_names.get(x, x))
    
    # Create box plot
    sns.boxplot(x='model', y='probability', hue='region', data=models_df)
    
    # Add title and labels
    plt.title('Probability Distribution by Model and Region', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(title='Region', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/probability_by_model_and_region.png', dpi=300)
    plt.close()
    
    # 3. Probability vs Gap to Win % - All models
    plt.figure(figsize=(14, 10))
    
    # Sort by gap to win
    sorted_by_gap = results_df.sort_values('gap_to_win_pct')
    
    # Add jitter for each model
    n_models = len(model_order)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_order)}
    
    # Plot each model in specified order
    for model in model_order:
        if model in model_names and model in sorted_by_gap.columns:
            x_jittered = sorted_by_gap['gap_to_win_pct'] + model_jitter[model]
            plt.plot(x_jittered, sorted_by_gap[model],
                    marker=model_markers[model], 
                    linestyle=model_linestyles[model], 
                    color=model_colors[model], 
                    label=model_names[model],
                    markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Probability Models vs Gap to Win Percentage', fontsize=16)
    plt.xlabel('Gap to Win Percentage', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/all_models_by_gap.png', dpi=300)
    plt.close()
    
    # 4. Probability vs Exhaust % - All models
    plt.figure(figsize=(14, 10))
    
    # Sort by exhaust percent
    sorted_by_exhaust = results_df.sort_values('exhaust_pct')
    
    # Add jitter for each model
    n_models = len(model_order)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_order)}
    
    # Plot each model in specified order
    for model in model_order:
        if model in model_names and model in sorted_by_exhaust.columns:
            x_jittered = sorted_by_exhaust['exhaust_pct'] + model_jitter[model]
            plt.plot(x_jittered, sorted_by_exhaust[model],
                    marker=model_markers[model], 
                    linestyle=model_linestyles[model],
                    color=model_colors[model], 
                    label=model_names[model],
                    markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Probability Models vs Exhaust Percentage', fontsize=16)
    plt.xlabel('Exhaust Percentage', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/all_models_by_exhaust.png', dpi=300)
    plt.close()
    
    # 5. Model agreement heatmap - Correlation between models
    plt.figure(figsize=(12, 10))
    
    # Extract probability columns in order
    prob_columns = [col for col in model_order if col in results_df.columns]
    prob_df = results_df[prob_columns]
    
    # Create correlation matrix
    corr_matrix = prob_df.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
               xticklabels=[model_names[m] for m in corr_matrix.columns],
               yticklabels=[model_names[m] for m in corr_matrix.index])
    
    # Add title
    plt.title('Correlation Between Probability Models', fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/model_correlation_heatmap.png', dpi=300)
    plt.close()
    
    # 6. Strategy-to-Exhaust Ratio vs Probability
    plt.figure(figsize=(14, 10))
    
    # Filter to reasonable ratios for better visualization
    filtered_df = results_df[results_df['strategy_exhaust_ratio'] <= 1]
    
    if not filtered_df.empty:
        # Sort by ratio
        sorted_by_ratio = filtered_df.sort_values('strategy_exhaust_ratio')
        
        # Add jitter for each model
        n_models = len(model_order)
        jitter_vals = np.linspace(-0.01, 0.01, n_models)
        model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_order)}
        
        # Plot each model in specified order
        for model in model_order:
            if model in model_names and model in sorted_by_ratio.columns:
                x_jittered = sorted_by_ratio['strategy_exhaust_ratio'] + model_jitter[model]
                plt.plot(x_jittered, sorted_by_ratio[model],
                        marker=model_markers[model], 
                        linestyle=model_linestyles[model], 
                        color=model_colors[model], 
                        label=model_names[model],
                        markersize=10, linewidth=1.2, alpha=0.6)
        
        # Add reference line
        plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        
        # Add title and labels
        plt.title('Probability Models vs Strategy-to-Exhaust Ratio', fontsize=16)
        plt.xlabel('Strategy/Exhaust Ratio', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('model_comparison_results/figures/all_models_by_ratio.png', dpi=300)
        plt.close()
    
    # 7. Add a new plot for Probability vs Strategy/Exhaust Ratio for all data points
    plt.figure(figsize=(14, 10))
    
    # Sort by ratio but don't filter
    sorted_by_ratio_all = results_df.sort_values('strategy_exhaust_ratio')
    
    # Add jitter for each model
    n_models = len(model_order)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_order)}
    
    # Plot each model in specified order
    for model in model_order:
        if model in model_names and model in sorted_by_ratio_all.columns:
            x_jittered = sorted_by_ratio_all['strategy_exhaust_ratio'] + model_jitter[model]
            plt.plot(x_jittered, sorted_by_ratio_all[model],
                    marker=model_markers[model], 
                    linestyle=model_linestyles[model], 
                    color=model_colors[model], 
                    label=model_names[model],
                    markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Probability Models vs Strategy-to-Exhaust Ratio (All Data Points)', fontsize=16)
    plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/all_models_by_ratio_full.png', dpi=300)
    plt.close()
    
    # 8. Comparing Models - Scatter plots
    fig = plt.figure(figsize=(18, 16))
    
    # Create grid of scatter plots comparing models
    # Focus on bootstrap vs theoretical comparisons
    
    # Define all model comparisons to make
    comparisons = [
        ('beta_model_prob', 'category_bootstrap_prob'),
        ('beta_model_prob', 'limited_bootstrap_prob'),
        ('beta_model_prob', 'unconditional_bootstrap_prob'),
        ('posterior_beta_prob', 'category_bootstrap_prob'),
        ('posterior_beta_prob', 'limited_bootstrap_prob'),
        ('posterior_beta_prob', 'unconditional_bootstrap_prob'),
        ('prior_posterior_beta_prob', 'category_bootstrap_prob'),
        ('prior_posterior_beta_prob', 'limited_bootstrap_prob'),
        ('prior_posterior_beta_prob', 'unconditional_bootstrap_prob')
    ]
    
    # Create a 3x3 grid of subplots
    for i, (x_model, y_model) in enumerate(comparisons):
        plt.subplot(3, 3, i+1)
        plt.scatter(results_df[x_model], results_df[y_model], 
                   alpha=0.7, s=60, c=results_df['required_preference_pct'], cmap='viridis')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # reference line
        plt.colorbar(label='Required Preference %')
        plt.xlabel(model_names[x_model], fontsize=12)
        plt.ylabel(model_names[y_model], fontsize=12)
        plt.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle('Model Comparison: Theoretical vs Bootstrap Methods', fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # adjust for suptitle
    plt.savefig('model_comparison_results/figures/model_comparison_matrix.png', dpi=300)
    plt.close()
    
    # 9. Scatter plot matrix - without color mapping to avoid dimension mismatch
    pd.plotting.scatter_matrix(results_df[[m for m in model_order if m in results_df.columns]], 
                              figsize=(16, 16), diagonal='kde', alpha=0.7)
    plt.suptitle('Scatter Matrix of All Probability Models', fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # adjust for suptitle
    plt.savefig('model_comparison_results/figures/model_scatter_matrix.png', dpi=300)
    plt.close()
    
    print(f"Created comparative visualizations in model_comparison_results/figures/")

def create_alaska_model_comparison(results_df):
    """
    Create Alaska-specific visualizations comparing models
    """
    # Filter for Alaska elections only
    alaska_df = results_df[results_df['region'] == 'Alaska']
    if alaska_df.empty:
        print("No Alaska election data available for visualizations.")
        return
    os.makedirs('model_comparison_results/figures', exist_ok=True)
    model_names = {
        'beta_model_prob': 'Gap-Based Beta',
        'prior_posterior_beta_prob': 'Prior-Posterior Beta',
        'posterior_beta_prob': 'Similarity Beta',
        'category_bootstrap_prob': 'Similarity Bootstrap',
        'unconditional_bootstrap_prob': 'Unconditional Bootstrap'
    }
    model_colors = {
        'beta_model_prob': 'blue',
        'prior_posterior_beta_prob': 'magenta',
        'posterior_beta_prob': 'green',
        'category_bootstrap_prob': 'red',
        'unconditional_bootstrap_prob': 'orange'
    }
    model_markers = {
        'beta_model_prob': 'o',
        'prior_posterior_beta_prob': 's',
        'posterior_beta_prob': '*',
        'category_bootstrap_prob': '^',
        'unconditional_bootstrap_prob': 'X'
    }
    
    # Create figure for models vs. required preference percentage
    plt.figure(figsize=(14, 10))
    
    # Sort by required preference percentage
    sorted_df = alaska_df.sort_values('required_preference_pct')
    
    # Add jitter for each model
    model_keys = list(model_names.keys())
    n_models = len(model_keys)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_keys)}
    
    # Plot each model
    for model, name in model_names.items():
        x_jittered = sorted_df['required_preference_pct'] + model_jitter[model]
        plt.plot(x_jittered, sorted_df[model],
                marker=model_markers[model], linestyle='-', color=model_colors[model], label=name,
                markersize=10, linewidth=2, alpha=0.6)
    
    # Add reference line at 0.5 probability
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Alaska Elections: Model Comparison', fontsize=16)
    plt.xlabel('Required Preference Percentage', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Remove annotation of election IDs
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/alaska_models_comparison.png', dpi=300)
    plt.close()
    
    # Create figure for models vs. gap to win percentage
    plt.figure(figsize=(14, 10))
    
    # Sort by gap to win
    sorted_by_gap = alaska_df.sort_values('gap_to_win_pct')
    
    # Add jitter for each model
    model_keys = list(model_names.keys())
    n_models = len(model_keys)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_keys)}
    
    # Plot each model
    for model, name in model_names.items():
        x_jittered = sorted_by_gap['gap_to_win_pct'] + model_jitter[model]
        plt.plot(x_jittered, sorted_by_gap[model],
                marker=model_markers[model], linestyle='-', color=model_colors[model], label=name,
                markersize=10, linewidth=2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Alaska Elections: Models vs Gap to Win Percentage', fontsize=16)
    plt.xlabel('Gap to Win Percentage', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Remove annotation of election IDs
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/alaska_models_by_gap.png', dpi=300)
    plt.close()
    
    # New plot: Strategy/Exhaust Ratio
    plt.figure(figsize=(14, 10))
    
    # Sort by ratio
    sorted_by_ratio = alaska_df.sort_values('strategy_exhaust_ratio')
    
    # Add jitter for each model
    model_keys = list(model_names.keys())
    n_models = len(model_keys)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_keys)}
    
    # Plot each model
    for model, name in model_names.items():
        x_jittered = sorted_by_ratio['strategy_exhaust_ratio'] + model_jitter[model]
        plt.plot(x_jittered, sorted_by_ratio[model],
                marker=model_markers[model], linestyle='-', color=model_colors[model], label=name,
                markersize=10, linewidth=2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Probability Of Alternate Winners Vs Strategy-Exhaust Ratio', fontsize=16)
    plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/alaska_models_by_ratio.png', dpi=300)
    plt.close()
    
    print("Created Alaska-specific model comparison visualizations in model_comparison_results/figures/")

def create_nyc_model_comparison(results_df):
    """
    Create NYC-specific visualizations comparing all five models
    """
    # Filter for NYC elections only
    nyc_df = results_df[results_df['region'] == 'NYC']
    if nyc_df.empty:
        print("No NYC election data available for visualizations.")
        return
    os.makedirs('model_comparison_results/figures', exist_ok=True)
    
    # Updated model names per user request
    model_names = {
        'beta_model_prob': 'Gap-Based Beta',
        'posterior_beta_prob': 'Similarity Beta',
        'prior_posterior_beta_prob': 'Prior-Posterior Beta',
        'unconditional_bootstrap_prob': 'Unconditional Bootstrap',
        'category_bootstrap_prob': 'Similarity Bootstrap',
        'limited_bootstrap_prob': 'Rank-Restricted Bootstrap'
    }
    
    # Updated model colors
    model_colors = {
        'beta_model_prob': 'blue',
        'posterior_beta_prob': 'green',
        'prior_posterior_beta_prob': 'magenta',
        'unconditional_bootstrap_prob': 'orange',
        'category_bootstrap_prob': 'red',
        'limited_bootstrap_prob': 'purple'
    }
    
    # Updated model markers
    model_markers = {
        'beta_model_prob': 'o',
        'posterior_beta_prob': '*',
        'prior_posterior_beta_prob': 's',
        'unconditional_bootstrap_prob': 'X',
        'category_bootstrap_prob': '^',
        'limited_bootstrap_prob': 'D'
    }
    
    # Updated model linestyles
    model_linestyles = {
        'beta_model_prob': '-',
        'posterior_beta_prob': '-',
        'prior_posterior_beta_prob': '-',
        'unconditional_bootstrap_prob': ':',
        'category_bootstrap_prob': ':',
        'limited_bootstrap_prob': ':'
    }
    
    # Create figure for models vs. strategy/exhaust ratio
    plt.figure(figsize=(14, 10))
    
    # Sort by ratio
    sorted_by_ratio = nyc_df.sort_values('strategy_exhaust_ratio')
    
    # Specify the order of models to plot
    model_order = [
        'beta_model_prob',          # 1. Gap-based beta
        'posterior_beta_prob',      # 2. Similarity beta
        'prior_posterior_beta_prob', # 3. Prior-posterior beta
        'unconditional_bootstrap_prob', # 4. Unconditional bootstrap
        'category_bootstrap_prob',   # 5. Similarity bootstrap
        'limited_bootstrap_prob'     # 6. Rank-restricted bootstrap
    ]
    
    # Add jitter for each model
    n_models = len(model_order)
    jitter_vals = np.linspace(-0.01, 0.01, n_models)
    model_jitter = {k: jitter_vals[i] for i, k in enumerate(model_order)}
    
    # Plot each model in specified order
    for model in model_order:
        if model in model_names and model in sorted_by_ratio.columns:
            x_jittered = sorted_by_ratio['strategy_exhaust_ratio'] + model_jitter[model]
            plt.plot(x_jittered, sorted_by_ratio[model],
                    marker=model_markers[model], 
                    linestyle=model_linestyles[model], 
                    color=model_colors[model], 
                    label=model_names[model],
                    markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Probability Of Alternate Winners Vs Strategy-Exhaust Ratio', fontsize=16)
    plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/nyc_models_by_ratio.png', dpi=300)
    plt.close()
    
    # Create additional figure with scaled beta parameters
    create_nyc_scaled_beta_comparison(nyc_df)
    
    print("Created NYC-specific model comparison visualizations in model_comparison_results/figures/")

def create_nyc_scaled_beta_comparison(nyc_df):
    """
    Create a NYC-specific visualization with scaled beta parameters
    """
    if nyc_df.empty:
        return
        
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Sort by ratio
    sorted_by_ratio = nyc_df.sort_values('strategy_exhaust_ratio')
    
    # Calculate scaled beta probabilities with different scaling factors
    scaling_factors = [1, 5, 10, 20]
    
    # Model names and colors
    model_colors = {
        1: 'darkred',
        5: 'red',
        10: 'orange',
        20: 'gold'
    }
    
    model_markers = {
        1: 'o',
        5: 's',
        10: '^',
        20: 'd'
    }
    
    # For each election in the dataframe
    for idx, row in sorted_by_ratio.iterrows():
        # Extract the required values
        required_proportion = row['required_preference_pct'] / 100
        
        # For each scaling factor
        for scale in scaling_factors:
            # If we have the posterior beta data
            if 'posterior_beta_prob' in row and not pd.isna(row['posterior_beta_prob']):
                # Get the B>A and A>B percentages - extract from diagnostics if available
                # For simplicity, let's estimate from the probability using the required preference
                if row['posterior_beta_prob'] > 0.5:
                    # If probability > 0.5, B>A percentage is higher than required
                    b_over_a_pct = max(required_proportion * 100 + 10, 55)
                    a_over_b_pct = 100 - b_over_a_pct
                else:
                    # If probability < 0.5, B>A percentage is lower than required
                    b_over_a_pct = min(required_proportion * 100 - 10, 45)
                    a_over_b_pct = 100 - b_over_a_pct
                
                # Scale down the parameters
                alpha = (b_over_a_pct / 100) * scale
                beta = (a_over_b_pct / 100) * scale
                
                # Calculate new probability
                scaled_prob = 1 - stats.beta.cdf(required_proportion, alpha, beta)
                
                # Store in the dataframe for this election
                sorted_by_ratio.loc[idx, f'scaled_beta_{scale}'] = scaled_prob
    
    # Add original posterior beta for comparison
    plt.plot(sorted_by_ratio['strategy_exhaust_ratio'], sorted_by_ratio['posterior_beta_prob'],
            marker='*', linestyle='-', color='green', 
            label='Original Similarity Beta',
            markersize=12, linewidth=1.5, alpha=0.8)
    
    # Plot each scaled model
    for scale in scaling_factors:
        col_name = f'scaled_beta_{scale}'
        if col_name in sorted_by_ratio.columns:
            plt.plot(sorted_by_ratio['strategy_exhaust_ratio'], sorted_by_ratio[col_name],
                    marker=model_markers[scale], linestyle='-', color=model_colors[scale], 
                    label=f'Scaled Beta (factor={scale})',
                    markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add bootstrap results for comparison
    plt.plot(sorted_by_ratio['strategy_exhaust_ratio'], sorted_by_ratio['category_bootstrap_prob'],
            marker='^', linestyle=':', color='darkblue', 
            label='Similarity Bootstrap',
            markersize=10, linewidth=1.2, alpha=0.6)
    
    # Add reference line
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('NYC Elections: Scaled Beta Models vs Strategy-to-Exhaust Ratio', fontsize=16)
    plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results/figures/nyc_scaled_beta_comparison.png', dpi=300)
    plt.close()

def create_region_heatmaps(results_df):
    """
    Create heatmap visualizations showing probability by gap and exhaust percentage
    for both NYC and Alaska regions, with exact data points labeled.
    
    Args:
        results_df: DataFrame containing results for all elections and models
    """
    # Create directory for results
    os.makedirs('model_comparison_results/figures', exist_ok=True)
    
    # Define regions to process
    regions = ['NYC', 'Alaska']
    
    # Define the model to use for coloring (posterior beta is a good choice)
    probability_model = 'posterior_beta_prob'
    
    for region in regions:
        # Filter for region elections only
        region_df = results_df[results_df['region'] == region]
        
        if region_df.empty:
            print(f"No {region} election data available for heatmap visualizations.")
            continue
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Create scatter plot with larger point sizes
        scatter = plt.scatter(
            region_df['exhaust_pct'], 
            region_df['gap_to_win_pct'],
            c=region_df[probability_model], 
            cmap='YlOrRd', 
            s=150,
            alpha=0.9,
            edgecolors='black',
            vmin=0.0,
            vmax=0.5
        )
        
        # Add data labels to each point with contrasting outline for better visibility
        for idx, row in region_df.iterrows():
            # Determine text color based on data point color for better readability
            if row[probability_model] > 0.3:
                text_color = 'black'
            else:
                text_color = 'white'
                
            # Add percentage label with outline
            plt.annotate(
                f"{row[probability_model]:.1%}",
                (row['exhaust_pct'], row['gap_to_win_pct']),
                color=text_color,
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.7)
            )
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, format='%.1f')
        cbar.set_label('Probability of Candidate B Winning', rotation=270, labelpad=20, fontsize=12)
        
        # Add title and labels with enhanced styling
        plt.title(f'{region}: Probability by Gap and Exhaust', fontsize=18, pad=20)
        plt.xlabel('Exhausted Ballot Percentage', fontsize=14)
        plt.ylabel('Gap Needed for Candidate B to Win (%)', fontsize=14)
        
        # Add improved grid and axes
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Adjust axis limits slightly to provide some padding
        x_min = max(0, region_df['exhaust_pct'].min() - 2)
        x_max = region_df['exhaust_pct'].max() + 2
        y_min = max(0, region_df['gap_to_win_pct'].min() - 0.5)
        y_max = region_df['gap_to_win_pct'].max() + 0.5
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Add contour lines for probability regions (if sufficient data points)
        if len(region_df) >= 5:
            try:
                from scipy.interpolate import griddata
                
                # Create a meshgrid for contour plotting
                x_range = np.linspace(x_min, x_max, 100)
                y_range = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_range, y_range)
                
                # Interpolate values for contour
                points = np.column_stack((region_df['exhaust_pct'], region_df['gap_to_win_pct']))
                Z = griddata(points, region_df[probability_model], (X, Y), method='linear')
                
                # Draw contour lines
                contour = plt.contour(X, Y, Z, levels=[0.1, 0.2, 0.3, 0.4, 0.5], 
                                     colors='black', alpha=0.5, linestyles='dotted')
                plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            except:
                print(f"Could not create contour lines for {region} (likely insufficient data points).")
        
        # Save the figure with higher resolution
        plt.tight_layout()
        plt.savefig(f'model_comparison_results/figures/{region.lower()}_probability_heatmap.png', dpi=400)
        plt.close()
    
    print("Created enhanced region-specific probability heatmaps in model_comparison_results/figures/")

def prior_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots, gap_to_win_pct):
    """Calculate probability using a Beta prior (from beta_parameters) updated with observed data (B>A, A>B completions).
    This is a Bayesian update: posterior = Beta(a + B>A, b + A>B).
    """
    # Get prior parameters (already in percentage scale)
    a_prior, b_prior = beta_parameters(gap_to_win_pct)

    # Categorize exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in exhausted_ballots.items():
        if not ballot:
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count

    # Identify non-exhausted ballots that express preferences for A and/or B
    # group them by first preference to match the exhausted ballot categories
    complete_by_first_pref = {}
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
        has_a = 'A' in ballot
        has_b = 'B' in ballot
        if has_a and has_b:
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {'b_over_a': 0, 'a_over_b': 0, 'total': 0}
            complete_by_first_pref[first_pref]['total'] += count
            if ballot.index('B') < ballot.index('A'):
                complete_by_first_pref[first_pref]['b_over_a'] += count
            else:
                complete_by_first_pref[first_pref]['a_over_b'] += count

    # Aggregate expected completions for B>A and A>B
    total_expected_b_over_a = 0
    total_expected_a_over_b = 0
    for first_pref, count in exh_by_first_pref.items():
        if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
            category_data = complete_by_first_pref[first_pref]
            prob_b_over_a = category_data['b_over_a'] / category_data['total']
            total_expected_b_over_a += count * prob_b_over_a
            total_expected_a_over_b += count * (1 - prob_b_over_a)
        else:
            total_complete = sum(data['total'] for data in complete_by_first_pref.values())
            total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
            if total_complete > 0:
                overall_prob_b_over_a = total_b_over_a / total_complete
                total_expected_b_over_a += count * overall_prob_b_over_a
                total_expected_a_over_b += count * (1 - overall_prob_b_over_a)

    # Calculate total expected completions
    total_expected = total_expected_b_over_a + total_expected_a_over_b
    
    if total_expected > 0:
        # Convert raw counts to percentages (0-100 scale)
        b_over_a_pct = 100 * total_expected_b_over_a / total_expected
        a_over_b_pct = 100 * total_expected_a_over_b / total_expected
        
        # Since our prior parameters (a_prior, b_prior) are already in percentage scale,
        # we should directly combine them with the observed percentages
        weight_prior = 1.0  # Weight for prior (can be adjusted)
        weight_data = 1.0   # Weight for observed data (can be adjusted)
        
        # Weighted combination of prior and observed percentages
        a_post = (weight_prior * a_prior + weight_data * b_over_a_pct) / (weight_prior + weight_data)
        b_post = (weight_prior * b_prior + weight_data * a_over_b_pct) / (weight_prior + weight_data)
    else:
        # If no data, just use the prior
        a_post = a_prior
        b_post = b_prior

    # Required preference is already in percentage (0-100 scale)
    required_proportion = required_preference_pct / 100
    
    # For Beta CDF, we need to convert parameters to standard Beta parameters (proportion scale)
    alpha = a_post
    beta = b_post
    
    # Calculate probability using Beta CDF
    probability = 1 - stats.beta.cdf(required_proportion, alpha, beta)

    print(f"[Prior-Posterior Beta] Prior: a={a_prior:.2f}, b={b_prior:.2f}; " +
          f"Observed: B>A={b_over_a_pct:.2f}%, A>B={a_over_b_pct:.2f}%; " +
          f"Posterior: a={a_post:.2f}, b={b_post:.2f}; " +
          f"Required: {required_preference_pct:.2f}%; Probability: {probability:.4f}")
    
    return probability


# =============================================================================
# MULTI-WINNER STV PROBABILITY MODELS
# =============================================================================
# These functions are adapted for multi-winner STV elections where we compare
# a candidate against ALL other active candidates (not just A vs B).
# Ported from case_studies/portland/strategy_analysis.py which produced the
# paper's Table 6 (Portland exhaustion analysis).

def analyze_preference_patterns_multi_winner(ballot_counts, exhausted_ballots, candidate_letter, active_candidates):
    """
    Analyze preference patterns for candidate vs all other active candidates,
    categorized by first preference (similar to single-winner RCV analysis).

    For multi-winner STV, we compare candidate against ALL other active candidates,
    not just one winner. A ballot "prefers" the candidate if it ranks them highest
    among all active candidates.

    Args:
        ballot_counts (dict): All ballot counts
        exhausted_ballots (dict): Exhausted ballot counts
        candidate_letter (str): The candidate we're analyzing
        active_candidates (set): Set of all active candidates at elimination

    Returns:
        dict: Preference analysis results with:
            - exhausted_by_first_pref: dict mapping first pref to exhausted count
            - complete_by_first_pref: dict mapping first pref to {candidate_first, others_first, total}
            - total_exhausted: total exhausted ballot count
    """
    # Categorize exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count

    # Find non-exhausted ballots that rank at least one active candidate
    # and categorize by first preference
    complete_by_first_pref = {}

    for ballot, count in ballot_counts.items():
        if not ballot:
            continue

        # Check if ballot ranks at least one active candidate
        ballot_active_candidates = [c for c in ballot if c in active_candidates]

        if ballot_active_candidates:  # Ballot ranks at least one active candidate
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {
                    'candidate_first': 0,  # candidate_letter ranked highest among active
                    'others_first': 0,     # other active candidates ranked highest
                    'total': 0
                }

            complete_by_first_pref[first_pref]['total'] += count

            # Determine if candidate_letter or others are ranked highest among active candidates
            if candidate_letter in ballot_active_candidates:
                # Find position of candidate_letter among active candidates in this ballot
                candidate_pos = ballot.index(candidate_letter)
                other_active_positions = [ballot.index(c) for c in ballot_active_candidates if c != candidate_letter]

                if not other_active_positions or candidate_pos < min(other_active_positions):
                    # candidate_letter is ranked highest among active candidates
                    complete_by_first_pref[first_pref]['candidate_first'] += count
                else:
                    # Some other active candidate is ranked higher
                    complete_by_first_pref[first_pref]['others_first'] += count
            else:
                # candidate_letter not ranked, so others win by default
                complete_by_first_pref[first_pref]['others_first'] += count

    return {
        'exhausted_by_first_pref': exh_by_first_pref,
        'complete_by_first_pref': complete_by_first_pref,
        'total_exhausted': sum(exhausted_ballots.values())
    }


def beta_probability_multi_winner(required_preference_pct, gap_to_win_pct):
    """
    Calculate probability using Beta distribution for multi-winner STV.
    Uses same parameters as single-winner analysis: α + β = 100 (percentage scale).
    """
    # Ensure required_preference_pct is within bounds
    required_preference_pct = max(0, min(required_preference_pct, 100))

    # Convert from percentage to proportion for Beta CDF
    required_proportion = required_preference_pct / 100

    # Use same beta parameters as single-winner analysis (percentage scale)
    base_param = 50.0
    max_shift = min(gap_to_win_pct * 0.5, 40)
    a = max(base_param - max_shift, 10.0)  # Parameter for others (like B/challenger in single-winner)
    b = max(base_param + max_shift, 10.0)  # Parameter for candidate (like A/leader in single-winner)

    # Calculate probability
    probability = 1 - stats.beta.cdf(required_proportion, a, b)

    return probability


def similarity_beta_multi_winner(preference_analysis, candidate_letter, active_candidates,
                                required_preference_pct, gap_to_win_pct):
    """
    Similarity Beta model for multi-winner STV.
    Uses observed preference patterns to set Beta parameters directly (α + β = 100).
    """
    exh_by_first_pref = preference_analysis['exhausted_by_first_pref']
    complete_by_first_pref = preference_analysis['complete_by_first_pref']
    total_exhausted = preference_analysis['total_exhausted']

    if total_exhausted == 0:
        return 0.5

    # Calculate expected completions based on observed patterns
    total_expected_candidate = 0
    total_expected_others = 0

    for first_pref, count in exh_by_first_pref.items():
        if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
            category_data = complete_by_first_pref[first_pref]
            prob_candidate_first = category_data['candidate_first'] / category_data['total']

            expected_candidate = count * prob_candidate_first
            expected_others = count * (1 - prob_candidate_first)

            total_expected_candidate += expected_candidate
            total_expected_others += expected_others
        else:
            # Use overall distribution if no category data
            total_complete = sum(data['total'] for data in complete_by_first_pref.values())
            total_candidate_first = sum(data['candidate_first'] for data in complete_by_first_pref.values())

            if total_complete > 0:
                overall_prob = total_candidate_first / total_complete
                expected_candidate = count * overall_prob
                expected_others = count * (1 - overall_prob)

                total_expected_candidate += expected_candidate
                total_expected_others += expected_others

    # Calculate preference percentages from expected completions (0-100 scale)
    total_completions = total_expected_candidate + total_expected_others
    if total_completions > 0:
        candidate_pct = 100 * total_expected_candidate / total_completions
        others_pct = 100 * total_expected_others / total_completions

        # Use these percentages directly as Beta parameters
        alpha = candidate_pct
        beta_param = others_pct

        # Calculate probability
        required_proportion = required_preference_pct / 100
        probability = 1 - stats.beta.cdf(required_proportion, alpha, beta_param)
    else:
        probability = 0.5

    return probability


def prior_posterior_beta_multi_winner(preference_analysis, candidate_letter, active_candidates,
                                    required_preference_pct, gap_to_win_pct):
    """
    Prior-Posterior Beta model for multi-winner STV.
    Combines prior beliefs (gap-based) with observed evidence.
    """
    exh_by_first_pref = preference_analysis['exhausted_by_first_pref']
    complete_by_first_pref = preference_analysis['complete_by_first_pref']
    total_exhausted = preference_analysis['total_exhausted']

    if total_exhausted == 0:
        return 0.5

    # Prior parameters based on gap (percentage scale, α + β = 100)
    base_param = 50.0
    max_shift = min(gap_to_win_pct * 0.5, 40)
    a_prior = max(base_param - max_shift, 10.0)
    b_prior = max(base_param + max_shift, 10.0)

    # Calculate observed evidence (expected completions)
    total_expected_candidate = 0
    total_expected_others = 0

    for first_pref, count in exh_by_first_pref.items():
        if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
            category_data = complete_by_first_pref[first_pref]
            prob_candidate_first = category_data['candidate_first'] / category_data['total']

            total_expected_candidate += count * prob_candidate_first
            total_expected_others += count * (1 - prob_candidate_first)
        else:
            total_complete = sum(data['total'] for data in complete_by_first_pref.values())
            total_candidate_first = sum(data['candidate_first'] for data in complete_by_first_pref.values())

            if total_complete > 0:
                overall_prob = total_candidate_first / total_complete
                total_expected_candidate += count * overall_prob
                total_expected_others += count * (1 - overall_prob)

    # Convert observed evidence to percentages
    total_expected = total_expected_candidate + total_expected_others
    if total_expected > 0:
        candidate_over_others_pct = 100 * total_expected_candidate / total_expected
        others_over_candidate_pct = 100 * total_expected_others / total_expected

        # Weighted combination
        weight_prior = 1.0
        weight_data = 1.0

        a_post = (weight_prior * a_prior + weight_data * candidate_over_others_pct) / (weight_prior + weight_data)
        b_post = (weight_prior * b_prior + weight_data * others_over_candidate_pct) / (weight_prior + weight_data)
    else:
        a_post = a_prior
        b_post = b_prior

    # Calculate probability
    required_proportion = required_preference_pct / 100
    probability = 1 - stats.beta.cdf(required_proportion, a_post, b_post)

    return probability


def category_bootstrap_multi_winner(preference_analysis, candidate_letter, active_candidates,
                                  required_preference_pct, gap_to_win_pct, n_bootstrap=1000):
    """
    Category-based bootstrap for multi-winner STV.
    Samples completions based on first-preference category patterns.
    """
    exh_by_first_pref = preference_analysis['exhausted_by_first_pref']
    complete_by_first_pref = preference_analysis['complete_by_first_pref']
    total_exhausted = preference_analysis['total_exhausted']

    if total_exhausted == 0:
        return 0.5, (0.5, 0.5)

    # Calculate votes needed for candidate to win (net advantage needed)
    votes_needed = int((required_preference_pct - 50) * total_exhausted / 100)

    # Run bootstrap iterations
    candidate_win_counts = 0

    for i in range(n_bootstrap):
        net_votes_for_candidate = 0

        # Process each category of exhausted ballots
        for first_pref, count in exh_by_first_pref.items():
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                category_data = complete_by_first_pref[first_pref]
                prob_candidate_first = category_data['candidate_first'] / category_data['total']

                # Sample candidate vs others preferences for this category
                candidate_completions = np.random.binomial(count, prob_candidate_first)
                others_completions = count - candidate_completions
                net_votes_for_candidate += (candidate_completions - others_completions)
            else:
                # Use overall distribution if no category data
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_candidate_first = sum(data['candidate_first'] for data in complete_by_first_pref.values())

                if total_complete > 0:
                    overall_prob = total_candidate_first / total_complete
                    candidate_completions = np.random.binomial(count, overall_prob)
                    others_completions = count - candidate_completions
                    net_votes_for_candidate += (candidate_completions - others_completions)

        # Check if candidate wins
        if net_votes_for_candidate >= votes_needed:
            candidate_win_counts += 1

    # Calculate probability and confidence interval
    win_probability = candidate_win_counts / n_bootstrap
    se = np.sqrt((win_probability * (1 - win_probability)) / n_bootstrap)
    ci_lower = max(0, win_probability - 1.96 * se)
    ci_upper = min(1, win_probability + 1.96 * se)

    return win_probability, (ci_lower, ci_upper)


def unconditional_bootstrap_multi_winner(ballot_counts, exhausted_ballots, candidate_letter, active_candidates,
                                        required_preference_pct, gap_to_win_pct, n_bootstrap=1000, max_rankings=6):
    """
    Unconditional Bootstrap model for multi-winner STV.
    Samples from ALL ballots that rank any active candidate.
    """
    total_exhausted = sum(exhausted_ballots.values())
    if total_exhausted == 0:
        return 0.5, (0.5, 0.5)

    # Calculate votes needed for candidate to win
    votes_needed = int((required_preference_pct - 50) * total_exhausted / 100)

    # Find all ballots that rank any active candidate
    candidate_first_total = 0
    others_first_total = 0

    for ballot, count in ballot_counts.items():
        if ballot and any(c in active_candidates for c in ballot):
            # Determine if candidate or others are ranked highest among active candidates
            ballot_active_candidates = [c for c in ballot if c in active_candidates]

            if candidate_letter in ballot_active_candidates:
                candidate_pos = ballot.index(candidate_letter)
                other_active_positions = [ballot.index(c) for c in ballot_active_candidates if c != candidate_letter]

                if not other_active_positions or candidate_pos < min(other_active_positions):
                    candidate_first_total += count
                else:
                    others_first_total += count
            else:
                others_first_total += count

    if candidate_first_total + others_first_total == 0:
        return 0.5, (0.5, 0.5)

    # Calculate overall probability that candidate is preferred
    overall_prob_candidate_first = candidate_first_total / (candidate_first_total + others_first_total)

    # Run bootstrap iterations
    candidate_win_counts = 0

    for i in range(n_bootstrap):
        candidate_completions = np.random.binomial(total_exhausted, overall_prob_candidate_first)
        others_completions = total_exhausted - candidate_completions
        net_votes_for_candidate = candidate_completions - others_completions

        if net_votes_for_candidate >= votes_needed:
            candidate_win_counts += 1

    # Calculate probability and confidence interval
    win_probability = candidate_win_counts / n_bootstrap
    se = np.sqrt((win_probability * (1 - win_probability)) / n_bootstrap)
    ci_lower = max(0, win_probability - 1.96 * se)
    ci_upper = min(1, win_probability + 1.96 * se)

    return win_probability, (ci_lower, ci_upper)


def limited_ranking_bootstrap_multi_winner(ballot_counts, exhausted_ballots, candidate_letter, active_candidates,
                                          required_preference_pct, gap_to_win_pct, n_bootstrap=1000, max_rankings=6):
    """
    Rank-Restricted Bootstrap for multi-winner STV.
    Only considers partial ballots (with room for more rankings).
    """
    total_exhausted = sum(exhausted_ballots.values())
    if total_exhausted == 0:
        return 0.5, (0.5, 0.5)

    # Count partial exhausted ballots (fewer than max_rankings)
    partial_exhausted = sum(count for ballot, count in exhausted_ballots.items()
                           if len(ballot) < max_rankings)

    if partial_exhausted == 0:
        return 0.5, (0.5, 0.5)

    # Calculate votes needed
    votes_needed = int((required_preference_pct - 50) * partial_exhausted / 100)

    # Find ballots that rank any active candidate (for sampling distribution)
    candidate_first_total = 0
    others_first_total = 0

    for ballot, count in ballot_counts.items():
        if ballot and any(c in active_candidates for c in ballot):
            ballot_active_candidates = [c for c in ballot if c in active_candidates]

            if candidate_letter in ballot_active_candidates:
                candidate_pos = ballot.index(candidate_letter)
                other_active_positions = [ballot.index(c) for c in ballot_active_candidates if c != candidate_letter]

                if not other_active_positions or candidate_pos < min(other_active_positions):
                    candidate_first_total += count
                else:
                    others_first_total += count
            else:
                others_first_total += count

    if candidate_first_total + others_first_total == 0:
        return 0.5, (0.5, 0.5)

    overall_prob_candidate_first = candidate_first_total / (candidate_first_total + others_first_total)

    # Run bootstrap iterations
    candidate_win_counts = 0

    for i in range(n_bootstrap):
        candidate_completions = np.random.binomial(partial_exhausted, overall_prob_candidate_first)
        others_completions = partial_exhausted - candidate_completions
        net_votes_for_candidate = candidate_completions - others_completions

        if net_votes_for_candidate >= votes_needed:
            candidate_win_counts += 1

    win_probability = candidate_win_counts / n_bootstrap
    se = np.sqrt((win_probability * (1 - win_probability)) / n_bootstrap)
    ci_lower = max(0, win_probability - 1.96 * se)
    ci_upper = min(1, win_probability + 1.96 * se)

    return win_probability, (ci_lower, ci_upper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze RCV elections to compare exhaust and strategy.')
    parser.add_argument('--process-data', action='store_true', help='Process election data')
    parser.add_argument('--analyze-nyc', action='store_true', help='Analyze NYC elections')
    parser.add_argument('--max-elections', type=int, default=None, help='Maximum number of elections to analyze')
    parser.add_argument('--bootstrap-iters', type=int, default=500, help='Number of bootstrap iterations')
    parser.add_argument('--run-models', action='store_true', help='Run theoretical and Bayesian models')
    parser.add_argument('--analyze-election', type=str, default=None, help='Analyze a specific election by ID')
    parser.add_argument('--letter', type=str, default=None, help='Candidate letter (for specific election analysis)')
    parser.add_argument('--compare-all-models', action='store_true', help='Run comprehensive comparison of all models across all elections')
    
    args = parser.parse_args()
    
    if args.process_data:
        print("Processing election data...")
        nyc_df = process_election_data()
    
    if args.analyze_nyc:
        print(f"Analyzing NYC elections (max={args.max_elections}, bootstrap_iters={args.bootstrap_iters}, run_models={args.run_models})...")
        results_df = analyze_nyc_elections(max_elections=args.max_elections, 
                                          bootstrap_iters=args.bootstrap_iters, 
                                          run_models=args.run_models)
        create_bootstrap_comparison_plots(results_df)
    
    if args.analyze_election:
        print(f"Analyzing specific election: {args.analyze_election}")
        result = analyze_specific_election(args.analyze_election, 
                                          letter=args.letter,
                                          bootstrap_iters=args.bootstrap_iters,
                                          run_models=args.run_models)
    
    if args.compare_all_models:
        print("Running comprehensive comparison of all models across all elections...")
        analyze_all_elections(bootstrap_iters=args.bootstrap_iters)