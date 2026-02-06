from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

#######################
# Core Analysis Functions
#######################

def analyze_voting_combinations(data_samples):
    """
    Analyzes voting data to track winning combinations and additional votes needed.
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample
                         with combinations as keys and (value, details) as values.
    
    Returns:
    tuple: A tuple containing:
           - winning_combinations_frequency: Dict tracking how often each combination wins outright
           - additional_votes_needed: Dict tracking how many additional votes are needed for each combination
           - vote_addition_frequency: Dict tracking how often votes are added to each candidate
    """
    # Initialize dictionaries for analysis
    winning_combinations_frequency = defaultdict(int)
    additional_votes_needed = defaultdict(lambda: defaultdict(int))
    vote_addition_frequency = defaultdict(lambda: defaultdict(int))

    # Analyze each sample
    for sample in data_samples:
        for combination, (value, details) in sample.items():
            # Winning combinations
            if value == 0:
                winning_combinations_frequency[combination] += 1
            else:
                # Additional votes needed
                additional_votes_needed[combination][value] += 1
                if isinstance(details, dict):
                    for candidate, votes in details.items():
                        # Frequency of vote addition to each candidate
                        vote_addition_frequency[combination][candidate] += 1

    # Convert defaultdict to regular dict for cleaner output
    winning_combinations_frequency = dict(winning_combinations_frequency)
    additional_votes_needed = {k: dict(v) for k, v in additional_votes_needed.items()}
    vote_addition_frequency = {k: dict(v) for k, v in vote_addition_frequency.items()}

    return winning_combinations_frequency, additional_votes_needed, vote_addition_frequency


def analyze_detailed_vote_additions(data_samples):
    """
    Analyzes voting data to track detailed vote addition patterns including 
    single, multi-ranked candidates and combinations.
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample
                         with combinations as keys and (value, details) as values.
    
    Returns:
    dict: A dictionary with combinations as keys and detailed vote addition frequency information
    """
    # Initialize dictionary for detailed vote addition frequency
    detailed_vote_addition_frequency = defaultdict(lambda: defaultdict(int))

    # Analyze each sample in data_samples
    for sample in data_samples:
        for combination, (value, details) in sample.items():
            if value > 0 and isinstance(details, dict):
                # Count single candidate additions
                for candidate in details:
                    if len(candidate) == 1:  # Single candidate
                        detailed_vote_addition_frequency[combination][f"Single-ranked: {candidate}"] += 1
                    else:  # Multi-ranked
                        detailed_vote_addition_frequency[combination][f"Multi-ranked: {candidate}"] += 1

                # Count combinations of separate additions
                if len(details) > 1:
                    separate_candidates = [candidate for candidate in details if len(candidate) == 1]
                    for num in range(2, len(separate_candidates) + 1):
                        for combo in combinations(separate_candidates, num):
                            combo_key = " & ".join(sorted(combo))
                            detailed_vote_addition_frequency[combination][f"Combination: {combo_key}"] += 1

    # Convert defaultdict to regular dict for cleaner output
    formatted_detailed_vote_addition_frequency = {k: dict(v) for k, v in detailed_vote_addition_frequency.items()}

    return formatted_detailed_vote_addition_frequency


def calculate_winning_frequency(data_samples):
    """
    Tracks how often each candidate wins outright (with 0 additional votes).
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample
                         with candidates as keys and (additional_votes, details) as values.
    
    Returns:
    Counter: A Counter object tracking the frequency of outright wins for each candidate
    """
    # Initialize a counter to track the frequency of candidates winning with 0 additions
    winning_frequency = Counter()

    # Iterate through each entry in the data
    for entry in data_samples:
        for candidate, values in entry.items():
            # Check if the additional votes required is 0
            if values[0] == 0:
                # Increment the count for this candidate
                winning_frequency[candidate] += 1

    return winning_frequency


#######################
# Statistical Analysis Functions
#######################

def calculate_vote_addition_statistics(data_samples, total_votes, winning_frequency, algo_works, budget=None):
    """
    Calculates detailed statistics about vote additions required for candidates to win.
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample.
    total_votes (int): The total number of votes in the election.
    winning_frequency (Counter): Counter object with frequencies of outright wins.
    algo_works (int): The total number of valid algorithm runs.
    budget (int, optional): The budget for vote additions.
    
    Returns:
    dict: A dictionary containing various statistics about vote additions:
        - std_dev_additions: Standard deviation of additions as percentages
        - win_hook_or_crook: Percentage of wins by any means (additions or outright)
        - average_additions_percentage: Average percentage of votes added
        - min_additions_percentage: Minimum percentage of votes added
        - max_additions_percentage: Maximum percentage of votes added
    """
    # Initialize dictionaries to track the total additions and the count of instances for each combination
    total_additions = Counter()
    win_with_additions = Counter()
    min_additions = {}
    max_additions = {}

    # Iterate through each entry in the data
    for entry in data_samples:
        for candidate, values in entry.items():
            additions, details = values
            # Check if the combination is not winning (additions not 0)
            if additions != 0:
                # Sum the additions made for each candidate within the combination
                total_additions_for_candidate = sum(details.values())
                # Update the total additions and count for this combination
                total_additions[candidate] += total_additions_for_candidate
                win_with_additions[candidate] += 1
                # Track minimum and maximum additions for each candidate
                if candidate not in min_additions:
                    min_additions[candidate] = total_additions_for_candidate
                    max_additions[candidate] = total_additions_for_candidate
                else:
                    min_additions[candidate] = min(min_additions[candidate], total_additions_for_candidate)
                    max_additions[candidate] = max(max_additions[candidate], total_additions_for_candidate)

    # Calculate the average additions for each combination
    average_additions = {candidate: total_additions[candidate] / win_with_additions[candidate] 
                         for candidate in total_additions}
    
    # Calculate the standard deviation of additions for each combination
    std_dev_additions = {}
    for candidate in total_additions:
        additions_list = []
        for entry in data_samples:
            if candidate in entry:
                additions, details = entry[candidate]
                if additions != 0:
                    additions_list.append(sum(details.values()))
        std_dev_additions[candidate] = np.std(additions_list)

    # Calculate all percentages and statistics
    percent_std_dev_additions = {candidate: round(value/total_votes*100, 3) 
                                for candidate, value in std_dev_additions.items()}
    
    win_hook_or_crook = {candidate: round((win_with_additions[candidate]+winning_frequency[candidate])
                                       /algo_works*100, 3) for candidate in win_with_additions}

    average_additions_percentage = {candidate: round(total_additions[candidate]*100 / (total_votes*win_with_additions[candidate]), 3)
                         for candidate in total_additions}
                         
    min_additions_percentage = {candidate: round(min_additions[candidate]*100 / total_votes, 3) 
                              for candidate in min_additions}
                              
    max_additions_percentage = {candidate: round(max_additions[candidate]*100 / total_votes, 3) 
                              for candidate in max_additions}

    results = {
        'std_dev_additions': percent_std_dev_additions,
        'win_hook_or_crook': win_hook_or_crook,
        'average_additions_percentage': average_additions_percentage,
        'min_additions_percentage': min_additions_percentage,
        'max_additions_percentage': max_additions_percentage,
        'win_with_additions': dict(win_with_additions),
        'winning_frequency': dict(winning_frequency),
        'total_additions': dict(total_additions),
        'average_additions': average_additions
    }
    
    # Add budget calculation if provided
    if budget is not None:
        results['budget_calculation'] = (budget + 1) / 8
        
    return results


def analyze_addition_distribution(data_samples):
    """
    Analyzes the distribution of different types of vote additions for each candidate.
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample.
    
    Returns:
    dict: A dictionary containing the percentage distribution of different addition types
          within each category for each candidate.
    """
    # Initialize the dictionary to track the frequencies of each unique type of addition
    addition_frequencies = defaultdict(Counter)

    # Function to extract individual candidates from the addition types
    def extract_candidates(addition_types):
        candidates = set()
        for addition in addition_types:
            for candidate in addition:
                candidates.add(candidate)
        return ''.join(sorted(candidates))

    # Iterate through each entry in the data
    for entry in data_samples:
        for candidate, values in entry.items():
            additions, details = values
            # Check if the combination is not winning (additions not 0)
            if additions != 0:
                # Extract individual candidates from the addition types
                addition_category = extract_candidates(details.keys())
                # Increment the frequency for this category of addition
                addition_frequencies[candidate][addition_category] += 1

    # Calculate the within-category distribution of additions
    within_category_distribution = defaultdict(lambda: defaultdict(Counter))

    for candidate, categories in addition_frequencies.items():
        for category, count in categories.items():
            for entry in data_samples:
                if candidate in entry:
                    additions, details = entry[candidate]
                    if additions != 0:
                        addition_category = extract_candidates(details.keys())
                        if addition_category == category:
                            for addition_type, votes in details.items():
                                within_category_distribution[candidate][category][addition_type] += votes

    # Convert counts to percentages within each category
    within_category_percentage_distribution = defaultdict(lambda: defaultdict(dict))

    for candidate, categories in within_category_distribution.items():
        for category, additions in categories.items():
            total_votes = sum(additions.values())
            for addition_type, votes in additions.items():
                within_category_percentage_distribution[candidate][category][addition_type] = (votes / total_votes) * 100

    # Convert defaultdict to regular dict for better readability
    formatted_distribution = {
        k: {
            sub_k: {
                inner_k: round(inner_v, 2) for inner_k, inner_v in sub_v.items()
            } for sub_k, sub_v in v.items()
        } for k, v in within_category_percentage_distribution.items()
    }
    
    return formatted_distribution, dict(addition_frequencies)


#######################
# Deviation Analysis Functions
#######################

def average_non_zero_deviations(data_samples):
    """
    Calculate the average non-zero deviations for each dictionary in data_samples.
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample
                         with key-value pairs where values are tuples with deviation as the first element.
    
    Returns:
    list: A list containing the average non-zero deviation for each dictionary in data_samples.
          Returns 0 if there are no non-zero deviations in a dictionary.
    """
    averages = []

    for dict_item in data_samples:
        total = 0
        count = 0
        for key, value in dict_item.items():
            deviation = value[0]
            if deviation != 0:
                total += deviation
                count += 1
        average = total / count if count != 0 else 0
        averages.append(average)

    return averages


def plot_deviation_histogram(data_samples):
    """
    Calculate average non-zero deviations and plot them as a histogram.
    
    Parameters:
    data_samples (list): A list of dictionaries containing samples to analyze.
    
    Returns:
    tuple: A tuple containing:
           - average_deviations (list): The calculated average deviations
           - zero_count (int): Number of zeros in the average deviations
           - figure: The matplotlib figure object for further customization
    """
    # Calculate the averages
    average_deviations = average_non_zero_deviations(data_samples)
    
    # Count zeros
    zero_count = average_deviations.count(0)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(average_deviations, bins=20, edgecolor='black')
    ax.set_title('Distribution of Average Non-Zero Deviations')
    ax.set_xlabel('Average Deviation')
    ax.set_ylabel('Frequency')
    
    # Add annotation for zero count
    ax.annotate(f'Number of zeros: {zero_count}', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add annotation for total count
    ax.annotate(f'Total samples: {len(average_deviations)}', 
                xy=(0.05, 0.89), 
                xycoords='axes fraction',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    return average_deviations, zero_count, fig


#######################
# Reporting Functions
#######################

def print_detailed_results(results, total_votes):
    """
    Prints detailed analysis results in the same format as the original code.
    
    Parameters:
    results (dict): The comprehensive results dictionary from comprehensive_voting_analysis
    total_votes (int): Total number of votes in the election
    """
    print("\n=== DETAILED ANALYSIS RESULTS ===\n")
    
    # Print winning frequency
    print("Winning frequency:")
    for candidate, freq in results['winning_frequency'].items():
        print(f"  {candidate}: {freq}")
    print()
    
    # Print standard deviation of additions
    print("Standard deviation of additions:")
    for candidate, value in results['addition_statistics']['std_dev_additions'].items():
        print(f"  {candidate}: {value}%")
    print()
    
    # Print win using additions
    print("Win using additions:")
    for candidate, count in results['addition_statistics']['win_with_additions'].items():
        print(f"  {candidate}: {count}")
    print()
    
    # Print win by hook or crook
    print("Win by hook or crook:")
    for candidate, percentage in results['addition_statistics']['win_hook_or_crook'].items():
        print(f"  {candidate}: {percentage}%")
    print()
    
    # Print average additions percentage
    print("Average additions percentage:")
    for candidate, percentage in results['addition_statistics']['average_additions_percentage'].items():
        print(f"  {candidate}: {percentage}%")
    print()
    
    # Print min additions percentage
    print("Minimum additions percentage:")
    for candidate, percentage in results['addition_statistics']['min_additions_percentage'].items():
        print(f"  {candidate}: {percentage}%")
    print()
    
    # Print max additions percentage
    print("Maximum additions percentage:")
    for candidate, percentage in results['addition_statistics']['max_additions_percentage'].items():
        print(f"  {candidate}: {percentage}%")
    print()
    
    # Print deviation analysis
    print("Deviation analysis:")
    print(f"  Total samples: {results['deviation_analysis']['total_samples']}")
    print(f"  Number of zeros: {results['deviation_analysis']['zero_count']}")
    percentage_zeros = (results['deviation_analysis']['zero_count'] / 
                        results['deviation_analysis']['total_samples']) * 100
    print(f"  Percentage of zeros: {percentage_zeros:.2f}%")
    print()
    
    # Print addition distribution summary
    print("Addition distribution summary:")
    for candidate, categories in results['addition_categories'].items():
        print(f"  {candidate}:")
        for category, count in categories.items():
            print(f"    Category '{category}': {count} occurrences")
    print()
    
    # Print detailed addition percentage distribution
    print("Detailed addition percentage distribution:")
    for candidate, categories in results['addition_distribution'].items():
        print(f"  {candidate}:")
        for category, additions in categories.items():
            print(f"    Category '{category}':")
            for addition_type, percentage in additions.items():
                print(f"      {addition_type}: {percentage}%")
    print()
    
    if 'budget_calculation' in results['addition_statistics']:
        print(f"Budget calculation: {results['addition_statistics']['budget_calculation']}")


#######################
# Main Analysis Function
#######################

def comprehensive_voting_analysis(data_samples, total_votes, algo_works, budget_percent=None, show_plots=True, print_results=True):
    """
    Performs a comprehensive analysis of voting data by calling all analysis functions in sequence.
    
    Parameters:
    data_samples (list): A list of dictionaries where each dictionary represents a sample.
    total_votes (int): The total number of votes in the election.
    algo_works (int): The total number of valid algorithm runs.
    budget_percent (float, optional): The budget percentage for vote additions.
    show_plots (bool, optional): Whether to display plots. Defaults to True.
    print_results (bool, optional): Whether to print detailed results. Defaults to True.
    
    Returns:
    dict: A dictionary containing all analysis results:
        - winning_combinations: Dict tracking how often each combination wins outright
        - additional_votes: Dict tracking how many additional votes are needed
        - vote_additions: Dict tracking how often votes are added to each candidate
        - detailed_additions: Dict with detailed vote addition frequency information
        - winning_frequency: Counter tracking the frequency of outright wins
        - addition_statistics: Dict containing various statistics about vote additions
        - addition_distribution: Dict containing percentage distribution of addition types
        - deviation_analysis: Dict with average non-zero deviation analysis
    """
    # Calculate budget if budget_percent is provided
    budget = total_votes * budget_percent * 0.01 if budget_percent is not None else None
    
    # Step 1: Analyze voting combinations
    winning_combinations, additional_votes, vote_additions = analyze_voting_combinations(data_samples)
    
    # Step 2: Analyze detailed vote additions
    detailed_additions = analyze_detailed_vote_additions(data_samples)
    
    # Step 3: Calculate winning frequency
    winning_freq = calculate_winning_frequency(data_samples)
    
    # Step 4: Calculate vote addition statistics
    addition_stats = calculate_vote_addition_statistics(
        data_samples, total_votes, winning_freq, algo_works, budget)
    
    # Step 5: Analyze addition distribution
    addition_dist, addition_categories = analyze_addition_distribution(data_samples)
    
    # Step 6: Calculate and plot average non-zero deviations
    avg_deviations, zero_count, fig = plot_deviation_histogram(data_samples)
    
    # Show the plot if requested
    if show_plots:
        plt.show()
    
    # Compile all results into a comprehensive dictionary
    comprehensive_results = {
        'winning_combinations': winning_combinations,
        'additional_votes': additional_votes,
        'vote_additions': vote_additions,
        'detailed_additions': detailed_additions,
        'winning_frequency': dict(winning_freq),
        'addition_statistics': addition_stats,
        'addition_distribution': addition_dist,
        'addition_categories': addition_categories,
        'deviation_analysis': {
            'average_deviations': avg_deviations,
            'zero_count': zero_count,
            'total_samples': len(avg_deviations)
        }
    }
    
    # Print detailed results if requested
    if print_results:
        print_detailed_results(comprehensive_results, total_votes)
    
    return comprehensive_results, fig