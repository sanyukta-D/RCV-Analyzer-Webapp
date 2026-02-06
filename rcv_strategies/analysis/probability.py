#!/usr/bin/env python3
"""
Create a comprehensive probability table for Portland multi-winner STV candidates.
Shows candidate names, letters, strategy %, exhaust %, required preference %, and various probabilities.
"""

import pandas as pd
from case_studies.portland.strategy_analysis import (
    analyze_required_preferences_for_candidates,
    calculate_multi_winner_probabilities
)

def create_comprehensive_table(districts_to_analyze=[1, 2, 4], n_bootstrap=500):
    """
    Create a comprehensive table with all candidate information and probabilities.
    """
    print("PORTLAND MULTI-WINNER STV: COMPREHENSIVE PROBABILITY TABLE")
    print("=" * 120)
    print("Candidates with Exhaust > Strategy: Detailed Analysis")
    print("=" * 120)
    
    all_results = []
    
    # Analyze each district
    for district in districts_to_analyze:
        print(f"\nAnalyzing District {district}...")
        
        try:
            # Get basic required preference analysis
            candidates_analysis = analyze_required_preferences_for_candidates(district)
            
            # Calculate probabilities for each candidate
            for candidate_info in candidates_analysis:
                candidate_letter = candidate_info['letter']
                
                print(f"  Calculating probabilities for {candidate_info['candidate_name']} ({candidate_letter})...")
                
                # Calculate all probability models
                prob_results = calculate_multi_winner_probabilities(
                    district, candidate_letter, k=3, n_bootstrap=n_bootstrap
                )
                
                if prob_results:
                    # Combine basic info with probability results
                    combined_result = {
                        'District': district,
                        'Candidate Name': candidate_info['candidate_name'],
                        'Letter': candidate_letter,
                        'Strategy %': candidate_info['strategy_pct'],
                        'Exhaust %': candidate_info['exhaust_pct'],
                        'Required Pref %': candidate_info['required_preference_pct'],
                        'Active Candidates': candidate_info['active_candidates_count'],
                        'Beta Prob': prob_results['beta_probability'],
                        'Bootstrap Prob': prob_results['bootstrap_probability'],
                        'Similarity Prob': prob_results['similarity_probability'],
                        'Prior-Post Prob': prob_results['prior_posterior_probability'],
                        'Unconditional Prob': prob_results['unconditional_probability'],
                        'Total Exhausted': prob_results['total_exhausted_ballots']
                    }
                    all_results.append(combined_result)
                    
        except Exception as e:
            print(f"Error analyzing District {district}: {e}")
            continue
    
    # Create and display the comprehensive table
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Sort by required preference percentage (ascending - easiest first)
        df = df.sort_values('Required Pref %')
        
        print(f"\n{'='*120}")
        print("COMPREHENSIVE RESULTS TABLE")
        print(f"{'='*120}")
        print(f"Total candidates with exhaust > strategy: {len(df)}")
        print(f"{'='*120}")
        
        # Display the main table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        # Format the dataframe for better display
        display_df = df.copy()
        display_df['Strategy %'] = display_df['Strategy %'].round(2)
        display_df['Exhaust %'] = display_df['Exhaust %'].round(2)
        display_df['Required Pref %'] = display_df['Required Pref %'].round(1)
        display_df['Beta Prob'] = display_df['Beta Prob'].round(4)
        display_df['Bootstrap Prob'] = display_df['Bootstrap Prob'].round(4)
        display_df['Similarity Prob'] = display_df['Similarity Prob'].round(4)
        display_df['Prior-Post Prob'] = display_df['Prior-Post Prob'].round(4)
        display_df['Unconditional Prob'] = display_df['Unconditional Prob'].round(4)
        
        print(display_df.to_string(index=False))
        
        print(f"\n{'='*120}")
        print("SUMMARY STATISTICS")
        print(f"{'='*120}")
        
        print(f"Required Preference Range: {df['Required Pref %'].min():.1f}% to {df['Required Pref %'].max():.1f}%")
        print(f"Strategy Percentage Range: {df['Strategy %'].min():.2f}% to {df['Strategy %'].max():.2f}%")
        print(f"Exhaust Percentage Range: {df['Exhaust %'].min():.2f}% to {df['Exhaust %'].max():.2f}%")
        
        # Calculate average probabilities
        avg_beta = df['Beta Prob'].mean()
        avg_bootstrap = df['Bootstrap Prob'].mean()
        avg_similarity = df['Similarity Prob'].mean()
        avg_prior_post = df['Prior-Post Prob'].mean()
        avg_unconditional = df['Unconditional Prob'].mean()
        
        print(f"\nAverage Probabilities:")
        print(f"  Beta Model: {avg_beta:.1%}")
        print(f"  Bootstrap: {avg_bootstrap:.1%}")
        print(f"  Similarity: {avg_similarity:.1%}")
        print(f"  Prior-Posterior: {avg_prior_post:.1%}")
        print(f"  Unconditional: {avg_unconditional:.1%}")
        
        # Most promising candidates (highest average probability)
        df['Average Prob'] = (df['Beta Prob'] + df['Bootstrap Prob'] + df['Similarity Prob'] + 
                             df['Prior-Post Prob'] + df['Unconditional Prob']) / 5
        
        top_candidates = df.nlargest(3, 'Average Prob')
        
        print(f"\nMost Promising Strategic Opportunities:")
        for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
            print(f"{i}. {row['Candidate Name']} (District {row['District']}, Letter {row['Letter']})")
            print(f"   Required preference: {row['Required Pref %']:.1f}%")
            print(f"   Strategy: {row['Strategy %']:.2f}%, Exhaust: {row['Exhaust %']:.2f}%")
            print(f"   Average probability: {row['Average Prob']:.1%}")
            print(f"   Competes against {row['Active Candidates']-1} other candidates")
        
        print(f"\n{'='*120}")
        print("COLUMN EXPLANATIONS:")
        print("• Strategy %: Gap to win through strategic voting")
        print("• Exhaust %: Percentage of ballots that exhaust before ranking this candidate")
        print("• Required Pref %: Minimum % of exhausted voters who must prefer this candidate")
        print("• Beta Prob: Theoretical probability using Beta distribution")
        print("• Bootstrap Prob: Empirical probability using category-based bootstrap")
        print("• Similarity Prob: Probability using observed preference patterns")
        print("• Prior-Post Prob: Bayesian probability combining theory and observations")
        print("• Unconditional Prob: Probability assuming random ballot completions")
        print(f"{'='*120}")
        
        return df
    else:
        print("No candidates found with exhaust > strategy.")
        return None

if __name__ == "__main__":
    # Run the analysis
    results_df = create_comprehensive_table(districts_to_analyze=[1, 2, 4], n_bootstrap=500) 