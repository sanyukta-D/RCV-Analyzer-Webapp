from rcv_strategies.core.stv_irv import STV_optimal_result_simple, IRV_ballot_exhaust
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.core import candidate_removal
from rcv_strategies.utils import helpers as utils
from rcv_strategies.utils.case_study_helpers import process_ballot_counts_post_elim_no_print
import pandas as pd
from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
import os
from string import ascii_uppercase
import time

def analyze_high_candidate_elections():
    """
    Analyze NYC RCV elections with >10 candidates, comparing before/after strategy optimization
    """
    nyc_folder = "Case_Studies/NewYork_City_RCV/nyc_files"
    
    # Results for comparison
    before_results = []  # Without strategy optimization
    after_results = []   # With strategy optimization
    
    file_iter = 0
    processed_elections = []
    
    print("="*80)
    print("ANALYZING NYC RCV ELECTIONS WITH >10 CANDIDATES")
    print("="*80)
    
    for input_file in os.listdir(nyc_folder):
        if not input_file.endswith('.csv'):
            continue
            
        file_iter += 1
        print(f"\nProcessing file {file_iter}: {input_file}")
        
        try:
            full_path = os.path.join(nyc_folder, input_file)
            df, processed_file = process_single_file(full_path)
            
            k = 1
            budget_percent = 15
            
            # Initial processing: build original mapping and ballot counts
            candidates_mapping = create_candidate_mapping(processed_file)
            ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
            
            candidates = list(candidates_mapping.values())
            
            # Only process elections with >10 candidates
            if len(candidates) <= 10:
                print(f"  Skipping - only {len(candidates)} candidates")
                continue
                
            print(f"  Processing election with {len(candidates)} candidates")
            
            # Reorder candidates by STV results
            rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, sum(ballot_counts.values())/(k+1))
            results, subresults = utils.return_main_sub(rt)
            
            # Create final mapping with ordered candidates
            reverse_mapping = {code: name for name, code in candidates_mapping.items()}
            ordered_candidate_names = [reverse_mapping[code] for code in results]
            final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
            
            ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
            candidates = list(final_mapping.values())
            
            # Calculate exhausted ballots
            exhausted_ballots_list, exhausted_ballots_dict = IRV_ballot_exhaust(candidates, ballot_counts)
            exhausted_ballots_dict_percent = {key: round(value/sum(ballot_counts.values())*100, 3) 
                                            for key, value in exhausted_ballots_dict.items()}
            
            elim_cands = []
            
            # BEFORE: Analysis without strategy optimization
            print("  Running BEFORE analysis (no strategy optimization)...")
            start_time = time.time()
            before_result = process_ballot_counts_post_elim_no_print(
                ballot_counts,
                k,
                candidates,
                elim_cands,
                check_strats=False,  # No strategy checking
                budget_percent=budget_percent,
                check_removal_here=True,
                keep_at_least=9
            )
            before_time = time.time() - start_time
            
            before_result["file_name"] = input_file
            before_result["exhaust_percents"] = exhausted_ballots_dict_percent
            before_result["budget_percent"] = budget_percent
            before_result["processing_time"] = before_time
            before_result["analysis_type"] = "Before (No Strategies)"
            before_results.append(before_result)
            
            # AFTER: Analysis with strategy optimization
            print("  Running AFTER analysis (with strategy optimization)...")
            start_time = time.time()
            after_result = process_ballot_counts_post_elim_no_print(
                ballot_counts,
                k,
                candidates,
                elim_cands,
                check_strats=True,   # Enable strategy checking
                budget_percent=budget_percent,
                check_removal_here=True,
                keep_at_least=9
            )
            after_time = time.time() - start_time
            
            after_result["file_name"] = input_file
            after_result["exhaust_percents"] = exhausted_ballots_dict_percent
            after_result["budget_percent"] = budget_percent
            after_result["processing_time"] = after_time
            after_result["analysis_type"] = "After (With Strategies)"
            after_results.append(after_result)
            
            # Extract election name for display
            election_name = input_file.replace("NewYorkCity_06222021_", "").replace(".csv", "")
            processed_elections.append({
                "election_name": election_name,
                "candidates": len(candidates),
                "total_votes": before_result["total_votes"],
                "before_time": before_time,
                "after_time": after_time,
                "strategies_calculated": bool(after_result.get("Strategies"))
            })
            
            print(f"  ✓ Completed in {before_time + after_time:.2f}s total")
            
        except Exception as e:
            print(f"  ✗ Error processing {input_file}: {str(e)}")
            continue
    
    # Create comparison analysis
    print("\n" + "="*80)
    print("BEFORE vs AFTER COMPARISON RESULTS")
    print("="*80)
    
    if not before_results or not after_results:
        print("No elections with >10 candidates found to analyze.")
        return
    
    # Summary statistics
    print(f"\nProcessed {len(processed_elections)} elections with >10 candidates:")
    for election in processed_elections:
        print(f"  • {election['election_name']}: {election['candidates']} candidates, {election['total_votes']:,} votes")
    
    print(f"\nTotal processing time comparison:")
    total_before_time = sum(e['before_time'] for e in processed_elections)
    total_after_time = sum(e['after_time'] for e in processed_elections)
    print(f"  • Before (no strategies): {total_before_time:.2f}s")
    print(f"  • After (with strategies): {total_after_time:.2f}s")
    print(f"  • Additional time for strategies: {total_after_time - total_before_time:.2f}s")
    
    # Create comprehensive comparison DataFrame
    before_df = pd.DataFrame(before_results)
    after_df = pd.DataFrame(after_results)
    
    # Key comparison metrics
    comparison_columns = ["file_name", "num_candidates", "total_votes", "candidates_removed", 
                         "candidates_retained", "processing_time", "Strategies"]
    
    print(f"\n" + "="*80)
    print("DETAILED COMPARISON TABLE")
    print("="*80)
    
    # Create side-by-side comparison
    for i, (before, after) in enumerate(zip(before_results, after_results)):
        election_name = before["file_name"].replace("NewYorkCity_06222021_", "").replace(".csv", "")
        print(f"\n{i+1}. {election_name}")
        print(f"   Candidates: {before['num_candidates']}, Votes: {before['total_votes']:,}")
        print(f"   BEFORE: Removed {len(before['candidates_removed'])}, Retained {len(before['candidates_retained'])}")
        print(f"   AFTER:  Removed {len(after['candidates_removed'])}, Retained {len(after['candidates_retained'])}")
        
        # Strategy analysis
        if after.get('Strategies'):
            strategies = after['Strategies']
            if strategies:
                strategy_summary = []
                for candidate, data in strategies.items():
                    if isinstance(data, list) and len(data) >= 1:
                        cost = data[0] if isinstance(data[0], (int, float)) else 0
                        strategy_summary.append(f"{candidate}: {cost}%")
                print(f"   STRATEGIES: {', '.join(strategy_summary[:3])}{'...' if len(strategy_summary) > 3 else ''}")
        
        print(f"   TIME: Before {before['processing_time']:.2f}s → After {after['processing_time']:.2f}s")
    
    # Export detailed results
    all_results = before_results + after_results
    df_all = pd.DataFrame(all_results)
    
    export_columns = ["file_name", "analysis_type", "num_candidates", "total_votes", 
                     "candidates_removed", "candidates_retained", "Strategies", 
                     "processing_time", "budget_percent"]
    
    output_filename = "high_candidate_elections_comparison.xlsx"
    df_all[export_columns].to_excel(output_filename, index=False)
    print(f"\n📊 Detailed comparison exported to {output_filename}")
    
    # Summary statistics
    print(f"\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    avg_candidates = sum(e['candidates'] for e in processed_elections) / len(processed_elections)
    total_votes = sum(e['total_votes'] for e in processed_elections)
    
    print(f"Elections analyzed: {len(processed_elections)}")
    print(f"Average candidates per election: {avg_candidates:.1f}")
    print(f"Total votes across all elections: {total_votes:,}")
    
    # Strategy effectiveness summary
    strategy_elections = [r for r in after_results if r.get('Strategies')]
    print(f"Elections with strategy calculations: {len(strategy_elections)}")
    
    if strategy_elections:
        avg_strategy_time = sum(r['processing_time'] for r in strategy_elections) / len(strategy_elections)
        avg_no_strategy_time = sum(r['processing_time'] for r in before_results) / len(before_results)
        print(f"Average time without strategies: {avg_no_strategy_time:.2f}s")
        print(f"Average time with strategies: {avg_strategy_time:.2f}s")
        print(f"Strategy overhead: {avg_strategy_time - avg_no_strategy_time:.2f}s per election")
    
    return before_results, after_results, processed_elections

if __name__ == "__main__":
    before_results, after_results, processed_elections = analyze_high_candidate_elections() 