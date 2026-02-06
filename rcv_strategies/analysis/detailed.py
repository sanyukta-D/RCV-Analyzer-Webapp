import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('ballot_exhastion/model_comparison_results/all_elections_analysis.csv')

# Define the probability model columns to average
prob_columns = ['beta_model_prob', 'posterior_beta_prob', 'prior_posterior_beta_prob', 
                'category_bootstrap_prob', 'limited_bootstrap_prob', 'unconditional_bootstrap_prob']

# Replace empty strings and NaN values with 0 for probability columns
for col in prob_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Calculate average probability across all models for each row
df['avg_probability'] = df[prob_columns].mean(axis=1)

print("Detailed Analysis of Probability Models and Heatmap Data")
print("=" * 60)

# Show sample data for inspection
print("\nSample of raw data:")
print(df[['region', 'gap_to_win_pct', 'exhaust_pct'] + prob_columns + ['avg_probability']].head(10))

# Define bins to match the original heatmap
gap_bins = [0, 5, 10, 15, 25, 100]
exhaust_bins = [0, 10, 15, 25, 100]

# Create bin labels
def create_bins(bin_edges):
    bin_labels = []
    for i in range(len(bin_edges)-1):
        bin_labels.append(f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}%")
    return bin_labels

gap_labels = create_bins(gap_bins)
exhaust_labels = create_bins(exhaust_bins)

# Assign bins
df['gap_bin'] = pd.cut(df['gap_to_win_pct'], bins=gap_bins, labels=gap_labels, include_lowest=True)
df['exhaust_bin'] = pd.cut(df['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels, include_lowest=True)

print("\n" + "="*60)
print("BINNED DATA ANALYSIS")
print("="*60)

for region in ['NYC', 'Alaska']:
    print(f"\n{region} Region Analysis:")
    print("-" * 30)
    
    region_data = df[df['region'] == region]
    
    # Group by bins and calculate statistics
    grouped = region_data.groupby(['gap_bin', 'exhaust_bin']).agg({
        'avg_probability': ['mean', 'count', 'std'],
        'gap_to_win_pct': ['min', 'max'],
        'exhaust_pct': ['min', 'max']
    }).round(4)
    
    print("Bin-wise average probabilities:")
    for gap_bin in gap_labels:
        for exhaust_bin in exhaust_labels:
            subset = region_data[(region_data['gap_bin'] == gap_bin) & 
                               (region_data['exhaust_bin'] == exhaust_bin)]
            if len(subset) > 0:
                avg_prob = subset['avg_probability'].mean()
                count = len(subset)
                print(f"  {gap_bin} gap, {exhaust_bin} exhaust: {avg_prob:.4f} (n={count})")

print("\n" + "="*60)
print("ORIGINAL HEATMAP VALUES COMPARISON")
print("="*60)

# Based on visual inspection of the original heatmap, let's extract some values
print("\nFrom the original heatmap (approximate visual readings):")
print("NYC:")
print("  0-5% gap, 0-10% exhaust: ~0.65 (our calculation will show actual)")
print("  0-5% gap, 10-15% exhaust: ~0.72")
print("  2-5% gap, 25-100% exhaust: ~0.29")

print("\nAlaska:")
print("  0-5% gap, 15-25% exhaust: ~0.79")
print("  2-5% gap, 15-25% exhaust: ~0.08")

print("\n" + "="*60)
print("KEY FINDINGS FROM OUR ANALYSIS")
print("="*60)

# Calculate actual values for key cells
nyc_data = df[df['region'] == 'NYC']
alaska_data = df[df['region'] == 'Alaska']

# NYC key values
nyc_low_gap_low_exhaust = nyc_data[(nyc_data['gap_to_win_pct'] <= 5) & 
                                   (nyc_data['exhaust_pct'] <= 10)]
if len(nyc_low_gap_low_exhaust) > 0:
    print(f"NYC 0-5% gap, 0-10% exhaust: {nyc_low_gap_low_exhaust['avg_probability'].mean():.4f} (n={len(nyc_low_gap_low_exhaust)})")

# Alaska key values  
alaska_low_gap_mid_exhaust = alaska_data[(alaska_data['gap_to_win_pct'] <= 5) & 
                                        (alaska_data['exhaust_pct'] >= 15) & 
                                        (alaska_data['exhaust_pct'] <= 25)]
if len(alaska_low_gap_mid_exhaust) > 0:
    print(f"Alaska 0-5% gap, 15-25% exhaust: {alaska_low_gap_mid_exhaust['avg_probability'].mean():.4f} (n={len(alaska_low_gap_mid_exhaust)})")

print("\n" + "="*60)
print("INDIVIDUAL MODEL CONTRIBUTIONS")
print("="*60)

for region in ['NYC', 'Alaska']:
    region_data = df[df['region'] == region]
    print(f"\n{region} - Average contribution by model:")
    for col in prob_columns:
        avg_val = region_data[col].mean()
        non_zero_count = (region_data[col] > 0).sum()
        print(f"  {col}: {avg_val:.6f} (non-zero in {non_zero_count}/{len(region_data)} cases)")

# Show cases where models disagree significantly
print("\n" + "="*60)
print("CASES WITH HIGH MODEL DISAGREEMENT")
print("="*60)

df['model_std'] = df[prob_columns].std(axis=1)
high_disagreement = df[df['model_std'] > 0.1]

if len(high_disagreement) > 0:
    print("Elections with high standard deviation across models (>0.1):")
    for idx, row in high_disagreement.iterrows():
        print(f"  {row['region']}: Gap={row['gap_to_win_pct']:.1f}%, Exhaust={row['exhaust_pct']:.1f}%, Std={row['model_std']:.3f}")
        print(f"    Individual models: {[f'{row[col]:.3f}' for col in prob_columns]}")
else:
    print("No elections with high model disagreement found.") 