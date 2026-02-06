# Paper figure produced by this script:
#   - Fig 7: heatmap.png (Fig \ref{fig:heatmap_prb}, label app:ballot_exhaustion_single_winner)
#
# Input data:
#   - ballot_exhaustion/model_comparison_results/all_elections_analysis.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Read the data
df = pd.read_csv('ballot_exhastion/model_comparison_results/all_elections_analysis.csv')

# Define the probability model columns to average (excluding rank-restricted/limited bootstrap)
prob_columns = ['beta_model_prob', 'posterior_beta_prob', 'prior_posterior_beta_prob', 
                'category_bootstrap_prob', 'unconditional_bootstrap_prob']
# Excluded: 'limited_bootstrap_prob' (rank-restricted bootstrap)

# Replace empty strings and NaN values with 0 for probability columns
for col in prob_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Calculate average probability across all models except rank-restricted for each row
df['avg_probability'] = df[prob_columns].mean(axis=1)

# Create bins for gap_to_win_pct and exhaust_pct
def create_bins(values, bin_edges):
    """Create bins and return bin labels"""
    bin_labels = []
    for i in range(len(bin_edges)-1):
        if i == len(bin_edges)-2:  # Last bin
            bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.0f}%")
        else:
            bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}%")
    return bin_labels

# Define different bin edges for each region
# NYC: finer first gap bin (0-2.5%), keep original exhaust bins
nyc_gap_bins = [0, 2.5, 5, 10, 15, 25, 100]  
nyc_exhaust_bins = [0, 10, 15, 25, 100]

# Alaska: more granular bins to reduce averaging effect
alaska_gap_bins = [0, 1, 2.5, 5, 7.5, 10, 15, 100]
alaska_exhaust_bins = [0, 5, 10, 15, 20, 25, 100]

# Create pivot tables for each region
regions = ['NYC', 'Alaska']
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for idx, region in enumerate(regions):
    region_data = df[df['region'] == region]
    
    # Use different bins for each region
    if region == 'NYC':
        gap_bins = nyc_gap_bins
        exhaust_bins = nyc_exhaust_bins
    else:  # Alaska
        gap_bins = alaska_gap_bins
        exhaust_bins = alaska_exhaust_bins
    
    # Create bin labels
    gap_labels = create_bins(gap_bins, gap_bins)
    exhaust_labels = create_bins(exhaust_bins, exhaust_bins)
    
    # Assign bins for this region
    region_data = region_data.copy()
    region_data['gap_bin'] = pd.cut(region_data['gap_to_win_pct'], bins=gap_bins, labels=gap_labels, include_lowest=True)
    region_data['exhaust_bin'] = pd.cut(region_data['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels, include_lowest=True)
    
    # Create pivot table with average probabilities
    pivot_table = region_data.groupby(['gap_bin', 'exhaust_bin'], observed=True)['avg_probability'].agg(['mean', 'count']).reset_index()
    pivot_matrix = pivot_table.pivot(index='gap_bin', columns='exhaust_bin', values='mean')
    count_matrix = pivot_table.pivot(index='gap_bin', columns='exhaust_bin', values='count')
    
    # Fill NaN values with 0
    pivot_matrix = pivot_matrix.fillna(0)
    count_matrix = count_matrix.fillna(0)
    
    # Create the heatmap
    ax = axes[idx]
    
    # Create custom colormap (yellow to red like in original)
    colors = ['#FFFFCC', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot heatmap
    sns.heatmap(pivot_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                vmin=0, 
                vmax=0.5,
                ax=ax,
                cbar_kws={'label': 'Probability of Candidate B Winning'})
    
    # Add count annotations
    for i in range(len(pivot_matrix.index)):
        for j in range(len(pivot_matrix.columns)):
            count = count_matrix.iloc[i, j] if not pd.isna(count_matrix.iloc[i, j]) else 0
            if count > 0:
                ax.text(j + 0.5, i + 0.7, f'(n={int(count)})', 
                       ha='center', va='center', fontsize=8, color='black')
    
    ax.set_title(f'{region}: Average Combined Probability by Gap and Exhaust Bins')
    ax.set_xlabel('Exhausted Ballot Percentage Bin')
    ax.set_ylabel('Gap Needed for Candidate B to Win (Percentage Bin)')
    
    # Invert y-axis to match original format
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('corrected_heatmap_granular_bins.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Summary of Data (Excluding Rank-Restricted Bootstrap):")
print(f"Total number of elections: {len(df)}")
print(f"NYC elections: {len(df[df['region'] == 'NYC'])}")
print(f"Alaska elections: {len(df[df['region'] == 'Alaska'])}")
print("\nAverage probabilities by region:")
for region in regions:
    region_data = df[df['region'] == region]
    avg_prob = region_data['avg_probability'].mean()
    print(f"{region}: {avg_prob:.4f}")

# Show the probability model contributions (excluding rank-restricted)
print("\nProbability model contributions (averages, excluding rank-restricted):")
for col in prob_columns:
    avg_val = df[col].mean()
    print(f"{col}: {avg_val:.6f}")

print(f"\nExcluded model (rank-restricted):")
excluded_avg = df['limited_bootstrap_prob'].mean()
print(f"limited_bootstrap_prob: {excluded_avg:.6f}")

# Show detailed bin analysis for both regions
print("\n" + "="*60)
print("DETAILED BIN ANALYSIS (Granular Bins)")
print("="*60)

for region in ['NYC', 'Alaska']:
    print(f"\n{region} Region Analysis:")
    print("-" * 30)
    
    region_data = df[df['region'] == region]
    
    # Use different bins for each region
    if region == 'NYC':
        gap_bins = nyc_gap_bins
        exhaust_bins = nyc_exhaust_bins
    else:  # Alaska
        gap_bins = alaska_gap_bins
        exhaust_bins = alaska_exhaust_bins
    
    gap_labels = create_bins(gap_bins, gap_bins)
    exhaust_labels = create_bins(exhaust_bins, exhaust_bins)
    
    region_data = region_data.copy()
    region_data['gap_bin'] = pd.cut(region_data['gap_to_win_pct'], bins=gap_bins, labels=gap_labels, include_lowest=True)
    region_data['exhaust_bin'] = pd.cut(region_data['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels, include_lowest=True)
    
    print("Bin-wise average probabilities:")
    for gap_bin in gap_labels:
        for exhaust_bin in exhaust_labels:
            subset = region_data[(region_data['gap_bin'] == gap_bin) & 
                               (region_data['exhaust_bin'] == exhaust_bin)]
            if len(subset) > 0:
                avg_prob = subset['avg_probability'].mean()
                count = len(subset)
                print(f"  {gap_bin} gap, {exhaust_bin} exhaust: {avg_prob:.4f} (n={count})") 