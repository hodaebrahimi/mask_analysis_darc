import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

def load_masks_from_directory(masks_dir):
    """
    Load masks from directory with simple naming format:
    - IBD_0000.nii.gz, IBD_0001.nii.gz, etc.
    
    Args:
        masks_dir (str): Path to directory containing mask files
    
    Returns:
        dict: Dictionary with case IDs as keys and file paths as values
    """
    masks_dir = Path(masks_dir)
    mask_files = {}
    
    # Find all .nii.gz files in the directory
    all_files = list(masks_dir.glob("*.nii.gz"))
    
    print(f"Found {len(all_files)} .nii.gz files in {masks_dir}")
    
    # Show some example filenames to understand the pattern
    print(f"\nExample filenames (first 10):")
    for i, file_path in enumerate(all_files[:10]):
        print(f"  {i+1:2d}. {file_path.name}")
    
    # Process files - assuming all are probability masks
    for file_path in all_files:
        filename = file_path.name
        
        # Extract case ID from filename (remove .nii.gz extension)
        if filename.endswith('.nii.gz'):
            case_id = filename.replace('.nii.gz', '')
            mask_files[case_id] = str(file_path)
    
    print(f"Found {len(mask_files)} mask files to process")
    
    return mask_files

def calculate_edge_fuzziness(probability_mask_path, show_debug=False):
    """
    Calculate three different edge fuzziness measures from probability mask.
    
    NOTE: Input masks are PROBABILITY masks (values 0-1), not binary masks.
    These come from Connor's 3D U-Net model outputs.
    
    Args:
        probability_mask_path (str): Path to probability mask file (values between 0-1)
        show_debug (bool): Whether to show detailed debugging information
    
    Returns:
        dict: Dictionary containing all three metrics and additional statistics
    """
    try:
        # Load the NIfTI file - this contains PROBABILITY values (0-1)
        nii_img = nib.load(probability_mask_path)
        data = nii_img.get_fdata()
        
        # Debug: Check data characteristics
        if show_debug:
            unique_values = np.unique(data)
            print(f"\n=== DEBUG INFO ===")
            print(f"File: {os.path.basename(probability_mask_path)}")
            print(f"  Data shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Value range: [{data.min():.6f}, {data.max():.6f}]")
            print(f"  Unique values count: {len(unique_values)}")
            
            if len(unique_values) <= 20:
                print(f"  ALL unique values: {unique_values}")
            else:
                print(f"  First 20 unique values: {unique_values[:20]}")
                print(f"  Last 5 unique values: {unique_values[-5:]}")
            
            # Count voxels by value type
            zero_count = np.sum(data == 0)
            one_count = np.sum(data == 1)
            between_count = np.sum((data > 0) & (data < 1))
            total_nonzero = np.sum(data > 0)
            
            print(f"  Voxels == 0 (background): {zero_count:,} ({100*zero_count/data.size:.1f}%)")
            print(f"  Voxels == 1 (certain fg): {one_count:,} ({100*one_count/data.size:.1f}%)")
            print(f"  Voxels between 0-1 (uncertain): {between_count:,} ({100*between_count/data.size:.1f}%)")
            print(f"  Total non-zero voxels: {total_nonzero:,}")
            
            if between_count > 0:
                uncertain_vals = data[(data > 0) & (data < 1)]
                print(f"  Uncertain voxel stats:")
                print(f"    Min: {uncertain_vals.min():.6f}")
                print(f"    Max: {uncertain_vals.max():.6f}")
                print(f"    Mean: {uncertain_vals.mean():.6f}")
                print(f"    Median: {np.median(uncertain_vals):.6f}")
            else:
                print(f"  >>> NO UNCERTAIN VOXELS FOUND! All voxels are either 0 or 1.")
                print(f"  >>> This suggests the mask is BINARY, not a probability mask!")
        
        # Get uncertain voxels (probability between 0 and 1, excluding exactly 0 and 1)
        # 0 = background/certain background, 1 = certain foreground
        # Values between 0-1 represent uncertainty at boundaries
        uncertain_mask = (data > 0) & (data < 1)
        uncertain_voxels = data[uncertain_mask]
        
        # Calculate all three metrics
        
        # 1. Mean uncertainty (existing method)
        if len(uncertain_voxels) > 0:
            mean_uncertainty = np.mean(1 - uncertain_voxels)
        else:
            mean_uncertainty = 0.0
        
        # 2. Sum of uncertainties = 1 - sum(probability mask)
        # This considers all non-zero voxels
        sum_probabilities = np.sum(data)
        sum_uncertainties = np.sum(data > 0) - sum_probabilities  # count_nonzero - sum_probabilities
        
        # 3. Count of uncertain pixels
        count_uncertain = len(uncertain_voxels)
        
        if show_debug:
            print(f"  >>> METRICS CALCULATED:")
            print(f"  >>> Mean uncertainty: {mean_uncertainty:.6f}")
            print(f"  >>> Sum of uncertainties: {sum_uncertainties:.2f}")
            print(f"  >>> Count of uncertain pixels: {count_uncertain:,}")
        
        # Additional statistics
        stats = {
            'total_voxels': data.size,
            'uncertain_voxels': count_uncertain,
            'background_voxels': np.sum(data == 0),
            'certain_foreground_voxels': np.sum(data == 1),
            'sum_probabilities': sum_probabilities,
            'nonzero_voxels': np.sum(data > 0),
            'mean_probability': np.mean(uncertain_voxels) if len(uncertain_voxels) > 0 else 0.0,
            'min_prob': np.min(uncertain_voxels) if len(uncertain_voxels) > 0 else 0.0,
            'max_prob': np.max(uncertain_voxels) if len(uncertain_voxels) > 0 else 1.0
        }
        
        # Return all metrics
        result = {
            'mean_uncertainty': mean_uncertainty,
            'sum_uncertainties': sum_uncertainties,
            'count_uncertain': count_uncertain,
            **stats
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {probability_mask_path}: {str(e)}")
        return None

def process_all_masks(masks_dir, output_dir):
    """
    Process all masks in directory and save results with three ranking systems.
    
    Args:
        masks_dir (str): Directory containing mask files
        output_dir (str): Directory to save results
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mask file paths
    mask_files = load_masks_from_directory(masks_dir)
    
    results = []
    
    print(f"\nProcessing {len(mask_files)} cases (showing debug info for first 5)...")
    
    processed_count = 0
    for case_id, mask_path in tqdm(mask_files.items(), desc="Processing masks"):
        
        # Show debug info only for first 5 cases
        show_debug = processed_count < 5
        
        if show_debug:
            print(f"\n--- Processing case {processed_count + 1}: {case_id} ---")
            print(f"File path: {mask_path}")
        
        # Calculate all metrics
        result = calculate_edge_fuzziness(mask_path, show_debug=show_debug)
        
        if result is not None:
            result_record = {
                'case_id': case_id,
                'mask_path': mask_path,
                'mask_type': 'probability',
                **result
            }
            results.append(result_record)
        else:
            print(f"Failed to process {case_id}")
        
        processed_count += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Create three different rankings
        rankings = {}
        
        # 1. Ranking by mean uncertainty (descending - highest mean uncertainty first)
        df_mean = df.sort_values('mean_uncertainty', ascending=False).reset_index(drop=True)
        df_mean['difficulty_rank_mean'] = range(1, len(df_mean) + 1)
        rankings['mean'] = df_mean
        
        # 2. Ranking by sum of uncertainties (descending - highest sum first)
        df_sum = df.sort_values('sum_uncertainties', ascending=False).reset_index(drop=True)
        df_sum['difficulty_rank_sum'] = range(1, len(df_sum) + 1)
        rankings['sum'] = df_sum
        
        # 3. Ranking by count of uncertain pixels (descending - highest count first)
        df_count = df.sort_values('count_uncertain', ascending=False).reset_index(drop=True)
        df_count['difficulty_rank_count'] = range(1, len(df_count) + 1)
        rankings['count'] = df_count
        
        # Save individual ranking files
        for metric, df_ranked in rankings.items():
            output_file = output_dir / f'EdgeUncertaintyMasks_{metric}_ranking.csv'
            df_ranked.to_csv(output_file, index=False)
            print(f"Saved {metric} ranking to: {output_file}")
        
        # Save combined file with all rankings
        combined_df = df.copy()
        combined_df = combined_df.merge(df_mean[['case_id', 'difficulty_rank_mean']], on='case_id')
        combined_df = combined_df.merge(df_sum[['case_id', 'difficulty_rank_sum']], on='case_id') 
        combined_df = combined_df.merge(df_count[['case_id', 'difficulty_rank_count']], on='case_id')
        
        combined_output_file = output_dir / 'EdgeUncertaintyMasks_all_rankings.csv'
        combined_df.to_csv(combined_output_file, index=False)
        
        # Save comprehensive summary statistics
        summary_file = output_dir / 'EdgeUncertaintyMasks_comprehensive_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Comprehensive Edge Fuzziness Analysis Summary\n")
            f.write(f"=============================================\n\n")
            f.write(f"Total cases processed: {len(df)}\n")
            f.write(f"Input directory: {masks_dir}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            # Statistics for each metric
            for metric_name, metric_col, df_ranked in [
                ('Mean Uncertainty', 'mean_uncertainty', df_mean),
                ('Sum of Uncertainties', 'sum_uncertainties', df_sum),
                ('Count of Uncertain Pixels', 'count_uncertain', df_count)
            ]:
                f.write(f"{metric_name} Statistics:\n")
                f.write(f"  Mean: {df_ranked[metric_col].mean():.4f}\n")
                f.write(f"  Median: {df_ranked[metric_col].median():.4f}\n")
                f.write(f"  Std: {df_ranked[metric_col].std():.4f}\n")
                f.write(f"  Min: {df_ranked[metric_col].min():.4f}\n")
                f.write(f"  Max: {df_ranked[metric_col].max():.4f}\n\n")
                
                rank_col = f'difficulty_rank_{metric_name.split()[0].lower()}'
                f.write(f"Top 10 Most Difficult Cases (by {metric_name}):\n")
                for idx, row in df_ranked.head(10).iterrows():
                    f.write(f"  {row[rank_col]:2d}. {row['case_id']} ({metric_name}: {row[metric_col]:.4f})\n")
                    
                f.write(f"\nTop 10 Easiest Cases (by {metric_name}):\n")
                for idx, row in df_ranked.tail(10).iterrows():
                    f.write(f"  {row[rank_col]:2d}. {row['case_id']} ({metric_name}: {row[metric_col]:.4f})\n")
                f.write(f"\n" + "="*60 + "\n\n")
        
        print(f"\nResults saved to:")
        print(f"  Combined rankings: {combined_output_file}")
        print(f"  Individual rankings: EdgeUncertaintyMasks_*_ranking.csv")
        print(f"  Comprehensive summary: {summary_file}")
        print(f"\nProcessed {len(df)} cases successfully")
        
        # Show summary of what we found
        print(f"\n=== ANALYSIS SUMMARY ===")
        cases_with_uncertainty = df[df['count_uncertain'] > 0]
        binary_cases = df[df['count_uncertain'] == 0]
        
        print(f"Cases with uncertain voxels (probability masks): {len(cases_with_uncertainty)}")
        print(f"Cases with NO uncertain voxels (binary masks): {len(binary_cases)}")
        
        if len(binary_cases) == len(df):
            print(f">>> WARNING: ALL masks appear to be BINARY (only 0s and 1s)!")
            print(f">>> These may not be the probability masks you're looking for.")
        elif len(cases_with_uncertainty) > 0:
            print(f">>> SUCCESS: Found {len(cases_with_uncertainty)} cases with probability values!")
            avg_uncertain_voxels = cases_with_uncertainty['count_uncertain'].mean()
            print(f">>> Average uncertain voxels per case: {avg_uncertain_voxels:,.0f}")
        
        print(f"\nMetric ranges:")
        print(f"  Mean uncertainty: {df['mean_uncertainty'].min():.4f} to {df['mean_uncertainty'].max():.4f}")
        print(f"  Sum uncertainties: {df['sum_uncertainties'].min():.2f} to {df['sum_uncertainties'].max():.2f}")
        print(f"  Count uncertain: {df['count_uncertain'].min():,} to {df['count_uncertain'].max():,}")
        
        return rankings, combined_df
    
    else:
        print("No cases were processed successfully!")
        return None, None

# Main execution
if __name__ == "__main__":
    # Set paths
    masks_directory = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/labelsTr"
    output_directory = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    
    # Process masks with all three ranking methods
    print("Running Comprehensive Edge Fuzziness Analysis on IBD Dataset...")
    print("Note: Expecting probability masks with values (0-1) from Connor's 3D U-Net model")
    print("Calculating three ranking metrics: Mean Uncertainty, Sum of Uncertainties, Count of Uncertain Pixels")
    print(f"Input directory: {masks_directory}")
    print(f"Output directory: {output_directory}")
    
    rankings, combined_df = process_all_masks(
        masks_dir=masks_directory,
        output_dir=output_directory
    )
    
    if rankings is not None:
        print(f"\nAnalysis complete! Check the output directory for results.")
        
        # Display top 3 cases for each metric
        for metric_name, df_ranked in rankings.items():
            print(f"\nTop 3 most difficult cases (by {metric_name}):")
            metric_col = f"{metric_name}_uncertainty" if metric_name == "mean" else f"{metric_name}_uncertain{'ties' if metric_name == 'sum' else ''}"
            if metric_name == "sum":
                metric_col = "sum_uncertainties"
            elif metric_name == "count":
                metric_col = "count_uncertain"
            else:
                metric_col = "mean_uncertainty"
                
            for idx, row in df_ranked.head(3).iterrows():
                print(f"  {row['case_id']}: {metric_col} = {row[metric_col]:.4f}")
    else:
        print("Analysis failed!")