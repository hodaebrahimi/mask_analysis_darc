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

def calculate_edge_fuzziness(probability_mask_path, method='uncertainty', show_debug=False):
    """
    Calculate edge fuzziness measure from probability mask.
    
    NOTE: Input masks are PROBABILITY masks (values 0-1), not binary masks.
    These come from Connor's 3D U-Net model outputs.
    
    Args:
        probability_mask_path (str): Path to probability mask file (values between 0-1)
        method (str): 'uncertainty' for (1-p) approach, 'simple' for sum approach
        show_debug (bool): Whether to show detailed debugging information
    
    Returns:
        float: Edge fuzziness score
        dict: Additional statistics
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
        valid_mask = (data > 0) & (data < 1)
        uncertain_voxels = data[valid_mask]
        
        if len(uncertain_voxels) == 0:
            if show_debug:
                print(f"  >>> Returning zero scores for binary mask")
            
            return 0.0, {
                'total_voxels': data.size,
                'uncertain_voxels': 0,
                'background_voxels': np.sum(data == 0),
                'certain_foreground_voxels': np.sum(data == 1),
                'mean_uncertainty': 0.0,
                'std_uncertainty': 0.0,
                'mean_probability': 0.0,
                'min_prob': 0.0,
                'max_prob': 1.0
            }
        
        # Calculate fuzziness score based on method
        if method == 'uncertainty':
            # (1 - probability) approach - higher values for more uncertain voxels
            # Voxels close to 0.5 contribute most, voxels close to 0.99 contribute little
            uncertainties = 1 - uncertain_voxels
            fuzziness_score = np.sum(uncertainties)
        elif method == 'simple':
            # Simple sum approach - sum all probability values between 0 and 1
            fuzziness_score = np.sum(uncertain_voxels)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if show_debug:
            print(f"  >>> FOUND {len(uncertain_voxels):,} uncertain voxels!")
            print(f"  >>> Fuzziness score ({method}): {fuzziness_score:.2f}")
        
        # Additional statistics
        stats = {
            'total_voxels': data.size,
            'uncertain_voxels': len(uncertain_voxels),
            'background_voxels': np.sum(data == 0),
            'certain_foreground_voxels': np.sum(data == 1),
            'mean_uncertainty': np.mean(1 - uncertain_voxels) if len(uncertain_voxels) > 0 else 0.0,
            'std_uncertainty': np.std(1 - uncertain_voxels) if len(uncertain_voxels) > 0 else 0.0,
            'mean_probability': np.mean(uncertain_voxels) if len(uncertain_voxels) > 0 else 0.0,
            'min_prob': np.min(uncertain_voxels) if len(uncertain_voxels) > 0 else 0.0,
            'max_prob': np.max(uncertain_voxels) if len(uncertain_voxels) > 0 else 1.0
        }
        
        return fuzziness_score, stats
        
    except Exception as e:
        print(f"Error processing {probability_mask_path}: {str(e)}")
        return None, None

def process_all_masks(masks_dir, output_dir, method='uncertainty'):
    """
    Process all masks in directory and save results.
    
    Args:
        masks_dir (str): Directory containing mask files
        output_dir (str): Directory to save results
        method (str): Method for calculating fuzziness
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
        
        # Calculate fuzziness
        fuzziness_score, stats = calculate_edge_fuzziness(mask_path, method=method, show_debug=show_debug)
        
        if fuzziness_score is not None:
            result = {
                'case_id': case_id,
                'mask_path': mask_path,
                'mask_type': 'probability',  # Assuming all are probability masks
                'fuzziness_score': fuzziness_score,
                'method': method,
                **stats
            }
            results.append(result)
        else:
            print(f"Failed to process {case_id}")
        
        processed_count += 1
    
    # Convert to DataFrame and sort by mean uncertainty
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Sort by mean uncertainty (descending - highest mean uncertainty first)
        df_sorted = df.sort_values('mean_uncertainty', ascending=False).reset_index(drop=True)
        df_sorted['difficulty_rank'] = range(1, len(df_sorted) + 1)
        
        # Save results
        output_file = output_dir / 'EdgeUncertaintyMasks.csv'
        df_sorted.to_csv(output_file, index=False)
        
        # Save summary statistics
        summary_file = output_dir / 'EdgeUncertaintyMasks_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Edge Fuzziness Analysis Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Method: {method}\n")
            f.write(f"Total cases processed: {len(df_sorted)}\n")
            f.write(f"Input directory: {masks_dir}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            f.write(f"Mean Uncertainty Statistics:\n")
            f.write(f"  Mean: {df_sorted['mean_uncertainty'].mean():.4f}\n")
            f.write(f"  Median: {df_sorted['mean_uncertainty'].median():.4f}\n")
            f.write(f"  Std: {df_sorted['mean_uncertainty'].std():.4f}\n")
            f.write(f"  Min: {df_sorted['mean_uncertainty'].min():.4f}\n")
            f.write(f"  Max: {df_sorted['mean_uncertainty'].max():.4f}\n\n")
            
            f.write(f"Fuzziness Score Statistics:\n")
            f.write(f"  Mean: {df_sorted['fuzziness_score'].mean():.2f}\n")
            f.write(f"  Median: {df_sorted['fuzziness_score'].median():.2f}\n")
            f.write(f"  Std: {df_sorted['fuzziness_score'].std():.2f}\n")
            f.write(f"  Min: {df_sorted['fuzziness_score'].min():.2f}\n")
            f.write(f"  Max: {df_sorted['fuzziness_score'].max():.2f}\n\n")
            
            f.write(f"Top 10 Most Difficult Cases (by mean uncertainty):\n")
            for idx, row in df_sorted.head(10).iterrows():
                f.write(f"  {row['difficulty_rank']:2d}. {row['case_id']} (mean uncertainty: {row['mean_uncertainty']:.4f}, score: {row['fuzziness_score']:.2f})\n")
                
            f.write(f"\nTop 10 Easiest Cases (by mean uncertainty):\n")
            for idx, row in df_sorted.tail(10).iterrows():
                f.write(f"  {row['difficulty_rank']:2d}. {row['case_id']} (mean uncertainty: {row['mean_uncertainty']:.4f}, score: {row['fuzziness_score']:.2f})\n")
        
        print(f"\nResults saved to:")
        print(f"  Main results: {output_file}")
        print(f"  Summary: {summary_file}")
        print(f"\nProcessed {len(df_sorted)} cases successfully")
        print(f"Mean uncertainty ranges from {df_sorted['mean_uncertainty'].min():.4f} to {df_sorted['mean_uncertainty'].max():.4f}")
        print(f"Fuzziness scores range from {df_sorted['fuzziness_score'].min():.2f} to {df_sorted['fuzziness_score'].max():.2f}")
        
        # Show summary of what we found
        print(f"\n=== ANALYSIS SUMMARY ===")
        cases_with_uncertainty = df_sorted[df_sorted['uncertain_voxels'] > 0]
        binary_cases = df_sorted[df_sorted['uncertain_voxels'] == 0]
        
        print(f"Cases with uncertain voxels (probability masks): {len(cases_with_uncertainty)}")
        print(f"Cases with NO uncertain voxels (binary masks): {len(binary_cases)}")
        
        if len(binary_cases) == len(df_sorted):
            print(f">>> WARNING: ALL masks appear to be BINARY (only 0s and 1s)!")
            print(f">>> These may not be the probability masks you're looking for.")
        elif len(cases_with_uncertainty) > 0:
            print(f">>> SUCCESS: Found {len(cases_with_uncertainty)} cases with probability values!")
            avg_uncertain_voxels = cases_with_uncertainty['uncertain_voxels'].mean()
            print(f">>> Average uncertain voxels per case: {avg_uncertain_voxels:,.0f}")
        
        return df_sorted
    
    else:
        print("No cases were processed successfully!")
        return None

# Main execution
if __name__ == "__main__":
    # Set paths
    masks_directory = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/labelsTr"
    output_directory = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    
    # Process masks using uncertainty method (1 - probability)
    print("Running Edge Fuzziness Analysis on IBD Dataset...")
    print("Note: Expecting probability masks with values (0-1) from Connor's 3D U-Net model")
    print(f"Input directory: {masks_directory}")
    print(f"Output directory: {output_directory}")
    
    results_df = process_all_masks(
        masks_dir=masks_directory,
        output_dir=output_directory,
        method='uncertainty'  # Use 'simple' for the alternative approach
    )
    
    if results_df is not None:
        print(f"\nAnalysis complete! Check the output directory for results.")
        
        # Display top 5 most and least uncertain cases
        print(f"\nTop 5 most difficult cases (by mean uncertainty):")
        for idx, row in results_df.head(5).iterrows():
            print(f"  {row['case_id']}: mean uncertainty = {row['mean_uncertainty']:.4f}")
            
        print(f"\nTop 5 easiest cases (by mean uncertainty):")
        for idx, row in results_df.tail(5).iterrows():
            print(f"  {row['case_id']}: mean uncertainty = {row['mean_uncertainty']:.4f}")