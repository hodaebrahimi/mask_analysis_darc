#!/usr/bin/env python3
"""
Enhanced Uncertainty Mask Generator
Creates uncertainty masks with probability=1 regions included and generates histograms
Supports three ranking metrics: mean, sum, count
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

def create_uncertainty_mask_with_certain(probability_mask_path, output_path_uint8, output_path_float, case_id, output_dir):
    """
    Create uncertainty masks that include both uncertain regions and certain foreground regions.
    Certain foreground (probability = 1) gets assigned a value slightly smaller than min uncertainty.
    
    Args:
        probability_mask_path (str): Path to probability mask file
        output_path_uint8 (str): Output path for uint8 mask
        output_path_float (str): Output path for float mask
        case_id (str): Case identifier for histogram filename
        output_dir (Path): Output directory for histogram
        
    Returns:
        tuple: (success_bool, histogram_path, stats_dict)
    """
    try:
        print(f"Processing: {os.path.basename(probability_mask_path)}")
        
        # Load the probability mask
        nii_img = nib.load(probability_mask_path)
        prob_data = nii_img.get_fdata().astype(np.float32)
        
        # Find different types of voxels
        background_mask = (prob_data == 0)  # Background
        uncertain_mask = (prob_data > 0) & (prob_data < 1)  # Uncertain regions
        # Certain foreground: all non-background pixels that aren't uncertain
        certain_fg_mask = (prob_data > 0) & ~uncertain_mask  # More robust than prob_data == 1
        
        # Initialize uncertainty data with zeros
        uncertainty_float = np.zeros_like(prob_data, dtype=np.float32)
        uncertainty_uint8 = np.zeros_like(prob_data, dtype=np.uint8)
        
        stats = {
            'uncertain_voxels': np.sum(uncertain_mask),
            'certain_fg_voxels': np.sum(certain_fg_mask),
            'background_voxels': np.sum(background_mask),
            'total_voxels': prob_data.size
        }
        
        if np.sum(uncertain_mask) > 0:
            # Calculate uncertainty (1 - probability) for uncertain voxels
            uncertainty_values = 1.0 - prob_data[uncertain_mask]
            uncertainty_float[uncertain_mask] = uncertainty_values
            
            # Find minimum uncertainty value to assign to certain regions
            min_uncertainty = np.min(uncertainty_values)
            certain_uncertainty_value = min_uncertainty * 0.95  # Slightly smaller than minimum
            
            print(f"  Uncertain voxels: {np.sum(uncertain_mask):,}")
            print(f"  Min uncertainty: {min_uncertainty:.6f}")
            print(f"  Certain fg uncertainty value: {certain_uncertainty_value:.6f}")
            
            stats.update({
                'min_uncertainty': float(min_uncertainty),
                'max_uncertainty': float(np.max(uncertainty_values)),
                'mean_uncertainty': float(np.mean(uncertainty_values)),
                'certain_uncertainty_value': float(certain_uncertainty_value)
            })
        else:
            # No uncertain voxels - assign a small uncertainty value to certain regions
            certain_uncertainty_value = 0.01  # Small default value
            print(f"  No uncertain voxels found, using default certain value: {certain_uncertainty_value:.6f}")
            
            stats.update({
                'min_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'mean_uncertainty': 0.0,
                'certain_uncertainty_value': float(certain_uncertainty_value)
            })
        
        # Assign uncertainty values to certain foreground regions
        if np.sum(certain_fg_mask) > 0:
            uncertainty_float[certain_fg_mask] = certain_uncertainty_value
            print(f"  Certain fg voxels: {np.sum(certain_fg_mask):,}")
            stats['certain_fg_voxels'] = int(np.sum(certain_fg_mask))
        
        # Create uint8 version
        # Scale to [1, 255] range for non-background voxels (0 reserved for background)
        non_bg_mask = uncertainty_float > 0
        if np.sum(non_bg_mask) > 0:
            # Scale all non-zero uncertainty values to [1, 255]
            max_uncertainty = np.max(uncertainty_float[non_bg_mask])
            if max_uncertainty > 0:
                scaled_values = (uncertainty_float[non_bg_mask] / max_uncertainty * 254) + 1
                uncertainty_uint8[non_bg_mask] = np.clip(scaled_values, 1, 255).astype(np.uint8)
        
        # Create histograms of uncertainty values
        histogram_path = create_uncertainty_histogram(uncertainty_float, case_id, output_dir, stats)
        
        # Create and save NIfTI images
        # Float version
        uncertainty_nii_float = nib.Nifti1Image(uncertainty_float, nii_img.affine, nii_img.header)
        uncertainty_nii_float.header.set_data_dtype(np.float32)
        nib.save(uncertainty_nii_float, output_path_float)
        
        # Uint8 version  
        uncertainty_nii_uint8 = nib.Nifti1Image(uncertainty_uint8, nii_img.affine, nii_img.header)
        uncertainty_nii_uint8.header.set_data_dtype(np.uint8)
        nib.save(uncertainty_nii_uint8, output_path_uint8)
        
        print(f"  Saved float: {output_path_float}")
        print(f"  Saved uint8: {output_path_uint8}")
        print(f"  Saved histogram: {histogram_path}")
        
        return True, histogram_path, stats
        
    except Exception as e:
        print(f"ERROR processing {probability_mask_path}: {str(e)}")
        return False, None, None

def create_uncertainty_histogram(uncertainty_data, case_id, output_dir, stats):
    """
    Create and save histogram of uncertainty pixel values.
    
    Args:
        uncertainty_data (np.array): Uncertainty values
        case_id (str): Case identifier
        output_dir (Path): Output directory
        stats (dict): Statistics dictionary
        
    Returns:
        str: Path to saved histogram
    """
    # Get non-zero uncertainty values for histogram
    non_zero_uncertainties = uncertainty_data[uncertainty_data > 0]
    
    if len(non_zero_uncertainties) == 0:
        print(f"  No non-zero uncertainties for histogram: {case_id}")
        return None
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create normalized histogram
    n_bins = min(50, len(np.unique(non_zero_uncertainties)))
    n, bins, patches = plt.hist(non_zero_uncertainties, bins=n_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Customize plot
    plt.title(f'Normalized Uncertainty Values Distribution - {case_id}', fontsize=14, fontweight='bold')
    plt.xlabel('Uncertainty Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Total uncertain voxels: {len(non_zero_uncertainties):,}
Mean uncertainty: {np.mean(non_zero_uncertainties):.4f}
Min uncertainty: {np.min(non_zero_uncertainties):.4f}
Max uncertainty: {np.max(non_zero_uncertainties):.4f}
Certain fg voxels: {stats.get('certain_fg_voxels', 0):,}"""
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Highlight certain foreground value if present
    if 'certain_uncertainty_value' in stats and stats['certain_fg_voxels'] > 0:
        certain_val = stats['certain_uncertainty_value']
        plt.axvline(certain_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Certain FG value: {certain_val:.4f}')
        plt.legend()
    
    plt.tight_layout()
    
    # Save histogram
    histogram_path = output_dir / f"{case_id}_histogram.png"
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()  # Important: close figure to free memory
    
    return str(histogram_path)

def create_itksnap_colormap(colormap_path):
    """Create ITK-SNAP compatible colormap file (.lut format)."""
    colormap_content = '''# ITK-SNAP Label Description File
# File format: 
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields: 
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label index
#    MSH:   Label mesh visibility (0 or 1)
#    LABEL: Label description 
    0     0     0     0    0.00  0  0    "Background"
   25     0     0   255   0.70  1  0    "Very Low Uncertainty"
   51     0   100   255   0.70  1  0    "Low Uncertainty" 
   76     0   180   200   0.70  1  0    "Low-Med Uncertainty"
  102     0   255   125   0.70  1  0    "Med Uncertainty"
  127   125   255     0   0.70  1  0    "Med-High Uncertainty"
  153   200   180     0   0.70  1  0    "High Uncertainty"
  178   255   100     0   0.70  1  0    "Higher Uncertainty"
  204   255    50     0   0.70  1  0    "Very High Uncertainty"
  229   255    25     0   0.70  1  0    "Extreme Uncertainty"
  255   255     0     0   0.70  1  0    "Maximum Uncertainty"
'''
    
    with open(colormap_path, 'w') as f:
        f.write(colormap_content)
    
    print(f"Created ITK-SNAP colormap: {colormap_path}")
    return str(colormap_path)

def generate_enhanced_uncertainty_masks(base_dir, source_dir, top_n=10):
    """
    Generate uncertainty masks for all three ranking metrics with histograms.
    
    Args:
        base_dir (str): Base directory containing CSV files
        source_dir (str): Source directory containing probability masks  
        top_n (int): Number of top and bottom cases to process for each metric
    """
    
    base_dir = Path(base_dir)
    
    # Define the three metrics and their corresponding CSV files
    metrics = {
        'mean': 'EdgeUncertaintyMasks_mean_ranking.csv',
        'sum': 'EdgeUncertaintyMasks_sum_ranking.csv', 
        'count': 'EdgeUncertaintyMasks_count_ranking.csv'
    }
    
    # Create main output directory
    output_base = base_dir / "enhanced_uncertainty_masks"
    output_base.mkdir(exist_ok=True)
    
    # Create ITK-SNAP colormap (shared across all metrics)
    colormap_path = output_base / "uncertainty_colormap.txt"
    create_itksnap_colormap(colormap_path)
    
    total_files_created = 0
    all_histogram_paths = []
    
    # Process each metric
    for metric, csv_filename in metrics.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING METRIC: {metric.upper()}")
        print(f"{'='*60}")
        
        csv_path = base_dir / csv_filename
        
        if not csv_path.exists():
            print(f"WARNING: CSV file not found: {csv_path}")
            continue
            
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} cases from {csv_filename}")
        
        # Create metric-specific output directory
        metric_output_dir = output_base / metric
        metric_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for most/least difficult
        most_difficult_dir = metric_output_dir / "most_difficult"
        least_difficult_dir = metric_output_dir / "least_difficult" 
        most_difficult_dir.mkdir(exist_ok=True)
        least_difficult_dir.mkdir(exist_ok=True)
        
        # Get most difficult cases
        most_difficult = df.head(top_n)
        print(f"Selected top {len(most_difficult)} most difficult cases")
        
        # Get least difficult cases (prefer non-zero values)
        metric_col = f'{metric}_uncertainty' if metric == 'mean' else f'{metric}_uncertain{"ties" if metric == "sum" else ""}'
        if metric == 'sum':
            metric_col = 'sum_uncertainties'
        elif metric == 'count':
            metric_col = 'count_uncertain'
        else:
            metric_col = 'mean_uncertainty'
            
        non_zero_cases = df[df[metric_col] > 0]
        if len(non_zero_cases) >= top_n:
            least_difficult = non_zero_cases.tail(top_n)
            print(f"Selected bottom {len(least_difficult)} cases with non-zero {metric}")
        else:
            least_difficult = df.tail(top_n)
            print(f"Selected bottom {len(least_difficult)} cases (including zeros)")
        
        # Process most difficult cases
        print(f"\n--- Processing Most Difficult Cases ---")
        created_most = 0
        for idx, row in most_difficult.iterrows():
            case_id = row['case_id']
            source_file = Path(source_dir) / f"{case_id}.nii.gz"
            
            if source_file.exists():
                uint8_output = most_difficult_dir / f"{case_id}_uncertainty_mask_uint8.nii.gz"
                float_output = most_difficult_dir / f"{case_id}_uncertainty_mask_float.nii.gz"
                
                success, hist_path, stats = create_uncertainty_mask_with_certain(
                    source_file, uint8_output, float_output, case_id, most_difficult_dir
                )
                
                if success:
                    created_most += 1
                    total_files_created += 2  # uint8 + float
                    if hist_path:
                        all_histogram_paths.append(hist_path)
            else:
                print(f"WARNING: Source file not found: {source_file}")
        
        # Process least difficult cases  
        print(f"\n--- Processing Least Difficult Cases ---")
        created_least = 0
        for idx, row in least_difficult.iterrows():
            case_id = row['case_id']
            source_file = Path(source_dir) / f"{case_id}.nii.gz"
            
            if source_file.exists():
                uint8_output = least_difficult_dir / f"{case_id}_uncertainty_mask_uint8.nii.gz"
                float_output = least_difficult_dir / f"{case_id}_uncertainty_mask_float.nii.gz"
                
                success, hist_path, stats = create_uncertainty_mask_with_certain(
                    source_file, uint8_output, float_output, case_id, least_difficult_dir
                )
                
                if success:
                    created_least += 1 
                    total_files_created += 2  # uint8 + float
                    if hist_path:
                        all_histogram_paths.append(hist_path)
            else:
                print(f"WARNING: Source file not found: {source_file}")
        
        # Create metric-specific info file
        metric_info_file = metric_output_dir / f"{metric}_uncertainty_masks_info.txt"
        with open(metric_info_file, 'w') as f:
            f.write(f"Enhanced Uncertainty Masks - {metric.upper()} Ranking\n")
            f.write(f"={'='*50}\n\n")
            f.write(f"Ranking metric: {metric}\n")
            f.write(f"Source CSV: {csv_filename}\n")
            f.write(f"Total cases processed: {created_most + created_least}\n")
            f.write(f"  - Most difficult: {created_most} cases\n")
            f.write(f"  - Least difficult: {created_least} cases\n")
            f.write(f"Files per case: 2 mask files + 1 histogram\n")
            f.write(f"Total files created: {(created_most + created_least) * 3}\n\n")
            f.write(f"Mask types:\n")
            f.write(f"  - UINT8 masks (*_uint8.nii.gz): For ITK-SNAP visualization\n")
            f.write(f"  - FLOAT32 masks (*_float.nii.gz): For web-based viewers\n")
            f.write(f"  - Histograms (*_histogram.png): Uncertainty value distributions\n\n")
            f.write(f"Special feature: Certain foreground regions (probability=1) are\n")
            f.write(f"included in uncertainty masks with values slightly smaller than\n")
            f.write(f"the minimum uncertainty value found in each case.\n")
        
        print(f"\nMetric {metric} summary:")
        print(f"  Created {created_most} most difficult mask pairs")
        print(f"  Created {created_least} least difficult mask pairs") 
        print(f"  Total: {created_most + created_least} cases with masks + histograms")
    
    # Create overall summary
    overall_info_file = output_base / "enhanced_uncertainty_masks_summary.txt"
    with open(overall_info_file, 'w') as f:
        f.write(f"Enhanced Uncertainty Masks - Complete Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Generated uncertainty masks for 3 ranking metrics:\n")
        f.write(f"  1. Mean uncertainty ranking\n")
        f.write(f"  2. Sum of uncertainties ranking\n") 
        f.write(f"  3. Count of uncertain pixels ranking\n\n")
        f.write(f"Total mask files created: {total_files_created}\n")
        f.write(f"Total histogram files created: {len(all_histogram_paths)}\n")
        f.write(f"Colormap file: {colormap_path}\n\n")
        f.write(f"Directory structure:\n")
        f.write(f"  enhanced_uncertainty_masks/\n")
        f.write(f"  ├── mean/\n")
        f.write(f"  │   ├── most_difficult/\n")
        f.write(f"  │   └── least_difficult/\n")
        f.write(f"  ├── sum/\n")
        f.write(f"  │   ├── most_difficult/\n")
        f.write(f"  │   └── least_difficult/\n")
        f.write(f"  ├── count/\n")
        f.write(f"  │   ├── most_difficult/\n")
        f.write(f"  │   └── least_difficult/\n")
        f.write(f"  └── uncertainty_colormap.txt\n\n")
        f.write(f"Each case includes:\n")
        f.write(f"  - *_uncertainty_mask_uint8.nii.gz (for ITK-SNAP)\n")
        f.write(f"  - *_uncertainty_mask_float.nii.gz (for web viewers)\n")
        f.write(f"  - *_histogram.png (uncertainty distribution)\n")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Total mask files created: {total_files_created}")
    print(f"✓ Total histogram files created: {len(all_histogram_paths)}")
    print(f"✓ Processed all 3 ranking metrics")
    print(f"✓ Enhanced masks include certain foreground regions")
    print(f"✓ Output directory: {output_base}")
    print(f"✓ Overall summary: {overall_info_file}")
    
    return total_files_created, len(all_histogram_paths), str(output_base)

if __name__ == "__main__":
    print("Enhanced Uncertainty Mask Generator with Three Rankings")
    print("======================================================")
    print("Features:")
    print("- Processes all three ranking metrics (mean, sum, count)")
    print("- Includes certain foreground (prob=1) regions in uncertainty masks")
    print("- Generates histograms for each case")
    print("- Creates both uint8 and float32 versions")
    
    # Configuration
    base_directory = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    source_directory = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/labelsTr"
    
    print(f"\nConfiguration:")
    print(f"  Base directory: {base_directory}")
    print(f"  Source directory: {source_directory}")
    
    try:
        total_masks, total_histograms, output_dir = generate_enhanced_uncertainty_masks(
            base_dir=base_directory,
            source_dir=source_directory,
            top_n=10
        )
        
        print(f"\n🎉 SUCCESS! 🎉")
        print(f"Generated {total_masks} uncertainty mask files")
        print(f"Generated {total_histograms} histogram files")
        print(f"All files saved in: {output_dir}")
        print(f"\nNext steps:")
        print(f"- Use ITK-SNAP with the colormap for uint8 masks")
        print(f"- Use web viewer for float32 masks")
        print(f"- Review histograms to understand uncertainty distributions")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        print("Make sure to run the analysis script first to generate the ranking CSV files.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()