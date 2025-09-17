#!/usr/bin/env python3
"""
Enhanced Uncertainty Mask Generator
Creates both uint8 and float uncertainty masks with different modifiers for ITK-SNAP and web visualization
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path

def create_uncertainty_mask_uint8(probability_mask_path, output_path):
    """
    Create uint8 uncertainty mask for ITK-SNAP compatibility.
    Only uncertain regions (0 < probability < 1) will have non-zero values.
    """
    try:
        print(f"Processing uint8: {os.path.basename(probability_mask_path)}")
        
        # Load the probability mask
        nii_img = nib.load(probability_mask_path)
        prob_data = nii_img.get_fdata().astype(np.float32)
        
        # Find uncertain voxels (probability between 0 and 1)
        uncertain_mask = (prob_data > 0) & (prob_data < 1)
        
        # Initialize uint8 uncertainty data with zeros
        uncertainty_uint8 = np.zeros_like(prob_data, dtype=np.uint8)
        
        if np.sum(uncertain_mask) > 0:
            # Calculate uncertainty (1 - probability) for uncertain voxels
            uncertainty_values = 1.0 - prob_data[uncertain_mask]
            # Scale to [1, 255] range (0 reserved for background)
            uncertainty_uint8[uncertain_mask] = np.clip((uncertainty_values * 254) + 1, 1, 255).astype(np.uint8)
            
            print(f"  Uncertain voxels: {np.sum(uncertain_mask):,} out of {prob_data.size:,}")
            print(f"  Uncertainty range: [{uncertainty_uint8[uncertain_mask].min()}, {uncertainty_uint8[uncertain_mask].max()}]")
        else:
            print(f"  No uncertain voxels found (all values are 0 or 1)")
        
        # Create NIfTI image with uint8 data type
        uncertainty_nii = nib.Nifti1Image(uncertainty_uint8, nii_img.affine, nii_img.header)
        uncertainty_nii.header.set_data_dtype(np.uint8)
        
        # Save the uncertainty mask
        nib.save(uncertainty_nii, output_path)
        print(f"  Saved uint8: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing uint8 {probability_mask_path}: {str(e)}")
        return False

def create_uncertainty_mask_float(probability_mask_path, output_path):
    """
    Create float32 uncertainty mask for web-based visualization with smooth gradients.
    Preserves full uncertainty values as floating point numbers.
    """
    try:
        print(f"Processing float: {os.path.basename(probability_mask_path)}")
        
        # Load the probability mask
        nii_img = nib.load(probability_mask_path)
        prob_data = nii_img.get_fdata().astype(np.float32)
        
        # Find uncertain voxels (probability between 0 and 1)
        uncertain_mask = (prob_data > 0) & (prob_data < 1)
        
        # Initialize float32 uncertainty data with zeros
        uncertainty_float = np.zeros_like(prob_data, dtype=np.float32)
        
        if np.sum(uncertain_mask) > 0:
            # Calculate uncertainty (1 - probability) for uncertain voxels
            # Keep as float values between 0 and 1
            uncertainty_float[uncertain_mask] = 1.0 - prob_data[uncertain_mask]
            
            print(f"  Uncertain voxels: {np.sum(uncertain_mask):,} out of {prob_data.size:,}")
            print(f"  Uncertainty range: [{uncertainty_float[uncertain_mask].min():.4f}, {uncertainty_float[uncertain_mask].max():.4f}]")
        else:
            print(f"  No uncertain voxels found (all values are 0 or 1)")
        
        # Create NIfTI image with float32 data type
        uncertainty_nii = nib.Nifti1Image(uncertainty_float, nii_img.affine, nii_img.header)
        uncertainty_nii.header.set_data_dtype(np.float32)
        
        # Save the uncertainty mask
        nib.save(uncertainty_nii, output_path)
        print(f"  Saved float: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing float {probability_mask_path}: {str(e)}")
        return False

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

def generate_uncertainty_masks(csv_file_path, source_dir, output_dir, top_n=10):
    """
    Generate both uint8 and float32 uncertainty masks with different modifiers.
    
    Args:
        csv_file_path (str): Path to the EdgeUncertaintyMasks.csv file
        source_dir (str): Source directory containing probability masks
        output_dir (str): Output directory to save uncertainty masks
        top_n (int): Number of top and bottom cases to process
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different mask types
    most_difficult_dir = output_dir / "most_difficult_uncertainty_masks"
    least_difficult_dir = output_dir / "least_difficult_uncertainty_masks"
    most_difficult_dir.mkdir(exist_ok=True)
    least_difficult_dir.mkdir(exist_ok=True)
    
    # Create ITK-SNAP colormap
    colormap_path = output_dir / "uncertainty_colormap.txt"
    create_itksnap_colormap(colormap_path)
    
    # Read CSV and get cases
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} cases from CSV")
    
    # Get most difficult cases
    most_difficult = df.head(top_n)
    print(f"Selected top {len(most_difficult)} most difficult cases")
    
    # Get least difficult cases (non-zero uncertainty if possible)
    non_zero_cases = df[df['mean_uncertainty'] > 0]
    if len(non_zero_cases) >= top_n:
        least_difficult = non_zero_cases.tail(top_n)
        print(f"Selected bottom {len(least_difficult)} cases with non-zero uncertainty")
    else:
        least_difficult = df.tail(top_n)
        print(f"Selected bottom {len(least_difficult)} cases (including zero uncertainty)")
    
    # Process most difficult cases
    print(f"\n=== PROCESSING MOST DIFFICULT CASES ===")
    created_most = 0
    for idx, row in most_difficult.iterrows():
        case_id = row['case_id']
        source_file = Path(source_dir) / f"{case_id}.nii.gz"
        
        # Create both uint8 and float masks with descriptive names
        uint8_output = most_difficult_dir / f"{case_id}_uncertainty_mask_uint8.nii.gz"
        float_output = most_difficult_dir / f"{case_id}_uncertainty_mask_float.nii.gz"
        
        if source_file.exists():
            uint8_success = create_uncertainty_mask_uint8(source_file, uint8_output)
            float_success = create_uncertainty_mask_float(source_file, float_output)
            
            if uint8_success and float_success:
                created_most += 1
        else:
            print(f"WARNING: Source file not found: {source_file}")
    
    # Process least difficult cases
    print(f"\n=== PROCESSING LEAST DIFFICULT CASES ===")
    created_least = 0
    for idx, row in least_difficult.iterrows():
        case_id = row['case_id']
        source_file = Path(source_dir) / f"{case_id}.nii.gz"
        
        # Create both uint8 and float masks with descriptive names
        uint8_output = least_difficult_dir / f"{case_id}_uncertainty_mask_uint8.nii.gz"
        float_output = least_difficult_dir / f"{case_id}_uncertainty_mask_float.nii.gz"
        
        if source_file.exists():
            uint8_success = create_uncertainty_mask_uint8(source_file, uint8_output)
            float_success = create_uncertainty_mask_float(source_file, float_output)
            
            if uint8_success and float_success:
                created_least += 1
        else:
            print(f"WARNING: Source file not found: {source_file}")
    
    # Create info file
    info_file = output_dir / "uncertainty_masks_info.txt"
    with open(info_file, 'w') as f:
        f.write("Enhanced Uncertainty Masks Information\n")
        f.write("=====================================\n\n")
        f.write("Two types of uncertainty masks are created:\n\n")
        f.write("1. UINT8 MASKS (*_uint8.nii.gz):\n")
        f.write("   - Range: 0 (background/certain) to 255 (maximum uncertainty)\n")
        f.write("   - Optimized for ITK-SNAP visualization\n")
        f.write("   - Use with the provided colormap (.lut file)\n\n")
        f.write("2. FLOAT32 MASKS (*_float.nii.gz):\n")
        f.write("   - Range: 0.0 to 1.0 (continuous uncertainty values)\n")
        f.write("   - Optimized for web-based viewers with smooth gradients\n")
        f.write("   - Better precision for analysis and visualization\n\n")
        f.write("Only uncertain boundary regions have non-zero values in both types.\n\n")
        f.write(f"Files created:\n")
        f.write(f"- Most difficult cases: {created_most} case pairs in {most_difficult_dir}\n")
        f.write(f"- Least difficult cases: {created_least} case pairs in {least_difficult_dir}\n")
        f.write(f"- Each case has both uint8 and float32 versions\n")
        f.write(f"- Total mask files: {(created_most + created_least) * 2}\n")
        f.write(f"- Colormap for ITK-SNAP: {colormap_path}\n")
    
    print(f"\n=== SUMMARY ===")
    print(f"Created {created_most} mask pairs for most difficult cases")
    print(f"Created {created_least} mask pairs for least difficult cases")
    print(f"Total: {created_most + created_least} cases with both uint8 and float32 masks")
    print(f"Total files: {(created_most + created_least) * 2} uncertainty masks")
    print(f"Colormap: {colormap_path}")
    print(f"Info file: {info_file}")
    
    return (created_most + created_least) * 2, str(colormap_path)

if __name__ == "__main__":
    print("Enhanced Uncertainty Mask Generator")
    print("==================================")
    print("Creates both uint8 and float32 uncertainty masks for different viewers")
    
    # Configuration
    csv_file = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks/EdgeUncertaintyMasks.csv"
    source_directory = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/labelsTr"
    output_directory = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    
    print(f"\nConfiguration:")
    print(f"  CSV file: {csv_file}")
    print(f"  Source dir: {source_directory}")
    print(f"  Output dir: {output_directory}")
    
    try:
        total_masks, colormap_path = generate_uncertainty_masks(
            csv_file_path=csv_file,
            source_dir=source_directory,
            output_dir=output_directory,
            top_n=10
        )
        
        print(f"\n✓ Successfully generated {total_masks} uncertainty masks!")
        print(f"✓ Each case now has both uint8 (ITK-SNAP) and float32 (web) versions")
        print(f"✓ Use the web viewer script to visualize the float versions")
        print(f"✓ Use ITK-SNAP with the colormap file for uint8 versions")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        print("Make sure the CSV file and source directory exist.")
    except Exception as e:
        print(f"\n✗ Error: {e}")