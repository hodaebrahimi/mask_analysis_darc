#!/usr/bin/env python3
"""
ITK-SNAP Uncertainty Mask Viewer
Launches ITK-SNAP with uncertainty masks and proper colormap setup
"""

import os
import subprocess
import sys
from pathlib import Path

def find_matching_volume(case_id, images_dir="/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/imagesTr"):
    """Find the volume file for a given case ID."""
    volume_file = Path(images_dir) / f"{case_id}_0000.nii.gz"
    return volume_file if volume_file.exists() else None

def launch_itksnap_with_overlay(volume_path, uncertainty_mask_path, colormap_path, display=None):
    """
    Launch ITK-SNAP with proper segmentation loading and colormap instructions.
    """
    
    # ITK-SNAP command with volume as main image and uncertainty mask as segmentation
    cmd = ["itksnap", "-g", str(volume_path), "-s", str(uncertainty_mask_path)]
    
    print(f"Launching ITK-SNAP...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Volume: {volume_path}")
    print(f"Uncertainty segmentation: {uncertainty_mask_path}")
    print(f"Colormap: {colormap_path}")
    
    # Set up environment
    env = os.environ.copy()
    if display:
        env['DISPLAY'] = display
        print(f"Using DISPLAY: {display}")
    elif 'DISPLAY' in env:
        print(f"Using DISPLAY: {env['DISPLAY']}")
    else:
        print("Warning: No DISPLAY variable set")
    
    try:
        # Launch ITK-SNAP
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        print(f"\nITK-SNAP launched (PID: {process.pid})")
        
        # Print detailed setup instructions
        print(f"\n" + "="*60)
        print(f"ITK-SNAP SETUP INSTRUCTIONS")
        print(f"="*60)
        print(f"1. Wait for ITK-SNAP to fully load")
        print(f"2. The uncertainty mask should load as a SEGMENTATION (not overlay)")
        print(f"3. To apply the custom colormap:")
        print(f"   a. Go to Segmentation → Label Editor")
        print(f"   b. Click 'Load Label Descriptions' button")
        print(f"   c. Navigate to and select: {colormap_path}")
        print(f"   d. Click 'Load'")
        print(f"4. Adjust segmentation opacity:")
        print(f"   a. Go to View → Segmentation Display Options")
        print(f"   b. Set '3D Opacity' to 60-70%")
        print(f"   c. Set '2D Opacity' to 60-70%")
        print(f"5. You should now see:")
        print(f"   - Blue regions: Low uncertainty (labels ~25-76)")
        print(f"   - Red regions: High uncertainty (labels ~178-255)")
        print(f"   - Transparent: Background (label 0)")
        print(f"6. The segmentation will show uncertainty boundaries with proper labels")
        print(f"="*60)
        
        return process
        
    except FileNotFoundError:
        print("\nError: ITK-SNAP not found!")
        print("Make sure ITK-SNAP is installed: sudo apt-get install itksnap")
        return None
    except Exception as e:
        print(f"\nError launching ITK-SNAP: {e}")
        return None

def find_available_masks(base_dir="/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"):
    """Find all available uncertainty masks."""
    base_dir = Path(base_dir)
    
    # Look in both directories
    most_difficult_dir = base_dir / "most_difficult_uncertainty_masks"
    least_difficult_dir = base_dir / "least_difficult_uncertainty_masks"
    
    available_masks = {}
    
    for masks_dir in [most_difficult_dir, least_difficult_dir]:
        if masks_dir.exists():
            for mask_file in masks_dir.glob("*_uncertainty_mask_uint8.nii.gz"):
                case_id = mask_file.name.replace("_uncertainty_mask_uint8.nii.gz", "")
                difficulty = "MOST" if "most_difficult" in str(mask_file) else "LEAST"
                available_masks[case_id] = {
                    'path': mask_file, 
                    'difficulty': difficulty
                }
    
    return available_masks

def interactive_viewer():
    """Interactive uncertainty mask viewer."""
    
    # Configuration
    base_dir = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    images_dir = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/imagesTr"
    colormap_path = Path(base_dir) / "uncertainty_colormap.txt"
    
    print(f"ITK-SNAP Uncertainty Mask Viewer")
    print(f"================================")
    
    # Check if colormap exists
    if not colormap_path.exists():
        print(f"Error: Colormap not found at {colormap_path}")
        print(f"Run the mask generator script first!")
        return
    
    # Find available masks
    available_masks = find_available_masks(base_dir)
    
    if not available_masks:
        print(f"No uncertainty masks found in {base_dir}")
        print(f"Run the mask generator script first!")
        return
    
    # Display available masks
    sorted_cases = sorted(available_masks.keys())
    print(f"\nFound {len(available_masks)} uncertainty masks:")
    for i, case_id in enumerate(sorted_cases, 1):
        difficulty = available_masks[case_id]['difficulty']
        print(f"  {i:2d}. {case_id} ({difficulty} difficult)")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect a case to view (1-{len(sorted_cases)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Goodbye!")
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sorted_cases):
                selected_case = sorted_cases[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(sorted_cases)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    # Get file paths
    uncertainty_mask_path = available_masks[selected_case]['path']
    volume_path = find_matching_volume(selected_case, images_dir)
    
    if volume_path is None:
        print(f"Error: Volume file not found for case {selected_case}")
        print(f"Expected: {images_dir}/{selected_case}_0000.nii.gz")
        return
    
    print(f"\n=== Viewing Case: {selected_case} ===")
    print(f"Difficulty: {available_masks[selected_case]['difficulty']}")
    
    # Check for X11 display
    current_display = os.environ.get('DISPLAY', None)
    if not current_display:
        print("Warning: No DISPLAY variable found.")
        user_display = input("Enter DISPLAY value (or press Enter to continue): ").strip()
        if user_display:
            current_display = user_display
    
    # Launch ITK-SNAP
    process = launch_itksnap_with_overlay(
        volume_path=volume_path,
        uncertainty_mask_path=uncertainty_mask_path,
        colormap_path=str(colormap_path),
        display=current_display
    )
    
    if process:
        print(f"\nPress Enter when done viewing (or Ctrl+C to exit)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nExiting...")

def view_specific_case(case_id):
    """View a specific case by ID."""
    
    # Configuration  
    base_dir = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    images_dir = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/imagesTr"
    colormap_path = Path(base_dir) / "uncertainty_colormap.lut"
    
    # Find the uncertainty mask
    available_masks = find_available_masks(base_dir)
    
    if case_id not in available_masks:
        print(f"Case {case_id} not found!")
        print(f"Available cases: {', '.join(sorted(available_masks.keys()))}")
        return False
    
    # Get file paths
    uncertainty_mask_path = available_masks[case_id]['path']
    volume_path = find_matching_volume(case_id, images_dir)
    
    if volume_path is None:
        print(f"Volume file not found for case {case_id}")
        return False
    
    if not colormap_path.exists():
        print(f"Colormap not found: {colormap_path}")
        return False
    
    print(f"Viewing case: {case_id}")
    print(f"Difficulty: {available_masks[case_id]['difficulty']}")
    
    # Launch ITK-SNAP
    process = launch_itksnap_with_overlay(
        volume_path=volume_path,
        uncertainty_mask_path=uncertainty_mask_path,
        colormap_path=str(colormap_path)
    )
    
    return process is not None

if __name__ == "__main__":
    print("ITK-SNAP Uncertainty Mask Viewer")
    print("=================================")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        case_id = sys.argv[1]
        print(f"Viewing specific case: {case_id}")
        success = view_specific_case(case_id)
        if not success:
            print("Failed to view case")
    else:
        # Interactive mode
        interactive_viewer()
    
    print(f"\n=== USAGE NOTES ===")
    print(f"For X11 forwarding from VS Code to Window app:")
    print(f"1. In Window app: ssh -X username@server")
    print(f"2. Check: echo $DISPLAY")
    print(f"3. In VS Code: export DISPLAY=localhost:10.0")
    print(f"4. Run this script")