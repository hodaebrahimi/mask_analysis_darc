#!/usr/bin/env python3
"""
Python-based Uncertainty Viewer
Offline viewer using matplotlib - no internet required
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path
import argparse

def find_matching_volume(case_id, images_dir="/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/imagesTr"):
    """Find the volume file for a given case ID."""
    volume_file = Path(images_dir) / f"{case_id}_0000.nii.gz"
    return volume_file if volume_file.exists() else None

def find_available_masks(base_dir="/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"):
    """Find all available uncertainty masks."""
    base_dir = Path(base_dir)
    
    most_difficult_dir = base_dir / "most_difficult_uncertainty_masks"
    least_difficult_dir = base_dir / "least_difficult_uncertainty_masks"
    
    available_masks = {}
    
    for masks_dir in [most_difficult_dir, least_difficult_dir]:
        if masks_dir.exists():
            for mask_type in ['uint8', 'float']:
                for mask_file in masks_dir.glob(f"*_uncertainty_mask_{mask_type}.nii.gz"):
                    case_id = mask_file.name.replace(f"_uncertainty_mask_{mask_type}.nii.gz", "")
                    difficulty = "MOST" if "most_difficult" in str(mask_file) else "LEAST"
                    
                    if case_id not in available_masks:
                        available_masks[case_id] = {
                            'difficulty': difficulty,
                            'masks': {}
                        }
                    
                    available_masks[case_id]['masks'][mask_type] = mask_file
    
    return {k: v for k, v in available_masks.items() if v['masks']}

class UncertaintyViewer:
    def __init__(self, volume_path, mask_path, case_id, mask_type, difficulty):
        self.volume_path = volume_path
        self.mask_path = mask_path
        self.case_id = case_id
        self.mask_type = mask_type
        self.difficulty = difficulty
        
        # Load data
        print(f"Loading volume: {volume_path}")
        self.volume_nii = nib.load(volume_path)
        self.volume_data = self.volume_nii.get_fdata()
        
        print(f"Loading mask: {mask_path}")
        self.mask_nii = nib.load(mask_path)
        self.mask_data = self.mask_nii.get_fdata()
        
        # Data info
        print(f"Volume shape: {self.volume_data.shape}")
        print(f"Volume range: {self.volume_data.min():.2f} - {self.volume_data.max():.2f}")
        print(f"Mask shape: {self.mask_data.shape}")
        print(f"Mask range: {self.mask_data.min():.4f} - {self.mask_data.max():.4f}")
        
        # Find non-zero mask voxels
        non_zero_mask = self.mask_data > 0
        if np.any(non_zero_mask):
            print(f"Uncertain voxels: {np.sum(non_zero_mask):,} ({100*np.sum(non_zero_mask)/self.mask_data.size:.2f}%)")
            print(f"Uncertainty range: {self.mask_data[non_zero_mask].min():.4f} - {self.mask_data[non_zero_mask].max():.4f}")
        else:
            print("No uncertain voxels found (all uncertainty values are 0)")
        
        # Initialize viewing parameters
        self.current_slice = self.volume_data.shape[2] // 2
        self.current_axis = 2  # axial view
        self.opacity = 0.7
        self.colormap = 'jet'
        self.uncertainty_threshold = 0.01
        
        # Setup the plot
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the matplotlib interactive plot."""
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Uncertainty Viewer - {self.case_id} ({self.difficulty} difficult, {self.mask_type} mask)', 
                         fontsize=14, fontweight='bold')
        
        # Main viewing axes
        ax_volume = plt.subplot(2, 3, 1)
        ax_mask = plt.subplot(2, 3, 2)
        ax_overlay = plt.subplot(2, 3, 3)
        
        # Statistics plot
        ax_stats = plt.subplot(2, 3, (4, 6))
        
        self.axes = {
            'volume': ax_volume,
            'mask': ax_mask,
            'overlay': ax_overlay,
            'stats': ax_stats
        }
        
        # Setup control sliders
        self.setup_controls()
        
        # Initial display
        self.update_display()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for controls
        
    def setup_controls(self):
        """Setup interactive controls."""
        
        # Slice slider
        ax_slice = plt.axes([0.1, 0.15, 0.5, 0.03])
        max_slices = self.volume_data.shape[self.current_axis] - 1
        self.slice_slider = Slider(ax_slice, 'Slice', 0, max_slices, 
                                  valinit=self.current_slice, valfmt='%d')
        self.slice_slider.on_changed(self.update_slice)
        
        # Opacity slider
        ax_opacity = plt.axes([0.1, 0.10, 0.5, 0.03])
        self.opacity_slider = Slider(ax_opacity, 'Opacity', 0, 1, 
                                   valinit=self.opacity, valfmt='%.2f')
        self.opacity_slider.on_changed(self.update_opacity)
        
        # Threshold slider
        ax_threshold = plt.axes([0.1, 0.05, 0.5, 0.03])
        self.threshold_slider = Slider(ax_threshold, 'Threshold', 0, 0.5, 
                                     valinit=self.uncertainty_threshold, valfmt='%.3f')
        self.threshold_slider.on_changed(self.update_threshold)
        
        # Colormap radio buttons
        ax_cmap = plt.axes([0.7, 0.05, 0.15, 0.15])
        self.cmap_radio = RadioButtons(ax_cmap, ['jet', 'hot', 'viridis', 'plasma', 'cool'])
        self.cmap_radio.on_clicked(self.update_colormap)
        
        # Axis radio buttons  
        ax_axis = plt.axes([0.85, 0.05, 0.1, 0.15])
        self.axis_radio = RadioButtons(ax_axis, ['Axial', 'Coronal', 'Sagittal'])
        self.axis_radio.on_clicked(self.update_axis)
    
    def get_slice_data(self):
        """Get current slice data based on selected axis."""
        
        if self.current_axis == 0:  # sagittal
            vol_slice = self.volume_data[self.current_slice, :, :].T
            mask_slice = self.mask_data[self.current_slice, :, :].T
        elif self.current_axis == 1:  # coronal
            vol_slice = self.volume_data[:, self.current_slice, :].T
            mask_slice = self.mask_data[:, self.current_slice, :].T
        else:  # axial (default)
            vol_slice = self.volume_data[:, :, self.current_slice].T
            mask_slice = self.mask_data[:, :, self.current_slice].T
        
        return vol_slice, mask_slice
    
    def update_display(self):
        """Update all displays."""
        
        # Get current slice data
        vol_slice, mask_slice = self.get_slice_data()
        
        # Clear axes
        for ax in self.axes.values():
            ax.clear()
        
        # Display volume
        self.axes['volume'].imshow(vol_slice, cmap='gray', origin='lower')
        self.axes['volume'].set_title('Volume')
        self.axes['volume'].axis('off')
        
        # Display uncertainty mask
        mask_display = np.ma.masked_where(mask_slice <= self.uncertainty_threshold, mask_slice)
        im_mask = self.axes['mask'].imshow(mask_display, cmap=self.colormap, 
                                         origin='lower', vmin=0, vmax=1)
        self.axes['mask'].set_title('Uncertainty Mask')
        self.axes['mask'].axis('off')
        
        # Display overlay
        self.axes['overlay'].imshow(vol_slice, cmap='gray', origin='lower')
        mask_overlay = np.ma.masked_where(mask_slice <= self.uncertainty_threshold, mask_slice)
        self.axes['overlay'].imshow(mask_overlay, cmap=self.colormap, alpha=self.opacity,
                                  origin='lower', vmin=0, vmax=1)
        self.axes['overlay'].set_title('Overlay')
        self.axes['overlay'].axis('off')
        
        # Update statistics
        self.update_statistics()
        
        # Add colorbar
        plt.colorbar(im_mask, ax=self.axes['mask'], fraction=0.046, pad=0.04)
        
        plt.draw()
    
    def update_statistics(self):
        """Update statistics display."""
        
        # Calculate statistics
        vol_slice, mask_slice = self.get_slice_data()
        
        # Overall statistics
        non_zero = mask_slice > 0
        above_threshold = mask_slice > self.uncertainty_threshold
        
        # Calculate mean safely
        mean_uncertainty = mask_slice[non_zero].mean() if np.any(non_zero) else 0.0
        
        stats_text = f"""
    Case: {self.case_id}
    Difficulty: {self.difficulty}
    Mask Type: {self.mask_type}
    Axis: {['Sagittal', 'Coronal', 'Axial'][self.current_axis]}
    Slice: {self.current_slice + 1}/{self.volume_data.shape[self.current_axis]}

    Volume Range: {self.volume_data.min():.2f} - {self.volume_data.max():.2f}
    Mask Range: {self.mask_data.min():.4f} - {self.mask_data.max():.4f}

    Current Slice:
    Total pixels: {vol_slice.size:,}
    Non-zero uncertainty: {np.sum(non_zero):,} ({100*np.sum(non_zero)/vol_slice.size:.1f}%)
    Above threshold: {np.sum(above_threshold):,} ({100*np.sum(above_threshold)/vol_slice.size:.1f}%)
    
    Slice uncertainty range: {mask_slice.min():.4f} - {mask_slice.max():.4f}
    Mean uncertainty: {mean_uncertainty:.4f}

    Controls:
    Arrow keys: Navigate slices
    A/C/S: Switch to Axial/Coronal/Sagittal
    Q: Quit viewer
    """
        
        self.axes['stats'].text(0.05, 0.95, stats_text, transform=self.axes['stats'].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=9)
        self.axes['stats'].set_xlim(0, 1)
        self.axes['stats'].set_ylim(0, 1)
        self.axes['stats'].axis('off')
    
    def update_slice(self, val):
        """Update slice from slider."""
        self.current_slice = int(self.slice_slider.val)
        self.update_display()
    
    def update_opacity(self, val):
        """Update opacity from slider."""
        self.opacity = self.opacity_slider.val
        self.update_display()
    
    def update_threshold(self, val):
        """Update uncertainty threshold from slider."""
        self.uncertainty_threshold = self.threshold_slider.val
        self.update_display()
    
    def update_colormap(self, label):
        """Update colormap from radio buttons."""
        self.colormap = label
        self.update_display()
    
    def update_axis(self, label):
        """Update viewing axis from radio buttons."""
        axis_map = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}
        self.current_axis = axis_map[label]
        
        # Update slice slider range
        max_slices = self.volume_data.shape[self.current_axis] - 1
        self.current_slice = max_slices // 2
        self.slice_slider.valmax = max_slices
        self.slice_slider.val = self.current_slice
        self.slice_slider.reset()
        
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        
        if event.key == 'right' or event.key == 'up':
            # Next slice
            if self.current_slice < self.volume_data.shape[self.current_axis] - 1:
                self.current_slice += 1
                self.slice_slider.set_val(self.current_slice)
        
        elif event.key == 'left' or event.key == 'down':
            # Previous slice
            if self.current_slice > 0:
                self.current_slice -= 1
                self.slice_slider.set_val(self.current_slice)
        
        elif event.key.lower() == 'a':
            # Axial view
            self.update_axis('Axial')
            
        elif event.key.lower() == 'c':
            # Coronal view
            self.update_axis('Coronal')
            
        elif event.key.lower() == 's':
            # Sagittal view
            self.update_axis('Sagittal')
            
        elif event.key.lower() == 'q':
            # Quit
            plt.close()
    
    def show(self):
        """Show the viewer."""
        print("\\n" + "="*60)
        print("PYTHON UNCERTAINTY VIEWER")
        print("="*60)
        print("Controls:")
        print("  Arrow keys: Navigate slices")
        print("  A/C/S: Switch to Axial/Coronal/Sagittal view")
        print("  Sliders: Adjust opacity, threshold, slice")
        print("  Radio buttons: Change colormap and axis")
        print("  Q: Quit")
        print("="*60)
        
        plt.show()

def select_case_interactive():
    """Interactive case selection."""
    
    available_masks = find_available_masks()
    
    if not available_masks:
        print("No uncertainty masks found!")
        return None, None, None
    
    # Display available masks
    sorted_cases = sorted(available_masks.keys())
    print(f"\\nFound {len(available_masks)} cases:")
    for i, case_id in enumerate(sorted_cases, 1):
        difficulty = available_masks[case_id]['difficulty']
        mask_types = list(available_masks[case_id]['masks'].keys())
        print(f"  {i:2d}. {case_id} ({difficulty}) - [{', '.join(mask_types)}]")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\\nSelect case (1-{len(sorted_cases)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None, None, None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sorted_cases):
                selected_case = sorted_cases[choice_idx]
                break
            else:
                print(f"Please enter 1-{len(sorted_cases)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Select mask type
    available_types = list(available_masks[selected_case]['masks'].keys())
    if len(available_types) == 1:
        mask_type = available_types[0]
    else:
        print(f"\\nAvailable mask types:")
        for i, mt in enumerate(available_types, 1):
            print(f"  {i}. {mt}")
        
        while True:
            try:
                choice = int(input(f"Select mask type (1-{len(available_types)}): ")) - 1
                if 0 <= choice < len(available_types):
                    mask_type = available_types[choice]
                    break
            except ValueError:
                pass
    
    return selected_case, mask_type, available_masks[selected_case]

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Python-based Uncertainty Viewer')
    parser.add_argument('--case', help='Case ID to view')
    parser.add_argument('--mask-type', choices=['uint8', 'float'], default='float',
                       help='Mask type to use')
    args = parser.parse_args()
    
    print("Python-based Uncertainty Viewer")
    print("==============================")
    print("Offline viewer using matplotlib")
    
    if args.case:
        # Direct case specification
        case_id = args.case
        mask_type = args.mask_type
        
        available_masks = find_available_masks()
        if case_id not in available_masks:
            print(f"Case {case_id} not found!")
            return
        
        if mask_type not in available_masks[case_id]['masks']:
            print(f"Mask type {mask_type} not available for {case_id}")
            return
            
        case_info = available_masks[case_id]
    else:
        # Interactive selection
        case_id, mask_type, case_info = select_case_interactive()
        if not case_id:
            print("No case selected")
            return
    
    # Get file paths
    mask_path = case_info['masks'][mask_type]
    difficulty = case_info['difficulty']
    volume_path = find_matching_volume(case_id)
    
    if not volume_path:
        print(f"Volume file not found for {case_id}")
        return
    
    print(f"\\nViewing case: {case_id}")
    print(f"Difficulty: {difficulty}")
    print(f"Mask type: {mask_type}")
    print(f"Volume: {volume_path}")
    print(f"Mask: {mask_path}")
    
    # Create and show viewer
    try:
        viewer = UncertaintyViewer(volume_path, mask_path, case_id, mask_type, difficulty)
        viewer.show()
    except Exception as e:
        print(f"Error creating viewer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()