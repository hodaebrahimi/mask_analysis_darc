#!/usr/bin/env python3
"""
3D Python Uncertainty Viewer
Interactive 3D visualization of uncertainty masks
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, Button
from pathlib import Path
import argparse
from skimage import measure

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

class Uncertainty3DViewer:
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
        print(f"Mask range: {self.mask_data.min():.4f} - {self.mask_data.max():.4f}")
        
        # Find non-zero mask voxels
        non_zero_mask = self.mask_data > 0
        if np.any(non_zero_mask):
            print(f"Uncertain voxels: {np.sum(non_zero_mask):,} ({100*np.sum(non_zero_mask)/self.mask_data.size:.2f}%)")
        
        # Initialize parameters
        self.uncertainty_threshold = 0.1
        self.opacity = 0.7
        self.colormap = 'jet'
        self.downsample_factor = 4  # Reduce data for 3D performance
        self.volume_outline = True
        self.show_volume_slices = True
        
        # Downsample for 3D performance
        self.prepare_3d_data()
        
        # Setup plot
        self.setup_3d_plot()
    
    def prepare_3d_data(self):
        """Prepare downsampled data for 3D visualization."""
        
        print("Preparing 3D data (downsampling for performance)...")
        
        # Downsample both volume and mask
        step = self.downsample_factor
        self.volume_3d = self.volume_data[::step, ::step, ::step]
        self.mask_3d = self.mask_data[::step, ::step, ::step]
        
        print(f"3D data shape: {self.volume_3d.shape}")
        
        # Create coordinate grids
        z, y, x = np.mgrid[0:self.volume_3d.shape[0], 
                          0:self.volume_3d.shape[1], 
                          0:self.volume_3d.shape[2]]
        
        self.coords_3d = (x, y, z)
        
        # Find volume outline using marching cubes on thresholded volume
        try:
            # Threshold volume to find tissue boundary
            vol_threshold = np.percentile(self.volume_data[self.volume_data > self.volume_data.min()], 20)
            vol_binary = self.volume_data > vol_threshold
            
            # Downsample binary volume
            vol_binary_3d = vol_binary[::step, ::step, ::step]
            
            # Generate mesh using marching cubes
            self.volume_verts, self.volume_faces, _, _ = measure.marching_cubes(
                vol_binary_3d, level=0.5, spacing=(step, step, step)
            )
            
            print(f"Volume mesh: {len(self.volume_verts)} vertices, {len(self.volume_faces)} faces")
            
        except Exception as e:
            print(f"Could not generate volume mesh: {e}")
            self.volume_verts = None
            self.volume_faces = None
    
    def setup_3d_plot(self):
        """Setup 3D matplotlib plot."""
        
        # Create figure with 3D subplot
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle(f'3D Uncertainty Viewer - {self.case_id} ({self.difficulty} difficult, {self.mask_type} mask)', 
                         fontsize=14, fontweight='bold')
        
        # Main 3D plot
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        
        # Setup controls
        self.setup_3d_controls()
        
        # Initial display
        self.update_3d_display()
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
    
    def setup_3d_controls(self):
        """Setup 3D viewer controls."""
        
        # Threshold slider
        ax_threshold = plt.axes([0.1, 0.08, 0.3, 0.03])
        self.threshold_slider = Slider(ax_threshold, 'Uncertainty Threshold', 0, 0.8, 
                                     valinit=self.uncertainty_threshold, valfmt='%.3f')
        self.threshold_slider.on_changed(self.update_threshold)
        
        # Opacity slider
        ax_opacity = plt.axes([0.1, 0.04, 0.3, 0.03])
        self.opacity_slider = Slider(ax_opacity, 'Opacity', 0.1, 1.0, 
                                   valinit=self.opacity, valfmt='%.2f')
        self.opacity_slider.on_changed(self.update_opacity)
        
        # Downsample slider
        ax_downsample = plt.axes([0.1, 0.00, 0.3, 0.03])
        self.downsample_slider = Slider(ax_downsample, 'Detail Level', 2, 8, 
                                      valinit=self.downsample_factor, valfmt='%d')
        self.downsample_slider.on_changed(self.update_downsample)
        
        # Colormap buttons
        ax_cmap = plt.axes([0.5, 0.02, 0.15, 0.1])
        self.cmap_radio = RadioButtons(ax_cmap, ['jet', 'hot', 'viridis', 'plasma', 'cool'])
        self.cmap_radio.on_clicked(self.update_colormap)
        
        # Toggle buttons
        ax_outline = plt.axes([0.7, 0.08, 0.08, 0.04])
        self.outline_button = Button(ax_outline, 'Volume\nOutline')
        self.outline_button.on_clicked(self.toggle_volume_outline)
        
        ax_slices = plt.axes([0.7, 0.02, 0.08, 0.04])
        self.slices_button = Button(ax_slices, 'Volume\nSlices')
        self.slices_button.on_clicked(self.toggle_volume_slices)
        
        # Rotate button
        ax_rotate = plt.axes([0.85, 0.05, 0.08, 0.04])
        self.rotate_button = Button(ax_rotate, 'Auto\nRotate')
        self.rotate_button.on_clicked(self.toggle_rotation)
        self.rotating = False
    
    def update_3d_display(self):
        """Update 3D display."""
        
        # Clear the 3D axis
        self.ax3d.clear()
        
        # Get uncertainty data above threshold
        uncertainty_mask = self.mask_3d > self.uncertainty_threshold
        
        if np.any(uncertainty_mask):
            # Get coordinates of uncertain voxels
            x, y, z = self.coords_3d
            x_unc = x[uncertainty_mask]
            y_unc = y[uncertainty_mask]
            z_unc = z[uncertainty_mask]
            colors = self.mask_3d[uncertainty_mask]
            
            # Create 3D scatter plot
            scatter = self.ax3d.scatter(x_unc, y_unc, z_unc, 
                                      c=colors, cmap=self.colormap, 
                                      alpha=self.opacity, s=20,
                                      vmin=0, vmax=1)
            
            # Add colorbar
            if hasattr(self, 'colorbar'):
                self.colorbar.remove()
            self.colorbar = plt.colorbar(scatter, ax=self.ax3d, shrink=0.6, aspect=20)
            self.colorbar.set_label('Uncertainty', rotation=270, labelpad=15)
        
        # Add volume outline if enabled
        if self.volume_outline and self.volume_verts is not None:
            # Sample subset of faces for performance
            n_faces = min(1000, len(self.volume_faces))
            face_indices = np.random.choice(len(self.volume_faces), n_faces, replace=False)
            
            for i in face_indices:
                face = self.volume_faces[i]
                vertices = self.volume_verts[face]
                
                # Create triangular face
                x_face = [vertices[j][0] for j in [0, 1, 2, 0]]
                y_face = [vertices[j][1] for j in [0, 1, 2, 0]]
                z_face = [vertices[j][2] for j in [0, 1, 2, 0]]
                
                self.ax3d.plot(x_face, y_face, z_face, 'k-', alpha=0.1, linewidth=0.5)
        
        # Add volume slices if enabled
        if self.show_volume_slices:
            self.add_volume_slices()
        
        # Set labels and title
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title(f'3D Uncertainty Distribution\nThreshold: {self.uncertainty_threshold:.3f}')
        
        # Set equal aspect ratio
        max_range = max(self.volume_3d.shape) // 2
        mid_x, mid_y, mid_z = np.array(self.volume_3d.shape) // 2
        self.ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add statistics
        uncertain_voxels = np.sum(self.mask_3d > self.uncertainty_threshold)
        total_voxels = self.mask_3d.size
        
        stats_text = f"""
Case: {self.case_id} ({self.difficulty})
Mask Type: {self.mask_type}
Uncertain Voxels: {uncertain_voxels:,} ({100*uncertain_voxels/total_voxels:.2f}%)
Threshold: {self.uncertainty_threshold:.3f}
3D Shape: {self.volume_3d.shape}
Downsample: {self.downsample_factor}x

Controls:
R: Reset view  |  Space: Toggle rotation
Mouse: Rotate and zoom view
"""
        
        self.ax3d.text2D(0.02, 0.98, stats_text, transform=self.ax3d.transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.draw()
    
    def add_volume_slices(self):
        """Add representative volume slices to 3D plot."""
        
        # Add three orthogonal slices through the middle
        mid_x, mid_y, mid_z = np.array(self.volume_3d.shape) // 2
        
        # Sagittal slice (YZ plane)
        if mid_x < self.volume_3d.shape[0]:
            Y, Z = np.meshgrid(range(self.volume_3d.shape[1]), range(self.volume_3d.shape[2]))
            X = np.full_like(Y, mid_x)
            slice_data = self.volume_3d[mid_x, :, :].T
            self.ax3d.plot_surface(X, Y, Z, facecolors=plt.cm.gray(slice_data/slice_data.max()), 
                                 alpha=0.3, shade=False)
        
        # Coronal slice (XZ plane)  
        if mid_y < self.volume_3d.shape[1]:
            X, Z = np.meshgrid(range(self.volume_3d.shape[0]), range(self.volume_3d.shape[2]))
            Y = np.full_like(X, mid_y)
            slice_data = self.volume_3d[:, mid_y, :].T
            self.ax3d.plot_surface(X, Y, Z, facecolors=plt.cm.gray(slice_data/slice_data.max()), 
                                 alpha=0.3, shade=False)
    
    def update_threshold(self, val):
        """Update uncertainty threshold."""
        self.uncertainty_threshold = self.threshold_slider.val
        self.update_3d_display()
    
    def update_opacity(self, val):
        """Update opacity."""
        self.opacity = self.opacity_slider.val
        self.update_3d_display()
    
    def update_colormap(self, label):
        """Update colormap."""
        self.colormap = label
        self.update_3d_display()
    
    def update_downsample(self, val):
        """Update downsample factor and regenerate 3D data."""
        self.downsample_factor = int(self.downsample_slider.val)
        print(f"Updating detail level (downsample factor: {self.downsample_factor})...")
        self.prepare_3d_data()
        self.update_3d_display()
    
    def toggle_volume_outline(self, event):
        """Toggle volume outline."""
        self.volume_outline = not self.volume_outline
        self.update_3d_display()
    
    def toggle_volume_slices(self, event):
        """Toggle volume slices."""
        self.show_volume_slices = not self.show_volume_slices
        self.update_3d_display()
    
    def toggle_rotation(self, event):
        """Toggle auto-rotation."""
        self.rotating = not self.rotating
        if self.rotating:
            self.start_rotation()
    
    def start_rotation(self):
        """Start auto-rotation animation."""
        def animate():
            if self.rotating:
                self.ax3d.view_init(azim=self.ax3d.azim + 2)
                plt.draw()
                self.fig.canvas.flush_events()
                self.fig.after(50, animate)
        
        animate()
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        
        if event.key == 'r':
            # Reset view
            self.ax3d.view_init()
            plt.draw()
            
        elif event.key == ' ':
            # Toggle rotation
            self.toggle_rotation(None)
            
        elif event.key == 'q':
            # Quit
            plt.close()
    
    def show(self):
        """Show the 3D viewer."""
        print("\n" + "="*60)
        print("3D UNCERTAINTY VIEWER")
        print("="*60)
        print("Controls:")
        print("  Mouse: Rotate and zoom the 3D view")
        print("  Sliders: Adjust threshold, opacity, detail level")
        print("  Buttons: Toggle volume outline/slices, auto-rotate")
        print("  R: Reset view")
        print("  Space: Toggle auto-rotation")  
        print("  Q: Quit")
        print("="*60)
        print("Note: Higher detail levels (lower downsample) = slower performance")
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
    print(f"\nFound {len(available_masks)} cases:")
    for i, case_id in enumerate(sorted_cases, 1):
        difficulty = available_masks[case_id]['difficulty']
        mask_types = list(available_masks[case_id]['masks'].keys())
        print(f"  {i:2d}. {case_id} ({difficulty}) - [{', '.join(mask_types)}]")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect case (1-{len(sorted_cases)}) or 'q' to quit: ").strip()
            
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
        print(f"\nAvailable mask types:")
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
    
    parser = argparse.ArgumentParser(description='3D Python Uncertainty Viewer')
    parser.add_argument('--case', help='Case ID to view')
    parser.add_argument('--mask-type', choices=['uint8', 'float'], default='float',
                       help='Mask type to use')
    args = parser.parse_args()
    
    print("3D Python Uncertainty Viewer")
    print("============================")
    print("Interactive 3D visualization with matplotlib")
    
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
    
    print(f"\nViewing case: {case_id}")
    print(f"Difficulty: {difficulty}")
    print(f"Mask type: {mask_type}")
    print(f"Volume: {volume_path}")
    print(f"Mask: {mask_path}")
    
    # Create and show viewer
    try:
        viewer = Uncertainty3DViewer(volume_path, mask_path, case_id, mask_type, difficulty)
        viewer.show()
    except Exception as e:
        print(f"Error creating 3D viewer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()