#!/usr/bin/env python3
"""
Enhanced Web-based Uncertainty Viewer Launcher
Creates a local web server and launches browser-based uncertainty visualization
Supports both uint8 and float32 uncertainty masks
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import shutil
import tempfile

def find_matching_volume(case_id, images_dir="/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/imagesTr"):
    """Find the volume file for a given case ID."""
    volume_file = Path(images_dir) / f"{case_id}_0000.nii.gz"
    return volume_file if volume_file.exists() else None

def find_available_masks(base_dir="/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"):
    """Find all available uncertainty masks with both uint8 and float32 versions."""
    base_dir = Path(base_dir)
    
    most_difficult_dir = base_dir / "most_difficult_uncertainty_masks"
    least_difficult_dir = base_dir / "least_difficult_uncertainty_masks"
    
    available_masks = {}
    
    for masks_dir in [most_difficult_dir, least_difficult_dir]:
        if masks_dir.exists():
            # Look for both uint8 and float masks
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
    
    # Filter to only include cases that have at least one mask type
    available_masks = {k: v for k, v in available_masks.items() if v['masks']}
    
    return available_masks

def create_web_viewer_with_case(case_id, volume_path, mask_path, mask_type, difficulty, output_path):
    """Create a customized web viewer HTML file for a specific case and save to specified path."""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Uncertainty Viewer - {case_id} ({mask_type})</title>
    <script src="https://unpkg.com/niivue@0.44.0/dist/niivue.umd.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #2c3e50;
            color: white;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .case-info {{
            background: #e74c3c;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .case-info.least {{
            background: #27ae60;
        }}
        .mask-type-info {{
            background: #8e44ad;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            text-align: center;
        }}
        .mask-type-info.uint8 {{
            background: #e67e22;
        }}
        .controls {{
            background: #34495e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .control-group {{
            margin-bottom: 10px;
        }}
        label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        input[type="range"], select {{
            width: 100%;
            padding: 8px;
            border: none;
            border-radius: 4px;
            background: #2c3e50;
            color: white;
            border: 1px solid #7f8c8d;
        }}
        button {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }}
        button:hover {{
            background: #2980b9;
        }}
        #niivue-container {{
            height: 600px;
            border: 2px solid #7f8c8d;
            border-radius: 8px;
            overflow: hidden;
        }}
        .colorbar {{
            display: flex;
            align-items: center;
            margin: 20px 0;
            justify-content: center;
        }}
        .colorbar-gradient {{
            width: 400px;
            height: 30px;
            background: linear-gradient(to right, 
                rgba(0,0,255,1) 0%,      
                rgba(0,255,255,1) 25%,   
                rgba(255,255,0,1) 50%,   
                rgba(255,128,0,1) 75%,   
                rgba(255,0,0,1) 100%);   
            border: 2px solid #7f8c8d;
            border-radius: 5px;
            margin: 0 15px;
        }}
        .instruction {{
            background: #27ae60;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .file-info {{
            background: #34495e;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .stats {{
            background: #2c3e50;
            padding: 10px;
            border-radius: 5px;
            margin-top: 15px;
            border: 1px solid #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Uncertainty Visualization</h1>
            <h2>{case_id} - {mask_type_display} Mask</h2>
        </div>

        <div class="case-info {difficulty_class}">
            <h3>{case_id} - {difficulty} Difficult Case</h3>
            <p><strong>Volume:</strong> {volume_file}</p>
            <p><strong>Mask:</strong> {mask_file}</p>
        </div>

        <div class="mask-type-info {mask_type}">
            <strong>Mask Type:</strong> {mask_type_description}
        </div>

        <div class="instruction">
            <strong>Instructions:</strong> The files are automatically loaded. 
            Blue = low uncertainty, Red = high uncertainty. 
            Use mouse to navigate: drag to pan, scroll to change slices, right-click-drag to adjust window/level.
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="opacitySlider">Uncertainty Opacity: <span id="opacityValue">0.6</span></label>
                <input type="range" id="opacitySlider" min="0" max="1" step="0.05" value="0.6" oninput="updateOpacity()">
            </div>

            <div class="control-group">
                <label for="colormapSelect">Colormap:</label>
                <select id="colormapSelect" onchange="updateColormap()">
                    <option value="blue2red" selected>Blue to Red</option>
                    <option value="hot">Hot</option>
                    <option value="jet">Jet</option>
                    <option value="winter">Winter</option>
                    <option value="plasma">Plasma</option>
                </select>
            </div>

            <div class="control-group">
                <button onclick="resetView()">Reset View</button>
                <button onclick="toggleCrosshair()">Toggle Crosshair</button>
            </div>

            <div class="control-group">
                <button onclick="showStats()">Show Intensity Stats</button>
                <button onclick="captureScreenshot()">Capture View</button>
            </div>
        </div>

        <div id="niivue-container"></div>

        <div class="colorbar">
            <span>Low Uncertainty</span>
            <div class="colorbar-gradient"></div>
            <span>High Uncertainty</span>
        </div>

        <div class="file-info" id="file-info">
            <p><strong>Status:</strong> Loading files...</p>
            <div class="stats" id="stats-display" style="display: none;">
                <h4>Uncertainty Statistics:</h4>
                <p id="stats-content">Click "Show Intensity Stats" to display statistics</p>
            </div>
        </div>
    </div>

    <script>
        // Initialize NiiVue
        const nv = new Niivue.Niivue();
        nv.attachTo('niivue-container');
        
        // Configure NiiVue
        nv.opts.dragAndDropEnabled = false;
        nv.opts.backColor = [0.2, 0.2, 0.2, 1];
        nv.opts.crosshairColor = [1, 1, 1, 0.8];
        nv.opts.show3Dcrosshair = true;

        // Load the files automatically
        async function loadFiles() {{
            try {{
                const volumeList = [
                    {{
                        url: './volume.nii.gz',
                        name: '{case_id} Volume'
                    }},
                    {{
                        url: './mask.nii.gz',
                        name: '{case_id} Uncertainty ({mask_type})',
                        opacity: 0.6,
                        colormap: 'blue2red'
                    }}
                ];
                
                await nv.loadVolumes(volumeList);
                
                document.getElementById('file-info').innerHTML = 
                    '<p><strong>Status:</strong> Files loaded successfully! Navigate with mouse: drag to pan, scroll for slices.</p>' +
                    '<p><strong>Mask Type:</strong> {mask_type_description}</p>';
                
            }} catch (error) {{
                console.error('Error loading files:', error);
                document.getElementById('file-info').innerHTML = 
                    '<p><strong>Status:</strong> Error loading files: ' + error.message + '</p>';
            }}
        }}

        function updateOpacity() {{
            const slider = document.getElementById('opacitySlider');
            const value = slider.value;
            document.getElementById('opacityValue').textContent = value;
            
            if (nv.volumes.length > 1) {{
                nv.setOpacity(1, parseFloat(value));
            }}
        }}

        function updateColormap() {{
            const select = document.getElementById('colormapSelect');
            const colormap = select.value;
            
            if (nv.volumes.length > 1) {{
                nv.setColormap(1, colormap);
            }}
        }}

        function resetView() {{
            nv.scene.volScaleMultiplier = 1.0;
            nv.scene.crosshairPos = [0.5, 0.5, 0.5];
            nv.drawScene();
        }}

        function toggleCrosshair() {{
            nv.opts.show3Dcrosshair = !nv.opts.show3Dcrosshair;
            nv.drawScene();
        }}

        function showStats() {{
            if (nv.volumes.length > 1) {{
                const volume = nv.volumes[1];
                const statsDiv = document.getElementById('stats-display');
                const statsContent = document.getElementById('stats-content');
                
                // Calculate basic statistics
                let nonZeroVoxels = 0;
                let minVal = Infinity;
                let maxVal = -Infinity;
                let sum = 0;
                
                // Simple statistics calculation
                const img = volume.img;
                if (img) {{
                    for (let i = 0; i < img.length; i++) {{
                        if (img[i] > 0) {{
                            nonZeroVoxels++;
                            minVal = Math.min(minVal, img[i]);
                            maxVal = Math.max(maxVal, img[i]);
                            sum += img[i];
                        }}
                    }}
                    
                    const meanVal = nonZeroVoxels > 0 ? sum / nonZeroVoxels : 0;
                    
                    statsContent.innerHTML = 
                        `<strong>Non-zero voxels:</strong> ${{nonZeroVoxels.toLocaleString()}}<br>` +
                        `<strong>Min uncertainty:</strong> ${{minVal.toFixed(4)}}<br>` +
                        `<strong>Max uncertainty:</strong> ${{maxVal.toFixed(4)}}<br>` +
                        `<strong>Mean uncertainty:</strong> ${{meanVal.toFixed(4)}}`;
                    
                    statsDiv.style.display = 'block';
                }}
            }}
        }}

        function captureScreenshot() {{
            if (nv.volumes.length > 0) {{
                const canvas = nv.canvas;
                const link = document.createElement('a');
                link.download = '{case_id}_uncertainty_{mask_type}_screenshot.png';
                link.href = canvas.toDataURL();
                link.click();
            }}
        }}

        // Load files when page loads
        window.onload = function() {{
            setTimeout(loadFiles, 500);
        }};
    </script>
</body>
</html>'''

    # Determine mask type descriptions
    mask_type_descriptions = {
        'uint8': 'UINT8 (0-255) - Optimized for ITK-SNAP',
        'float': 'FLOAT32 (0.0-1.0) - High precision with smooth gradients'
    }
    
    # Format the HTML template
    difficulty_class = "least" if difficulty == "LEAST" else "most"
    difficulty_display = "Least" if difficulty == "LEAST" else "Most"
    
    html_content = html_template.format(
        case_id=case_id,
        mask_type=mask_type,
        mask_type_display=mask_type.upper(),
        mask_type_description=mask_type_descriptions.get(mask_type, f'{mask_type} mask'),
        difficulty_class=difficulty_class,
        difficulty=difficulty_display,
        volume_file=os.path.basename(volume_path),
        mask_file=os.path.basename(mask_path)
    )
    
    # Write HTML file to the specified path
    html_file = Path(output_path)
    html_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    return html_file

def start_web_server(directory, port=8000):
    """Start a simple HTTP server in the given directory."""
    
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        
        def log_message(self, format, *args):
            pass  # Suppress server logs
    
    server = HTTPServer(('localhost', port), Handler)
    print(f"Starting web server at http://localhost:{port}")
    
    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return server

def select_mask_type(available_mask_types):
    """Allow user to select which mask type to view."""
    if len(available_mask_types) == 1:
        return available_mask_types[0]
    
    print(f"\nAvailable mask types:")
    for i, mask_type in enumerate(available_mask_types, 1):
        description = "UINT8 (ITK-SNAP optimized)" if mask_type == 'uint8' else "FLOAT32 (High precision)"
        print(f"  {i}. {mask_type} - {description}")
    
    while True:
        try:
            choice = input(f"\nSelect mask type (1-{len(available_mask_types)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_mask_types):
                return available_mask_types[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(available_mask_types)}")
        except ValueError:
            print("Please enter a valid number")

def interactive_web_viewer():
    """Interactive uncertainty mask viewer using web browser."""
    
    # Configuration
    base_dir = "/data/ailab/hoda2/pseudo_labels_analysis/EdgeUncertaintyMasks"
    images_dir = "/data/ibd/data/nnunet/nnUNet_raw/Dataset001_IBD/imagesTr"
    html_output_path = "/data/ailab/hoda2/pseudo_labels_analysis/ViewUncertaintyMasks.html"
    
    print("Enhanced Web-based Uncertainty Mask Viewer")
    print("==========================================")
    
    # Find available masks
    available_masks = find_available_masks(base_dir)
    
    if not available_masks:
        print(f"No uncertainty masks found in {base_dir}")
        print("Run the mask generator script first!")
        return
    
    # Display available masks
    sorted_cases = sorted(available_masks.keys())
    print(f"\nFound {len(available_masks)} cases with uncertainty masks:")
    for i, case_id in enumerate(sorted_cases, 1):
        difficulty = available_masks[case_id]['difficulty']
        mask_types = list(available_masks[case_id]['masks'].keys())
        mask_types_str = ", ".join(mask_types)
        print(f"  {i:2d}. {case_id} ({difficulty} difficult) - [{mask_types_str}]")
    
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
    
    # Select mask type if multiple available
    available_mask_types = list(available_masks[selected_case]['masks'].keys())
    selected_mask_type = select_mask_type(available_mask_types)
    
    # Get file paths
    mask_info = available_masks[selected_case]
    uncertainty_mask_path = mask_info['masks'][selected_mask_type]
    difficulty = mask_info['difficulty']
    volume_path = find_matching_volume(selected_case, images_dir)
    
    if volume_path is None:
        print(f"Error: Volume file not found for case {selected_case}")
        return
    
    print(f"\n=== Launching Web Viewer for Case: {selected_case} ===")
    print(f"Volume: {volume_path}")
    print(f"Uncertainty mask: {uncertainty_mask_path}")
    print(f"Mask type: {selected_mask_type}")
    print(f"Difficulty: {difficulty}")
    
    # Create temporary directory for web server
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy files to temp directory with simple names
        volume_dest = temp_path / "volume.nii.gz"
        mask_dest = temp_path / "mask.nii.gz"
        
        print("Preparing files for web viewer...")
        shutil.copy2(volume_path, volume_dest)
        shutil.copy2(uncertainty_mask_path, mask_dest)
        
        # Create HTML file at the specified path
        html_file = create_web_viewer_with_case(
            selected_case, volume_path, uncertainty_mask_path, 
            selected_mask_type, difficulty, html_output_path
        )
        
        print(f"Created viewer HTML at: {html_file}")
        
        # Also create a copy in the temp directory for serving
        temp_html = temp_path / "index.html"
        shutil.copy2(html_file, temp_html)
        
        # Start web server
        server = start_web_server(str(temp_path), port=8000)
        
        # Open browser
        url = f"http://localhost:8000"
        print(f"Opening browser at {url}")
        webbrowser.open(url)
        
        print("\n" + "="*70)
        print("ENHANCED WEB VIEWER LAUNCHED")
        print("="*70)
        print(f"Case: {selected_case} ({difficulty} difficult)")
        print(f"Mask Type: {selected_mask_type}")
        print("The uncertainty viewer should open in your browser.")
        print("- Blue regions: Low uncertainty")
        print("- Red regions: High uncertainty") 
        print("- Mouse controls: drag to pan, scroll for slices")
        print("- Use controls to adjust opacity, colormap, and more")
        print("- HTML file saved to:", html_output_path)
        print("="*70)
        print("Press Enter to stop the server and exit...")
        
        try:
            input()
        except KeyboardInterrupt:
            pass
        
        print("Stopping web server...")

def view_specific_case(case_id, mask_type=None):
    """View a specific case by ID in web browser."""
    
    available_masks = find_available_masks()
    
    if case_id not in available_masks:
        print(f"Case {case_id} not found!")
        available_cases = sorted(available_masks.keys())
        print(f"Available cases: {', '.join(available_cases)}")
        return False
    
    # Check available mask types for this case
    available_mask_types = list(available_masks[case_id]['masks'].keys())
    
    if mask_type and mask_type not in available_mask_types:
        print(f"Mask type '{mask_type}' not available for case {case_id}")
        print(f"Available types: {', '.join(available_mask_types)}")
        return False
    
    if not mask_type:
        if len(available_mask_types) == 1:
            mask_type = available_mask_types[0]
        else:
            print(f"Multiple mask types available: {', '.join(available_mask_types)}")
            mask_type = select_mask_type(available_mask_types)
    
    uncertainty_mask_path = available_masks[case_id]['masks'][mask_type]
    difficulty = available_masks[case_id]['difficulty']
    volume_path = find_matching_volume(case_id)
    
    if volume_path is None:
        print(f"Volume file not found for case {case_id}")
        return False
    
    print(f"Launching web viewer for case: {case_id}")
    print(f"Mask type: {mask_type}")
    print(f"Difficulty: {difficulty}")
    
    # Use the same logic as interactive viewer but without user input
    html_output_path = "/data/ailab/hoda2/pseudo_labels_analysis/ViewUncertaintyMasks.html"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        volume_dest = temp_path / "volume.nii.gz"
        mask_dest = temp_path / "mask.nii.gz"
        
        shutil.copy2(volume_path, volume_dest)
        shutil.copy2(uncertainty_mask_path, mask_dest)
        
        html_file = create_web_viewer_with_case(
            case_id, volume_path, uncertainty_mask_path,
            mask_type, difficulty, html_output_path
        )
        
        # Copy to temp for serving
        temp_html = temp_path / "index.html"
        shutil.copy2(html_file, temp_html)
        
        server = start_web_server(str(temp_path), port=8000)
        webbrowser.open(f"http://localhost:8000")
        
        print(f"HTML file saved to: {html_file}")
        print("Press Enter to stop the server...")
        try:
            input()
        except KeyboardInterrupt:
            pass
    
    return True

if __name__ == "__main__":
    print("Enhanced Web-based Uncertainty Mask Viewer")
    print("==========================================")
    
    if len(sys.argv) > 1:
        case_id = sys.argv[1]
        mask_type = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Viewing specific case: {case_id}")
        if mask_type:
            print(f"Mask type: {mask_type}")
        success = view_specific_case(case_id, mask_type)
        if not success:
            print("Failed to view case")
    else:
        interactive_web_viewer()