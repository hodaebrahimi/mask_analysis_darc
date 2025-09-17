#!/bin/bash

echo "🔍 GPU Selection for Medical Viewer"
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader

read -p "Enter GPU ID to use (or press Enter for default): " gpu_id

if [ -n "$gpu_id" ]; then
    # Validate GPU ID
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ "$gpu_id" -ge 0 ] && [ "$gpu_id" -lt "$gpu_count" ]; then
        echo "🚀 Using GPU $gpu_id: $(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)"
        GPU_SELECTION=$gpu_id
    else
        echo "❌ Invalid GPU ID. Using default configuration."
        GPU_SELECTION=""
    fi
else
    echo "🔧 Using default GPU configuration"
    GPU_SELECTION=""
fi

# Setup script for offline Medical Image Uncertainty Viewer
# Run this script to download NiiVue library and create a self-contained viewer

echo "🚀 Setting up offline Medical Image Uncertainty Viewer..."

# Create directory structure
mkdir -p uncertainty_viewer
cd uncertainty_viewer

# Download NiiVue library with proper redirect following
echo "📥 Downloading NiiVue library..."
curl -L -o niivue.umd.js "https://cdn.jsdelivr.net/npm/@niivue/niivue@latest/dist/niivue.umd.js"

if [ $? -eq 0 ]; then
    # Verify the file is actually JavaScript and not an HTML redirect
    if head -c 50 niivue.umd.js | grep -q "function\|!function\|var\|const\|let"; then
        echo "✅ NiiVue library downloaded successfully"
        echo "📊 File size: $(ls -lh niivue.umd.js | awk '{print $5}')"
    else
        echo "❌ Downloaded file appears to be invalid (HTML redirect page)"
        echo "🔄 Trying alternative download method..."
        curl -L -o niivue.umd.js "https://unpkg.com/@niivue/niivue@latest/dist/niivue.umd.js"
        
        if head -c 50 niivue.umd.js | grep -q "function\|!function\|var\|const\|let"; then
            echo "✅ Alternative download successful"
        else
            echo "❌ Failed to download valid NiiVue library"
            echo "Please try downloading manually from:"
            echo "https://cdn.jsdelivr.net/npm/@niivue/niivue@latest/dist/niivue.umd.js"
            exit 1
        fi
    fi
else
    echo "❌ Failed to download NiiVue library"
    echo "Please check your internet connection or download manually"
    exit 1
fi

# Create the HTML file with LOCAL script reference (not CDN)
cat > uncertainty_viewer.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Image Uncertainty Viewer (Offline)</title>
    <script src="./niivue.umd.js"></script>
    <style>
        * {
            box-sizing: border-box;
        }
        
        body {
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .file-upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .file-upload-box {
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 30px 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .file-upload-box input[type="file"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        
        .file-status {
            margin-top: 10px;
            padding: 8px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .file-status.success {
            background: rgba(76, 175, 80, 0.2);
            color: #4CAF50;
        }
        
        .file-status.error {
            background: rgba(244, 67, 54, 0.2);
            color: #f44336;
        }
        
        .controls-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }
        
        .control-group {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
        }
        
        .control-item {
            margin-bottom: 15px;
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.2);
            outline: none;
            -webkit-appearance: none;
        }
        
        select, button {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 14px;
            cursor: pointer;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .viewer-container {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        #niivue-container {
            height: 600px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            position: relative;
        }
        
        .status-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .status {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }
        
        .status.loading {
            background: rgba(33, 150, 243, 0.1);
            border-left-color: #2196F3;
        }
        
        .status.success {
            background: rgba(76, 175, 80, 0.1);
            border-left-color: #4CAF50;
        }
        
        .status.error {
            background: rgba(244, 67, 54, 0.1);
            border-left-color: #f44336;
        }
        
        .debug-panel {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            z-index: 10;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .offline-indicator {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid rgba(76, 175, 80, 0.5);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            color: #4CAF50;
        }

        .library-status {
            background: rgba(255, 193, 7, 0.2);
            border: 1px solid rgba(255, 193, 7, 0.5);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            color: #FFC107;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Medical Image Uncertainty Viewer</h1>
            <p>Advanced visualization tool for medical volumes and uncertainty masks</p>
            <div class="offline-indicator">
                ✅ Offline Mode - No internet connection required
            </div>
            <div class="library-status" id="library-status">
                🔄 Checking NiiVue library...
            </div>
        </div>

        <div class="upload-section">
            <h3>Load Medical Files</h3>
            <div class="file-upload-grid">
                <div class="file-upload-box">
                    <input type="file" id="volume-file" accept=".nii.gz,.nii" />
                    <h4>Volume File</h4>
                    <p>Upload main medical image<br>(.nii.gz or .nii)</p>
                    <div class="file-status" id="volume-status"></div>
                </div>
                
                <div class="file-upload-box">
                    <input type="file" id="mask-file" accept=".nii.gz,.nii" />
                    <h4>Uncertainty Mask</h4>
                    <p>Upload uncertainty segmentation<br>(.nii.gz or .nii)</p>
                    <div class="file-status" id="mask-status"></div>
                </div>
            </div>
            
            <div style="text-align: center; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <button id="load-btn" onclick="loadFiles()" disabled>
                    Load and Visualize
                </button>
                <button onclick="debugFileState()" style="background: rgba(255,193,7,0.3);">
                    Debug State
                </button>
            </div>
        </div>

        <div class="controls-panel">
            <div class="controls-grid">
                <div class="control-group">
                    <h4>Overlay Settings</h4>
                    <div class="control-item">
                        <label>Opacity: <span id="opacity-value">0.7</span></label>
                        <input type="range" id="opacity-slider" min="0" max="1" step="0.1" value="0.7" onchange="updateOpacity()" />
                    </div>
                    <div class="control-item">
                        <select id="colormap-select" onchange="updateColormap()">
                            <option value="blue2red" selected>Blue to Red</option>
                            <option value="red">Red</option>
                            <option value="hot">Hot</option>
                            <option value="jet">Jet</option>
                        </select>
                    </div>
                </div>

                <div class="control-group">
                    <h4>View Controls</h4>
                    <div class="control-item">
                        <button onclick="resetView()">Reset View</button>
                    </div>
                    <div class="control-item">
                        <button onclick="showStatistics()">Show Statistics</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="viewer-container">
            <div id="niivue-container">
                <div class="loading-overlay" id="loading-overlay" style="display: none;">
                    <div class="loading-spinner"></div>
                </div>
            </div>
        </div>

        <div class="status-panel">
            <div class="status loading" id="main-status">
                <strong>Status:</strong> Ready to load files.
            </div>
            
            <div class="debug-panel" id="debug-panel">
                Debug log will appear here...
            </div>
        </div>
    </div>

    <script>
        let nv = null;
        let volumeFile = null;
        let maskFile = null;
        let filesLoaded = false;

        function debugLog(message) {
            const timestamp = new Date().toLocaleTimeString();
            const debugPanel = document.getElementById('debug-panel');
            debugPanel.innerHTML += `[${timestamp}] ${message}<br>`;
            debugPanel.scrollTop = debugPanel.scrollHeight;
            console.log(`[DEBUG] ${message}`);
        }

        function updateLibraryStatus(message, type = 'loading') {
            const statusElement = document.getElementById('library-status');
            statusElement.innerHTML = message;
            
            // Update styling based on type
            statusElement.className = 'library-status';
            if (type === 'success') {
                statusElement.style.background = 'rgba(76, 175, 80, 0.2)';
                statusElement.style.borderColor = 'rgba(76, 175, 80, 0.5)';
                statusElement.style.color = '#4CAF50';
            } else if (type === 'error') {
                statusElement.style.background = 'rgba(244, 67, 54, 0.2)';
                statusElement.style.borderColor = 'rgba(244, 67, 54, 0.5)';
                statusElement.style.color = '#f44336';
            } else {
                statusElement.style.background = 'rgba(255, 193, 7, 0.2)';
                statusElement.style.borderColor = 'rgba(255, 193, 7, 0.5)';
                statusElement.style.color = '#FFC107';
            }
        }

        // WebGL capability check
        function checkWebGLSupport() {
            debugLog('Checking WebGL support...');
            
            const canvas = document.createElement('canvas');
            const gl2 = canvas.getContext('webgl2');
            const gl1 = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            const info = {
                webgl2: !!gl2,
                webgl1: !!gl1,
                renderer: null,
                vendor: null,
                version: null
            };
            
            if (gl2) {
                info.renderer = gl2.getParameter(gl2.RENDERER);
                info.vendor = gl2.getParameter(gl2.VENDOR);
                info.version = gl2.getParameter(gl2.VERSION);
                debugLog(`WebGL2 supported - Renderer: ${info.renderer}`);
            } else if (gl1) {
                info.renderer = gl1.getParameter(gl1.RENDERER);
                info.vendor = gl1.getParameter(gl1.VENDOR);
                info.version = gl1.getParameter(gl1.VERSION);
                debugLog(`Only WebGL1 supported - Renderer: ${info.renderer}`);
            } else {
                debugLog('No WebGL support detected');
            }
            
            return info;
        }

        function showWebGLError(webglInfo) {
            const container = document.getElementById('niivue-container');
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; color: #fff;">
                    <h2 style="color: #f44336; margin-bottom: 20px;">WebGL Not Supported</h2>
                    <p style="margin-bottom: 15px;">NiiVue requires WebGL2 for 3D medical image visualization.</p>
                    
                    <div style="background: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; margin: 20px 0; text-align: left;">
                        <h4>System Information:</h4>
                        <p><strong>WebGL 2.0:</strong> ${webglInfo.webgl2 ? '✅ Supported' : '❌ Not Supported'}</p>
                        <p><strong>WebGL 1.0:</strong> ${webglInfo.webgl1 ? '✅ Supported' : '❌ Not Supported'}</p>
                        ${webglInfo.renderer ? `<p><strong>Graphics:</strong> ${webglInfo.renderer}</p>` : ''}
                        ${webglInfo.vendor ? `<p><strong>Vendor:</strong> ${webglInfo.vendor}</p>` : ''}
                    </div>

                    <div style="background: rgba(255,193,7,0.2); padding: 15px; border-radius: 10px; margin: 20px 0;">
                        <h4>Solutions to try:</h4>
                        <ul style="text-align: left; margin: 10px 0;">
                            <li>Update your graphics drivers</li>
                            <li>Try a different browser (Chrome, Firefox, Edge)</li>
                            <li>Enable hardware acceleration in browser settings</li>
                            <li>For Chrome: Type <code>chrome://flags</code> and enable WebGL</li>
                            <li>For Firefox: Type <code>about:config</code> and set webgl.force-enabled to true</li>
                        </ul>
                    </div>

                    <div style="background: rgba(33,150,243,0.2); padding: 15px; border-radius: 10px;">
                        <h4>Alternative Options:</h4>
                        <p>Consider using medical imaging software like:</p>
                        <ul style="text-align: left; margin: 10px 0;">
                            <li><strong>3D Slicer:</strong> Free, cross-platform medical imaging software</li>
                            <li><strong>ITK-SNAP:</strong> Specialized for medical image segmentation</li>
                            <li><strong>ImageJ/FIJI:</strong> With medical imaging plugins</li>
                            <li><strong>MRIcroGL:</strong> Lightweight medical image viewer</li>
                        </ul>
                    </div>
                </div>
            `;
        }

        // Initialize NiiVue when page loads
        window.addEventListener('load', function() {
            debugLog('Page loaded, initializing...');
            
            // Give the script a moment to fully load
            setTimeout(() => {
                try {
                    // First check WebGL support
                    const webglInfo = checkWebGLSupport();
                    
                    if (!webglInfo.webgl2) {
                        updateStatus('WebGL2 not supported - cannot initialize 3D viewer', 'error');
                        updateLibraryStatus('❌ WebGL2 required but not available', 'error');
                        showWebGLError(webglInfo);
                        return;
                    }

                    debugLog('Checking for NiiVue library...');
                    
                    // Check all possible ways NiiVue might be exposed
                    let NiivueClass = null;
                    
                    if (typeof Niivue !== 'undefined') {
                        debugLog('Found Niivue in global scope');
                        NiivueClass = Niivue;
                    } else if (typeof niivue !== 'undefined' && niivue.Niivue) {
                        debugLog('Found niivue.Niivue in global scope');
                        NiivueClass = niivue.Niivue;
                    } else if (typeof window.niivue !== 'undefined' && window.niivue.Niivue) {
                        debugLog('Found window.niivue.Niivue');
                        NiivueClass = window.niivue.Niivue;
                    } else if (typeof window.Niivue !== 'undefined') {
                        debugLog('Found window.Niivue');
                        NiivueClass = window.Niivue;
                    } else {
                        // Check what's actually available in global scope
                        debugLog('Available globals: ' + Object.keys(window).filter(k => k.toLowerCase().includes('niiv')).join(', '));
                        throw new Error('NiiVue library not found in any expected location');
                    }

                    debugLog('NiiVue library found, creating instance...');
                    updateLibraryStatus('✅ NiiVue library loaded, WebGL2 supported', 'success');
                    
                    nv = new NiivueClass();
                    
                    // Try to attach to container and catch WebGL errors
                    try {
                        nv.attachTo('niivue-container');
                    } catch (webglError) {
                        if (webglError.message.includes('WebGL')) {
                            updateStatus(`WebGL initialization failed: ${webglError.message}`, 'error');
                            updateLibraryStatus('❌ WebGL context creation failed', 'error');
                            showWebGLError(webglInfo);
                            return;
                        } else {
                            throw webglError;
                        }
                    }

                    // Configure NiiVue settings
                    nv.opts.dragAndDropEnabled = true;
                    nv.opts.backColor = [0.1, 0.1, 0.1, 1];
                    nv.opts.crosshairColor = [1, 1, 1, 0.8];
                    nv.opts.show3Dcrosshair = true;
                    nv.opts.isColorbar = false;

                    updateStatus('Viewer initialized. Upload your files to get started!', 'success');
                    debugLog('NiiVue initialized successfully');

                } catch (error) {
                    updateStatus(`Failed to initialize viewer: ${error.message}`, 'error');
                    updateLibraryStatus(`❌ ${error.message}`, 'error');
                    debugLog(`Initialization error: ${error.message}`);
                    console.error('Full error:', error);
                    
                    // Provide troubleshooting info
                    debugLog('Troubleshooting: Check that niivue.umd.js is in the same folder');
                    debugLog('Current location: ' + window.location.href);
                    
                    // If it's a WebGL error, show the WebGL troubleshooting
                    if (error.message.toLowerCase().includes('webgl')) {
                        const webglInfo = checkWebGLSupport();
                        showWebGLError(webglInfo);
                    }
                }
            }, 100);
        });

        // File upload event listeners with detailed logging
        document.getElementById('volume-file').addEventListener('change', function(e) {
            debugLog('Volume file input changed');
            handleFileUpload(e, 'volume');
        });

        document.getElementById('mask-file').addEventListener('change', function(e) {
            debugLog('Mask file input changed');
            handleFileUpload(e, 'mask');
        });

        function handleFileUpload(event, type) {
            debugLog(`handleFileUpload called for type: ${type}`);
            
            const file = event.target.files[0];
            const statusElement = document.getElementById(type + '-status');
            
            debugLog(`File object: ${file ? file.name : 'null'}`);
            
            if (!file) {
                debugLog(`No file selected for ${type}`);
                return;
            }

            // Validate file type
            if (!file.name.toLowerCase().includes('.nii')) {
                statusElement.innerHTML = 'Please select a .nii or .nii.gz file';
                statusElement.className = 'file-status error';
                debugLog(`Invalid file type for ${type}: ${file.name}`);
                return;
            }

            // Store file reference
            if (type === 'volume') {
                volumeFile = file;
                statusElement.innerHTML = `✅ ${file.name} (${formatFileSize(file.size)})`;
                statusElement.className = 'file-status success';
                debugLog(`Volume file set: ${volumeFile.name}`);
            } else if (type === 'mask') {
                maskFile = file;
                statusElement.innerHTML = `✅ ${file.name} (${formatFileSize(file.size)})`;
                statusElement.className = 'file-status success';
                debugLog(`Mask file set: ${maskFile.name}`);
            }

            // Check button state
            checkButtonState();
        }

        function checkButtonState() {
            const bothFilesSelected = (volumeFile && maskFile);
            const loadBtn = document.getElementById('load-btn');
            
            debugLog(`Checking button state - Volume: ${volumeFile ? 'YES' : 'NO'}, Mask: ${maskFile ? 'YES' : 'NO'}`);
            
            loadBtn.disabled = !bothFilesSelected || !nv;
            
            if (!nv) {
                updateStatus('NiiVue not initialized - cannot load files', 'error');
            } else if (bothFilesSelected) {
                updateStatus('Both files loaded. Click "Load and Visualize" to begin!', 'success');
                debugLog('Both files ready - button enabled');
            } else {
                if (!volumeFile && !maskFile) {
                    updateStatus('Please upload both volume and mask files', 'loading');
                } else if (!volumeFile) {
                    updateStatus('Please upload volume file', 'loading');
                } else if (!maskFile) {
                    updateStatus('Please upload mask file', 'loading');
                }
                debugLog('Files missing - button disabled');
            }
        }

        function debugFileState() {
            debugLog('=== MANUAL DEBUG CHECK ===');
            debugLog(`volumeFile: ${volumeFile ? volumeFile.name + ' (' + volumeFile.size + ' bytes)' : 'null'}`);
            debugLog(`maskFile: ${maskFile ? maskFile.name + ' (' + maskFile.size + ' bytes)' : 'null'}`);
            debugLog(`nv exists: ${!!nv}`);
            debugLog(`filesLoaded: ${filesLoaded}`);
            
            const loadBtn = document.getElementById('load-btn');
            debugLog(`loadBtn.disabled: ${loadBtn.disabled}`);
            
            // Check actual input elements
            const volumeInput = document.getElementById('volume-file');
            const maskInput = document.getElementById('mask-file');
            debugLog(`Volume input files: ${volumeInput.files.length}`);
            debugLog(`Mask input files: ${maskInput.files.length}`);
            
            // Check what's in the global scope
            debugLog(`Checking globals for NiiVue:`);
            debugLog(`- typeof Niivue: ${typeof Niivue}`);
            debugLog(`- typeof niivue: ${typeof niivue}`);
            debugLog(`- typeof window.niivue: ${typeof window.niivue}`);
            debugLog(`- typeof window.Niivue: ${typeof window.Niivue}`);
            
            // Display in UI
            const statusMsg = `Debug: Volume=${volumeFile ? '✅' : '❌'} Mask=${maskFile ? '✅' : '❌'} NiiVue=${nv ? '✅' : '❌'}`;
            updateStatus(statusMsg, volumeFile && maskFile && nv ? 'success' : 'error');
        }

        async function loadFiles() {
            debugLog('loadFiles() called');
            debugLog(`Current state - Volume: ${volumeFile ? 'SET' : 'NULL'}, Mask: ${maskFile ? 'SET' : 'NULL'}, NV: ${nv ? 'SET' : 'NULL'}`);

            if (!nv) {
                updateStatus('NiiVue not initialized', 'error');
                debugLog('ERROR: NiiVue not initialized');
                return;
            }

            if (!volumeFile) {
                updateStatus('Please select a volume file', 'error');
                debugLog('ERROR: No volume file selected');
                return;
            }

            if (!maskFile) {
                updateStatus('Please select a mask file', 'error');
                debugLog('ERROR: No mask file selected');
                return;
            }

            try {
                showLoadingOverlay(true);
                updateStatus('Loading medical images...', 'loading');
                debugLog('Starting file load process...');

                // Clear any existing volumes
                nv.volumes = [];
                debugLog('Cleared existing volumes');

                // Create object URLs for the files
                const volumeUrl = URL.createObjectURL(volumeFile);
                const maskUrl = URL.createObjectURL(maskFile);
                debugLog(`Created URLs: ${volumeUrl.substring(0, 50)}... and ${maskUrl.substring(0, 50)}...`);

                const volumeList = [
                    {
                        url: volumeUrl,
                        name: volumeFile.name,
                        colormap: 'gray',
                        opacity: 1.0
                    },
                    {
                        url: maskUrl,
                        name: maskFile.name,
                        colormap: 'blue2red',
                        opacity: 0.7,
                        cal_min: 0.0,
                        cal_max: 1.0
                    }
                ];

                debugLog('Calling nv.loadVolumes()...');
                await nv.loadVolumes(volumeList);
                debugLog(`Volumes loaded: ${nv.volumes.length}`);

                // Clean up object URLs
                URL.revokeObjectURL(volumeUrl);
                URL.revokeObjectURL(maskUrl);
                debugLog('Object URLs revoked');

                if (nv.volumes.length < 2) {
                    throw new Error(`Expected 2 volumes, got ${nv.volumes.length}`);
                }

                filesLoaded = true;
                showLoadingOverlay(false);
                updateStatus(`Successfully loaded ${nv.volumes.length} volumes! Use mouse to navigate.`, 'success');
                debugLog('SUCCESS: Files loaded and displayed');
                
                // Log volume information
                nv.volumes.forEach((vol, idx) => {
                    debugLog(`Volume ${idx}: ${vol.name}, dims: ${vol.dims?.join('x') || 'unknown'}`);
                });

                // Apply initial settings
                setTimeout(() => {
                    forceDisplay();
                }, 500);

            } catch (error) {
                showLoadingOverlay(false);
                updateStatus(`Failed to load files: ${error.message}`, 'error');
                debugLog(`LOAD ERROR: ${error.message}`);
                console.error('Full error:', error);
            }
        }

        function forceDisplay() {
            if (!filesLoaded || nv.volumes.length < 2) {
                debugLog('Cannot force display - files not loaded properly');
                return;
            }

            try {
                const uncertaintyVol = nv.volumes[1];
                
                // Set intensity range for uncertainty mask
                uncertaintyVol.cal_min = 0.0;
                uncertaintyVol.cal_max = 1.0;
                
                // Apply current settings
                const opacity = parseFloat(document.getElementById('opacity-slider').value);
                const colormap = document.getElementById('colormap-select').value;
                
                nv.setColormap(1, colormap);
                nv.setOpacity(1, opacity);
                nv.drawScene();
                
                debugLog('Display settings applied successfully');
                
            } catch (error) {
                debugLog(`Force display error: ${error.message}`);
            }
        }

        function updateOpacity() {
            if (!filesLoaded || nv.volumes.length < 2) return;
            
            const slider = document.getElementById('opacity-slider');
            const value = parseFloat(slider.value);
            document.getElementById('opacity-value').textContent = value.toFixed(1);
            
            nv.setOpacity(1, value);
        }

        function updateColormap() {
            if (!filesLoaded || nv.volumes.length < 2) return;
            
            const select = document.getElementById('colormap-select');
            const colormap = select.value;
            
            nv.setColormap(1, colormap);
        }

        function resetView() {
            if (!nv) return;
            
            nv.scene.volScaleMultiplier = 1.0;
            nv.scene.crosshairPos = [0.5, 0.5, 0.5];
            nv.drawScene();
            
            updateStatus('View reset to default position', 'success');
        }

        function showStatistics() {
            if (!filesLoaded || nv.volumes.length < 2) return;

            const volumeVol = nv.volumes[0];
            const uncertaintyVol = nv.volumes[1];
            
            debugLog(`Volume stats: ${volumeVol.name}, dims: ${volumeVol.dims?.join('x')}`);
            debugLog(`Mask stats: ${uncertaintyVol.name}, dims: ${uncertaintyVol.dims?.join('x')}`);
        }

        // Utility functions
        function updateStatus(message, type = 'loading') {
            const statusElement = document.getElementById('main-status');
            statusElement.innerHTML = `<strong>Status:</strong> ${message}`;
            statusElement.className = `status ${type}`;
        }

        function showLoadingOverlay(show) {
            const overlay = document.getElementById('loading-overlay');
            overlay.style.display = show ? 'flex' : 'none';
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // Initialize debug logging
        debugLog('Script loaded, waiting for page load...');
    </script>
</body>
</html>
EOF

# Create GPU-aware launcher scripts
create_launcher_scripts() {
    # Create GPU-enabled browser launcher
    cat > start_gpu_browser.sh << EOF
#!/bin/bash
echo "🚀 Starting Medical Image Viewer with GPU acceleration..."

# Set GPU if specified during setup
if [ -n "${GPU_SELECTION:-}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU_SELECTION}
    echo "🎯 Using GPU ${GPU_SELECTION}: \$(nvidia-smi -i ${GPU_SELECTION} --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU info unavailable')"
else
    echo "🔧 Using default GPU configuration"
fi

# Launch browser with GPU acceleration
if command -v google-chrome &> /dev/null; then
    echo "🌐 Launching Chrome with GPU acceleration..."
    google-chrome \\
        --no-sandbox \\
        --use-gl=egl \\
        --enable-webgl \\
        --enable-accelerated-2d-canvas \\
        --enable-gpu-rasterization \\
        --gpu-no-sandbox \\
        uncertainty_viewer.html
elif command -v chromium-browser &> /dev/null; then
    echo "🌐 Launching Chromium with GPU acceleration..."
    chromium-browser \\
        --no-sandbox \\
        --use-gl=egl \\
        --enable-webgl \\
        --enable-accelerated-2d-canvas \\
        uncertainty_viewer.html
elif command -v firefox &> /dev/null; then
    echo "🌐 Launching Firefox..."
    firefox uncertainty_viewer.html
else
    echo "❌ No supported browser found"
    echo "Please install Chrome, Chromium, or Firefox"
    exit 1
fi
EOF

    # Create server launcher
    cat > start_server.sh << EOF
#!/bin/bash
echo "🚀 Starting Medical Image Uncertainty Viewer..."
echo "📂 Server will be available at: http://localhost:8000"
echo "📝 Open uncertainty_viewer.html in your browser"

# Set GPU if specified during setup
if [ -n "$GPU_SELECTION" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_SELECTION
    echo "🎯 GPU $GPU_SELECTION will be used for WebGL acceleration"
fi

echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Try different methods to start a local server
if command -v python3 &> /dev/null; then
    echo "Using Python 3..."
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    echo "Using Python 2..."
    python -m SimpleHTTPServer 8000
elif command -v php &> /dev/null; then
    echo "Using PHP..."
    php -S localhost:8000
else
    echo "❌ No suitable server found. Please install Python or PHP."
    echo "Or open uncertainty_viewer.html directly in your browser."
fi
EOF
    
    chmod +x start_gpu_browser.sh start_server.sh
}

# Create the launcher scripts with GPU support
create_launcher_scripts

echo ""
echo "✅ Setup complete!"
echo ""
echo "📁 Files created:"
echo "  - uncertainty_viewer.html (main viewer)"
echo "  - niivue.umd.js (NiiVue library)"
echo "  - start_gpu_browser.sh (GPU-accelerated browser launcher)"
echo "  - start_server.sh (server launcher with GPU support)"
echo ""
echo "🚀 To start the viewer:"
if [ -n "$GPU_SELECTION" ]; then
    echo "  GPU $GPU_SELECTION selected - optimized for hardware acceleration"
fi
echo "  1. Direct GPU browser: ./start_gpu_browser.sh"
echo "  2. HTTP server: ./start_server.sh"
echo "  3. Or open uncertainty_viewer.html directly in your browser"
echo ""
echo "📝 The viewer works completely offline with GPU acceleration!"