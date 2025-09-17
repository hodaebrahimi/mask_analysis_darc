#!/bin/bash

# GPU Selection and Assignment Script for Medical Image Viewer
# This script helps you choose and assign a specific GPU for WebGL acceleration

echo "🔍 GPU Selection and Assignment Tool"
echo "=================================="

# Function to check GPU availability
check_gpu_status() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    echo "📊 Current GPU Status:"
    nvidia-smi
    echo ""
}

# Function to list GPUs with detailed info
list_available_gpus() {
    echo "🖥️  Available GPUs:"
    echo "=================="
    
    # Get GPU count
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    
    if [ $gpu_count -eq 0 ]; then
        echo "❌ No NVIDIA GPUs found"
        exit 1
    fi
    
    # Create array to store GPU info
    declare -a gpu_info
    declare -a gpu_usage
    declare -a gpu_memory
    declare -a gpu_temp
    declare -a gpu_processes
    
    echo "ID | Name                    | Memory Usage | GPU Util | Temp | Processes | Status"
    echo "---|-------------------------|--------------|----------|------|-----------|--------"
    
    for i in $(seq 0 $((gpu_count-1))); do
        # Get GPU name
        local name=$(nvidia-smi -i $i --query-gpu=name --format=csv,noheader,nounits)
        
        # Get memory usage
        local mem_used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
        local mem_total=$(nvidia-smi -i $i --query-gpu=memory.total --format=csv,noheader,nounits)
        local mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "N/A")
        
        # Get GPU utilization
        local gpu_util=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        
        # Get temperature
        local temp=$(nvidia-smi -i $i --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        
        # Count processes
        local process_count=$(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c "^[0-9]" || echo "0")
        
        # Determine availability status
        local status
        if [ "$gpu_util" -lt 10 ] && [ "$mem_percent" != "N/A" ] && (( $(echo "$mem_percent < 10" | bc -l) )); then
            status="🟢 AVAILABLE"
        elif [ "$gpu_util" -lt 50 ]; then
            status="🟡 MODERATE"
        else
            status="🔴 BUSY"
        fi
        
        # Truncate long GPU names
        local short_name=$(echo "$name" | cut -c1-23)
        
        printf "%2d | %-23s | %4s/%4s MB | %8s%% | %3s°C | %9s | %s\n" \
               $i "$short_name" "$mem_used" "$mem_total" "$gpu_util" "$temp" "$process_count" "$status"
    done
    
    echo ""
    return $gpu_count
}

# Function to show detailed GPU processes
show_gpu_processes() {
    local gpu_id=$1
    echo "🔍 Processes on GPU $gpu_id:"
    echo "=========================="
    
    local processes=$(nvidia-smi -i $gpu_id --query-compute-apps=pid,process_name,used_memory --format=csv,header)
    
    if [ -z "$processes" ] || [ "$processes" = "pid, process_name, used_gpu_memory [MiB]" ]; then
        echo "✅ No active processes on GPU $gpu_id"
    else
        echo "$processes"
    fi
    echo ""
}

# Function to test GPU availability
test_gpu() {
    local gpu_id=$1
    echo "🧪 Testing GPU $gpu_id..."
    
    # Test basic CUDA access
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✅ CUDA runtime accessible"
    else
        echo "  ❌ CUDA runtime not accessible"
        return 1
    fi
    
    # Test WebGL capability (requires X server)
    if [ -n "$DISPLAY" ]; then
        echo "  ✅ X server available on $DISPLAY"
        
        # Test GLX
        if command -v glxinfo &> /dev/null; then
            local renderer=$(CUDA_VISIBLE_DEVICES=$gpu_id glxinfo | grep "OpenGL renderer" | cut -d':' -f2 | xargs)
            if [ -n "$renderer" ]; then
                echo "  ✅ OpenGL renderer: $renderer"
            else
                echo "  ⚠️  OpenGL renderer not detected"
            fi
        else
            echo "  ⚠️  glxinfo not available (install mesa-utils)"
        fi
    else
        echo "  ⚠️  No X server detected (headless mode)"
    fi
    
    echo ""
}

# Function to configure GPU assignment
configure_gpu() {
    local gpu_id=$1
    
    echo "⚙️  Configuring GPU $gpu_id for medical viewer..."
    
    # Create GPU-specific environment script
    cat > gpu_env_${gpu_id}.sh << EOF
#!/bin/bash
# GPU $gpu_id Environment Configuration
export CUDA_VISIBLE_DEVICES=$gpu_id
export GPU_ID=$gpu_id

echo "🚀 Using GPU $gpu_id for medical image viewer"
echo "GPU: \$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader,nounits)"
echo ""

# Export for current shell
set -a
source gpu_env_${gpu_id}.sh
set +a
EOF
    
    chmod +x gpu_env_${gpu_id}.sh
    
    # Get GPU info
    local gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader,nounits)
    local gpu_pci=$(nvidia-xconfig --query-gpu-info 2>/dev/null | grep "PCI BusID" | sed -n "$((gpu_id+1))p" | awk '{print $4}')
    
    # Create X server configuration for this GPU
    if [ -n "$gpu_pci" ]; then
        echo "📝 Creating X server configuration for GPU $gpu_id ($gpu_pci)..."
        
        cat > xorg_gpu_${gpu_id}.conf << EOF
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    BusID          "$gpu_pci"
    Option         "UseDisplayDevice" "None"
    Option         "ConnectedMonitor" "DFP-0"
    Option         "CustomEDID" "DFP-0:/etc/X11/edid.bin"
EndSection

Section "Monitor"
    Identifier     "Monitor0"
    Option         "Enable" "false"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    Monitor        "Monitor0"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
        Virtual     1920 1200
    EndSubSection
EndSection
EOF
        
        echo "✅ X server config created: xorg_gpu_${gpu_id}.conf"
    else
        echo "⚠️  Could not determine PCI bus ID for GPU $gpu_id"
    fi
    
    # Create GPU-specific launcher
    cat > start_viewer_gpu_${gpu_id}.sh << EOF
#!/bin/bash
echo "🚀 Starting Medical Viewer on GPU $gpu_id"
echo "GPU: $gpu_name"
echo ""

# Set GPU environment
export CUDA_VISIBLE_DEVICES=$gpu_id
export GPU_ID=$gpu_id

# Start with GPU acceleration
if command -v google-chrome &> /dev/null; then
    echo "🌐 Launching Chrome with GPU $gpu_id..."
    google-chrome \\
        --no-sandbox \\
        --use-gl=egl \\
        --enable-webgl \\
        --enable-accelerated-2d-canvas \\
        --enable-gpu-rasterization \\
        --gpu-no-sandbox \\
        uncertainty_viewer.html
elif command -v chromium-browser &> /dev/null; then
    echo "🌐 Launching Chromium with GPU $gpu_id..."
    chromium-browser \\
        --no-sandbox \\
        --use-gl=egl \\
        --enable-webgl \\
        --enable-accelerated-2d-canvas \\
        uncertainty_viewer.html
else
    echo "❌ No supported browser found"
    exit 1
fi
EOF
    
    chmod +x start_viewer_gpu_${gpu_id}.sh
    
    echo "✅ GPU $gpu_id configured successfully!"
    echo "📁 Files created:"
    echo "  - gpu_env_${gpu_id}.sh (environment setup)"
    echo "  - xorg_gpu_${gpu_id}.conf (X server config)"
    echo "  - start_viewer_gpu_${gpu_id}.sh (viewer launcher)"
    echo ""
}

# Main menu function
show_menu() {
    echo "🎯 GPU Assignment Options:"
    echo "1. Check GPU status"
    echo "2. List available GPUs"
    echo "3. Show processes on specific GPU"
    echo "4. Test GPU for WebGL"
    echo "5. Configure specific GPU"
    echo "6. Launch viewer with specific GPU"
    echo "7. Reset GPU assignments"
    echo "8. Exit"
    echo ""
    read -p "Choose an option (1-8): " choice
    echo ""
    
    case $choice in
        1)
            check_gpu_status
            ;;
        2)
            list_available_gpus
            ;;
        3)
            read -p "Enter GPU ID to check: " gpu_id
            show_gpu_processes $gpu_id
            ;;
        4)
            read -p "Enter GPU ID to test: " gpu_id
            test_gpu $gpu_id
            ;;
        5)
            list_available_gpus
            read -p "Enter GPU ID to configure: " gpu_id
            configure_gpu $gpu_id
            ;;
        6)
            echo "Available launchers:"
            ls -1 start_viewer_gpu_*.sh 2>/dev/null | sed 's/start_viewer_gpu_\(.*\)\.sh/  GPU \1: start_viewer_gpu_\1.sh/'
            echo ""
            read -p "Enter GPU ID to use: " gpu_id
            if [ -f "start_viewer_gpu_${gpu_id}.sh" ]; then
                ./start_viewer_gpu_${gpu_id}.sh
            else
                echo "❌ Launcher for GPU $gpu_id not found. Configure it first (option 5)."
            fi
            ;;
        7)
            echo "🧹 Cleaning up GPU configurations..."
            rm -f gpu_env_*.sh xorg_gpu_*.conf start_viewer_gpu_*.sh
            echo "✅ Cleanup complete"
            ;;
        8)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# Main script execution
main() {
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        echo "⚠️  Running as root. Some GPU operations may not work as expected."
        echo "Consider running as regular user for WebGL applications."
        echo ""
    fi
    
    # Check for required tools
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        echo "📦 Installing bc for calculations..."
        sudo apt-get update && sudo apt-get install -y bc
    fi
    
    # Interactive menu
    while true; do
        show_menu
    done
}

# Command line options
case "${1:-}" in
    "status")
        check_gpu_status
        ;;
    "list")
        list_available_gpus
        ;;
    "processes")
        if [ -n "$2" ]; then
            show_gpu_processes $2
        else
            echo "Usage: $0 processes <gpu_id>"
        fi
        ;;
    "test")
        if [ -n "$2" ]; then
            test_gpu $2
        else
            echo "Usage: $0 test <gpu_id>"
        fi
        ;;
    "configure")
        if [ -n "$2" ]; then
            configure_gpu $2
        else
            echo "Usage: $0 configure <gpu_id>"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "GPU Selection and Assignment Tool"
        echo ""
        echo "Usage: $0 [command] [gpu_id]"
        echo ""
        echo "Commands:"
        echo "  status                 Show current GPU status"
        echo "  list                  List all available GPUs"
        echo "  processes <gpu_id>    Show processes on specific GPU"
        echo "  test <gpu_id>         Test GPU for WebGL capability"
        echo "  configure <gpu_id>    Configure specific GPU"
        echo "  help                  Show this help"
        echo ""
        echo "Run without arguments for interactive mode"
        ;;
    *)
        main
        ;;
esac