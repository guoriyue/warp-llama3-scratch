display_help() {
    echo "Usage: $0 [library] [mode]" >&2
    echo
    echo "Options:"
    echo "   -h, --help          Show help"
    echo
    echo "Library:"
    echo "   wp                  Use Warp library (default if not specified)"
    echo "   torch               Use Torch library"
    echo
    echo "Mode:"
    echo "   (no mode)           Run inference (default mode if not specified)"
    echo "   prof                Run profiling (use with wp or torch)"
    echo
    echo "Examples:"
    echo "   $0                   Run Warp library in inference mode"
    echo "   $0 wp                Run Warp library in inference mode"
    echo "   $0 torch             Run Torch library in inference mode"
    echo "   $0 wp prof           Run Warp library with profiling"
    echo "   $0 torch prof        Run Torch library with profiling"
    echo
}

library="wp"
mode="inference"

if [ $# -eq 0 ]; then
    display_help
elif [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
    exit 1
elif [ "$1" == "wp" ] || [ "$1" == "torch" ]; then
    library="$1"
    if [ "$2" == "prof" ]; then
        mode="profile"
    fi
elif [ "$1" == "prof" ]; then
    mode="profile"
else
    echo "Unknown option: $1"
    display_help
    exit 1
fi

if [ "$library" == "wp" ]; then
    if [ "$mode" == "profile" ]; then
        echo "Running Warp with profiling..."
        python wp_inference.py --profile
    else
        echo "Running Warp with inference..."
        python wp_inference.py
    fi
elif [ "$library" == "torch" ]; then
    if [ "$mode" == "profile" ]; then
        echo "Running Torch with profiling..."
        python torch_inference.py --profile
    else
        echo "Running Torch with inference..."
        python torch_inference.py
    fi
else
    echo "Unknown library: $library"
    display_help
    exit 1
fi
