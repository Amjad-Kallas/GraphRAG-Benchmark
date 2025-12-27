#!/bin/bash

# Base paths
BASE_DATA_DIR="./Examples/hipporag2_workspace"
OUTPUT_DIR="./results_indexing"

# --- BATCH CONFIGURATION ---
# Change these numbers to control the batch (1-based index)
# Batch 1: Set 1 to 4
# Batch 2: Set 5 to 8, etc.
START_NUM=1
END_NUM=20
# ---------------------------

# 1. Collect all matching folders into an array and sort them naturally (version sort)
# This ensures Novel-2 comes before Novel-10
mapfile -t folders < <(find "$BASE_DATA_DIR" -maxdepth 1 -type d -name "Novel-*" | sort -V)

# 2. Get total count
total_folders=${#folders[@]}
echo "Found a total of $total_folders folders."

# 3. Loop through the specific range
# We use C-style loop. Bash arrays are 0-indexed, so we subtract 1 from START_NUM.
for ((i=START_NUM-1; i<END_NUM; i++)); do
    
    # Check if index exists (prevents crashing if you set END_NUM too high)
    if [ $i -ge $total_folders ]; then
        echo "Index $i is out of bounds. Stopping."
        break
    fi

    folder="${folders[$i]}"
    
    # Extract folder name (e.g., Novel-2544)
    folder_name=$(basename "$folder")
    
    # Calculate current number for display (i+1)
    current_num=$((i + 1))

    # Set input and output files
    DATA_FILE="$folder/meta-llama_Llama-3.1-8B-Instruct_BAAI_bge-large-en-v1.5/graph.pickle.json"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="$OUTPUT_DIR/indexing_metrics_${folder_name}.txt"

    # Run the evaluation
    echo "[$current_num/$total_folders] Running generation_eval for $folder_name..."
    
    LLM_API_KEY="NONE" python -m Evaluation.indexing_eval \
        --framework hipporag2   --base_path ./Examples/hipporag2_workspace   --folder_name "$folder_name/meta-llama_Llama-3.1-8B-Instruct_BAAI_bge-large-en-v1.5"   --output "$OUTPUT_FILE"
done
