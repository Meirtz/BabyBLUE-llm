#!/bin/bash

# Directory containing the JSON files
BENCHMARK_DIR="/home/meilingrui/workspace/BabyBLUE/benchmark_data"
# Directory to save the results
RESULTS_DIR="./results"

# Create results directory if it does not exist
#mkdir -p "$RESULTS_DIR"

# Iterate over each JSON file in the benchmark directory
for json_file in "$BENCHMARK_DIR"/*.json;
do
    # Extract the base name of the file (without directory and extension)
    base_name=$(basename "$json_file" .json)
    # Define the save path for the results
    save_path="$RESULTS_DIR/${base_name}_with_babyblue.json"

    # Check if the result file already exists
    if [ -f "$save_path" ]; then
        echo "Result file $save_path already exists. Skipping."
    else
        # Execute the python command
        python evaluate_completions.py --completions_path "$json_file" --save_path "$save_path"
        echo "Processed $json_file and saved results to $save_path."
    fi
done

echo "Processing complete. Results are saved in $RESULTS_DIR."
