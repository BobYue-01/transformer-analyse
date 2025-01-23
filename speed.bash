#!/bin/bash

# Output CSV file
output_csv="output.csv"

# Check if the output CSV file already exists
if [ -f "$output_csv" ]; then
    echo "The output CSV file already exists. Do you want to overwrite it? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
    rm "$output_csv"
fi

# Write CSV header
echo "model_name,model_type,start_time,cpu_time_total,device_time_total,self_cpu_time_total,self_device_time_total,self_device_memory_usage" > $output_csv

# Function to extract and append data to CSV
append_data_to_csv() {
    local model_name=$1
    local model_type=$2
    local start_time=$3
    local data=$4

    local cpu_time_total=$(echo "$data" | jq '.cpu_time_total | tonumber | . * 1000 | floor / 1000')
    local device_time_total=$(echo "$data" | jq '.device_time_total | tonumber | . * 1000 | floor / 1000')
    local self_cpu_time_total=$(echo "$data" | jq '.self_cpu_time_total | tonumber | . * 1000 | floor / 1000')
    local self_device_time_total=$(echo "$data" | jq '.self_device_time_total | tonumber | . * 1000 | floor / 1000')
    local self_device_memory_usage=$(echo "$data" | jq '.self_device_memory_usage')

    echo "$model_name,$model_type,$start_time,$cpu_time_total,$device_time_total,$self_cpu_time_total,$self_device_time_total,$self_device_memory_usage" >> "$output_csv"
}

# Iterate over JSON files in the ./log directory
for file in ./log/*_Speed_*.json; do
    # Extract model_name, date, and time from the filename
    filename=$(basename "$file")
    model_name=$(echo "$filename" | cut -d'_' -f1)
    start_time=$(echo "$filename" | cut -d'_' -f3 | cut -d'.' -f1)

    # Extract data from the JSON file
    folded=$(jq '."folded"."ProfilerStep*"' "$file")
    original=$(jq '."original"."ProfilerStep*"' "$file")

    # Append the extracted data to the CSV file
    append_data_to_csv "$model_name" "folded" "$start_time" "$folded"
    append_data_to_csv "$model_name" "original" "$start_time" "$original"
done
