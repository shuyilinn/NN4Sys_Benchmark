"""
This script extracts ONNX model information to populate a big table in the paper.
It uses the `onnx_tool` to analyze each ONNX model in the specified directory.
"""

import os
import sys

# Directory containing ONNX models
onnx_dir_path = '../Benchmarks/onnx'

# Directory to store output information about each ONNX model
onnx_info_path = "./onnx_info"

# Ensure the output directory exists, create it if necessary
if not os.path.exists(onnx_info_path):
    os.makedirs(onnx_info_path)

# List all files in the ONNX directory
files = os.listdir(onnx_dir_path)

# Iterate through each file in the ONNX directory
for file in files:
    # Ensure the file has a .onnx extension
    if not file.endswith(".onnx"):
        continue  # Skip non-ONNX files

    # Extract the base name (without extension) for the output file
    name = os.path.splitext(file)[0]

    # Define the output path for storing the ONNX model information
    out_path = os.path.join(onnx_info_path, f'{name}.txt')

    # Construct the command to run the onnx_tool on the ONNX file
    command = f"python -m onnx_tool -i {os.path.join(onnx_dir_path, file)} -f {out_path}"

    # Print the command being executed for reference
    print("------------------------------------->")
    print(f"Executing command: {command}")
    print("<------------------------------------->")

    # Execute the command to get the ONNX model information
    os.system(command)
