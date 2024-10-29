"""
This script runs Marabou verification for Bloom Filter models using ONNX models.
It generates results and stores them in the specified directory.
"""

import os
import argparse

# Define paths for the Marabou TXT and ONNX models
txt_dir_path = '../Benchmarks/marabou_txt'  # Path for Marabou TXT files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models
running_result_path = './bloom_filter_marabou_running_result'  # Path for saving results

# Create the result directory if it doesn't exist
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)

def main(marabou_path, size=10):
    """
    Main function to run Marabou verification for Bloom Filter models.

    Args:
        marabou_path (str): Path to the Marabou verifier executable/script.
        size (int): Number of verification instances to run (default: 10).
    """
    for i in range(size):
        # Construct the Marabou command for each instance
        command = f'python {marabou_path} {onnx_dir_path}/bloom_filter.onnx {txt_dir_path}/bloom_filter_{i}.txt | tee {running_result_path}/bloom_filter_{i}.txt'
        
        # Print and execute the command
        print(f"Executing: {command}")
        os.system(command)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Marabou verification for Bloom Filter models.")
    
    # Add argument for the Marabou path
    parser.add_argument("--marabou_path", help="Path to the Marabou verifier.")
    
    # Add optional argument for the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.marabou_path, args.size)
