"""
This script runs Marabou verification for Lindex models using ONNX models and TXT specifications.
It runs the Marabou verification for the specified number of instances and saves the results.
"""

import os
import sys
import argparse

# Paths
running_result_path = './lindex_marabou_running_result'  # Path to store results
txt_dir_path = '../Benchmarks/marabou_txt'  # Path for Marabou TXT files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models

# Lindex model names
MODEL_NAMES = ["lindex", "lindex_deep"]  # Models to run

def main(marabou_path, size=10):
    """
    Main function to run Marabou verification for Lindex models.

    Args:
        marabou_path (str): Path to the Marabou verifier executable/script.
        size (int): Number of verification instances to run for each model (default: 10).
    """
    # Ensure the results directory exists
    if not os.path.exists(running_result_path):
        os.makedirs(running_result_path)

    # Loop over the models and verification instances
    for model in MODEL_NAMES:
        for num in range(size):
            # Construct the Marabou command for each instance
            command = (f'python {marabou_path} {onnx_dir_path}/{model}.onnx '
                       f'{txt_dir_path}/lindex_0_{num}.txt | tee '
                       f'{running_result_path}/{model}_0_{num}.txt')

            # Print and execute the command
            print(f"Executing: {command}")
            os.system(command)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Marabou verification for Lindex models.")
    
    # Add argument for the path to the Marabou verifier
    parser.add_argument("marabou_path", help="Path to the Marabou verifier.")
    
    # Add optional argument for specifying the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.marabou_path, args.size)
