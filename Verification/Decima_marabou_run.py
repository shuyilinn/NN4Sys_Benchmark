"""
This script runs Marabou verification for Decima models using ONNX models and TXT specifications.
It executes the Marabou verification and stores the results in the specified directory.
"""

import os
import sys
import argparse

# Model configurations for Decima
MODELS = ['mid', 'mid', 'mid']  # Models to run
MODEL_TYPES = ['simple', 'simple']  # Model types
SPEC_TYPES = [1, 2]  # Specification types

# Paths
running_result_path = './decima_marabou_running_result'  # Path to store results
txt_dir_path = '../Benchmarks/marabou_txt'  # Path for Marabou TXT files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models

# Ensure the results directory exists
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)

def main(marabou_path, size=10):
    """
    Main function to run Marabou verification for Decima models.

    Args:
        marabou_path (str): Path to the Marabou verifier executable/script.
        size (int): Number of verification instances to run (default: 10).
    """
    for spec_type_ptr in range(len(SPEC_TYPES)):
        for i in range(size):
            # Construct the Marabou command for each instance
            command = (f'python {marabou_path} {onnx_dir_path}/decima_mid_marabou_{SPEC_TYPES[spec_type_ptr]}.onnx '
                       f'{txt_dir_path}/decima_{SPEC_TYPES[spec_type_ptr]}_0.txt | tee '
                       f'{running_result_path}/decima_mid_{SPEC_TYPES[spec_type_ptr]}_{i}.txt')
            print("------------------------------------->")
            print(command)
            print("<------------------------------------->")
            # Execute the command
            os.system(command)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run Marabou verification for Decima models.")
    
    # Add argument for the Marabou path
    parser.add_argument("marabou_path", help="Path to the Marabou verifier.")
    
    # Add optional argument for specifying the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.marabou_path, args.size)
