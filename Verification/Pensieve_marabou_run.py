"""
This script runs Marabou verification for Pensieve models using ONNX models and TXT specifications.
It executes the Marabou verification and stores the results in the specified directory.
"""

import os
import sys
import argparse

# Model configurations for Pensieve
MODELS = ['small', 'mid', 'big']  # Model sizes
MODEL_TYPES = ['simple', 'simple']  # Model types

# Paths
running_result_path = './pensieve_marabou_running_result'  # Path to store results
txt_dir_path = '../Benchmarks/marabou_txt'  # Path for Marabou TXT files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models

# Verification parameters
P_RANGE = [0.05, 0.1, 0.5, 0.7, 1]  # Parameter ranges
SPEC_TYPES = [1, 2]  # Specification types
DIMENSION_NUMBERS = [1, 2, 3]  # Dimension numbers

# Create necessary directories if they don't exist
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)

def main(marabou_path, size=10):
    """
    Main function to run Marabou verification for Pensieve models.

    Args:
        marabou_path (str): Path to the Marabou verifier executable/script.
        size (int): Number of verification instances to run for each model (default: 10).
    """
    for spec_type in range(len(SPEC_TYPES)):
        for range_ptr in range(len(P_RANGE)):
            for d_ptr in range(len(DIMENSION_NUMBERS)):
                dimension_number = DIMENSION_NUMBERS[d_ptr]
                for num in range(size):
                    if dimension_number != 2 or range_ptr != 0:
                        continue

                    # Construct the Marabou command for each instance
                    command = (f'python {marabou_path} {onnx_dir_path}/pensieve_small_{MODEL_TYPES[spec_type]}_marabou.onnx '
                               f'{txt_dir_path}/pensieve_{SPEC_TYPES[spec_type]}_{dimension_number}_{range_ptr}_{num}.txt | tee '
                               f'{running_result_path}/pensieve_small_{SPEC_TYPES[spec_type]}_{dimension_number}_{range_ptr}_{num}.txt')

                    # Print and execute the command
                    print("------------------------------------->")
                    print(command)
                    print("<------------------------------------->")
                    os.system(command)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Marabou verification for Pensieve models.")
    
    # Add argument for the path to the Marabou verifier
    parser.add_argument("marabou_path", help="Path to the Marabou verifier.")
    
    # Add optional argument for specifying the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.marabou_path, args.size)
