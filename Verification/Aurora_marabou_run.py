"""
This script runs Marabou verification for various Aurora model configurations using ONNX models.
The results are stored in the specified directory.
"""

import os
import sys
import argparse

# Define constants for running Marabou verification
STATISTIC_RANGE = [0.05, 0.01, 0, 1]
MODEL_SIZES = ['small', 'mid', 'big']  # Different model sizes
MODEL_TYPES = ['simple', 'simple', 'simple', 'parallel', 'concat']  # Model types
running_result_path = './aurora_marabou_running_result'  # Path to store results

# Specification configurations for Aurora
SPEC_TYPES = [101, 102, 2, 3]  # Different specification types
P_RANGE = [0.8, 1, 1.2, 1.4, 1.6]  # Parameter ranges to test
DIMENSION_NUMBERS = [1, 2, 3]  # Different dimension numbers for tasks

# Paths for ONNX models and Marabou results
txt_dir_path = '../Benchmarks/marabou_txt'
onnx_dir_path = '../Benchmarks/onnx'


def main(marabou_path, size=10):
    """
    Main function to run Marabou verification for each model configuration.

    Args:
        marabou_path (str): Path to the Marabou verifier executable/script.
        size (int): Number of verification instances to run for each configuration.
    """
    # Ensure the results directory exists
    if not os.path.exists(running_result_path):
        os.makedirs(running_result_path)

    # Loop through each specification type
    for spec_type_index, spec_type in enumerate(SPEC_TYPES):
        # Loop through each model size
        for model_size in MODEL_SIZES:
            # Loop through the number of runs specified for each spec type (using --size argument)
            for num in range(size):
                # Loop through parameter ranges and dimension numbers
                for range_ptr, p_range in enumerate(P_RANGE):
                    for dim_index, dimension_number in enumerate(DIMENSION_NUMBERS):
                        # Only run the task when dimension_number is 3 and range_ptr is 1
                        if dimension_number != 3 or range_ptr != 1:
                            continue
                        
                        # Construct the paths for ONNX and Marabou txt files
                        onnx_model_path = f"{onnx_dir_path}/aurora_{model_size}_{MODEL_TYPES[spec_type_index]}.onnx"
                        marabou_txt_path = f"{txt_dir_path}/aurora_{spec_type}_{dimension_number}_{range_ptr}_{num}.txt"
                        result_path = f"{running_result_path}/aurora_{model_size}_{spec_type}_{dimension_number}_{range_ptr}_{num}.txt"
                        
                        # Construct the command to run Marabou
                        command = f'python {marabou_path} {onnx_model_path} {marabou_txt_path} | tee {result_path}'

                        # Print the command being executed for reference
                        print("------------------------------------->")
                        print(f"Executing: {command}")
                        print("<-------------------------------------")
                        
                        # Execute the command
                        os.system(command)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run Marabou verification on Aurora models.")
    
    # Add required argument for the Marabou path
    parser.add_argument("marabou_path", help="Path to the Marabou verifier executable/script.")
    
    # Add optional size argument to specify the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.marabou_path, args.size)
