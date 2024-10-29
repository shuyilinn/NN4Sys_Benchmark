"""
This script runs ABCrown verification for Decima models using ONNX models and VNNLIB specifications.
It generates YAML configurations for each model and saves the results in the specified directory.
"""

import os
import argparse

# Set MKL threading layer to GNU
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Model configurations for Decima
MODEL_TYPES = ['simple', 'simple', 'concat']  # Types of models
MODEL_SIZES = ['mid']  # Size of the model
SPEC_TYPES = [1, 2]  # Different specification types

# Paths
vnn_dir_path = '../Benchmarks/vnnlib'  # Path for VNNLIB files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models
yaml_path = './decima_yaml'  # Path for YAML configuration files
running_result_path = './decima_abcrown_running_result'  # Path for saving results

# Create necessary directories if they don't exist
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path):
    """
    Creates a YAML configuration file for a specific Decima model.

    Args:
        yaml (str): Path to the YAML file to create.
        vnn_path (str): Path to the VNNLIB file.
        onnx_path (str): Path to the ONNX model file.
    """
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 1\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")

def main(abcrown_path, size=10):
    """
    Main function to run ABCrown verification for Decima models.

    Args:
        abcrown_path (str): Path to the ABCrown verifier executable/script.
        size (int): Number of verification instances to run for each specification type (default: 10).
    """
    for i in range(len(SPEC_TYPES)):
        for instance in range(size):  # Use the minimum of size or SIZES[i]
            for MODEL in MODEL_SIZES:
                # Construct paths for VNNLIB, ONNX model, and YAML configuration
                vnn_path = f'{vnn_dir_path}/decima_{SPEC_TYPES[i]}_{instance}.vnnlib'
                onnx_path = f'{onnx_dir_path}/decima_mid_{MODEL_TYPES[i]}.onnx'
                yaml = f'{yaml_path}/decima_{SPEC_TYPES[i]}_{instance}.yaml'

                # Create YAML configuration
                create_yaml(yaml, vnn_path, onnx_path)

                # Run ABCrown with the generated YAML configuration and save the output
                result_path = f'{running_result_path}/decima_{MODEL}_{SPEC_TYPES[i]}_{instance}.txt'
                command = f"python {abcrown_path} --config {yaml} | tee {result_path}"

                # Print and execute the command
                print(f"Executing: {command}")
                os.system(command)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run ABCrown verification for Decima models.")
    
    # Add argument for the path to the ABCrown verifier
    parser.add_argument("--abcrown_path", help="Path to the ABCrown verifier.")
    
    # Add optional argument for specifying the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run for each specification type (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.abcrown_path, args.size)
