"""
This script runs verification on Bloom Filter models using ABCrown, generating YAML configurations 
for each instance and saving the results.
"""

import os
import sys
import argparse

# Set MKL threading layer to GNU
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Paths
vnn_dir_path = '../Benchmarks/vnnlib'  # Path for vnnlib files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models
yaml_path = './bloom_filter_yaml'  # Path for YAML configuration files
running_result_path = './bloom_filter_abcrown_running_result'  # Path for saving results

# Create necessary directories if they don't exist
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    """
    Creates a YAML configuration file for a specific Bloom Filter model verification instance.

    Args:
        yaml (str): Path to the YAML file to create.
        vnn_path (str): Path to the VNNLIB file.
        onnx_path (str): Path to the ONNX model file.
        inputshape (int): Input shape of the model (default: 6).
    """
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write("solver:\n  batch_size: 2048\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")

def main(abcrown_path, size):
    """
    Runs ABCrown verification for Bloom Filter models, generating YAML and running the verification.

    Args:
        abcrown_path (str): Path to the ABCrown verifier.
        size (int): Number of verification instances to run.
    """
    for i in range(size=10):
        # Construct paths for VNNLIB, ONNX model, and YAML configuration
        vnn_path = f'{vnn_dir_path}/bloom_filter_{i}.vnnlib'
        onnx_path = os.path.join(onnx_dir_path, 'bloom_filter.onnx')
        yaml = os.path.join(yaml_path, f'bloom_filter_{i}.yaml')

        # Create YAML configuration
        create_yaml(yaml, vnn_path, onnx_path)

        # Run ABCrown with the generated YAML configuration and save the output
        result_path = os.path.join(running_result_path, f'bloom_filter_{i}.txt')
        command = f"python {abcrown_path} --config {yaml} | tee {result_path}"
        print(f"Executing command: {command}")
        os.system(command)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run ABCrown verification on Bloom Filter models.")
    
    # Add argument for the path to the ABCrown verifier
    parser.add_argument("--abcrown_path", help="Path to the ABCrown verifier.")
    
    # Add optional argument for specifying the number of instances to run (default is 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.abcrown_path, args.size)
