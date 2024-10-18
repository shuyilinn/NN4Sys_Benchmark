"""
This script runs ABCrown verification for Cardinality Estimation models using ONNX models.
It generates YAML configurations for each model and stores the results in the specified directory.
"""

import os
import argparse

# Set MKL threading layer to GNU
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Model names for Cardinality Estimation
MODEL_NAMES = ["mscn_128d", "mscn_128d_dual", "mscn_2048d", "mscn_2048d_dual"]

# Paths
vnn_dir_path = '../Benchmarks/vnnlib'  # Path for vnnlib files
onnx_dir_path = '../Benchmarks/onnx'  # Path for ONNX models
yaml_path = './cardinality_yaml'  # Path for YAML configuration files
running_result_path = './cardinality_abcrown_running_result'  # Path for saving results

# Create necessary directories if they don't exist
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path):
    """
    Creates a YAML configuration file for a specific Cardinality Estimation model.

    Args:
        yaml (str): Path to the YAML file to create.
        vnn_path (str): Path to the VNNLIB file.
        onnx_path (str): Path to the ONNX model file.
    """
    with open(yaml, mode='w') as f:
        f.write(
            "general:\n  enable_incomplete_verification: False\n  loss_reduction_func: max\n  conv_mode: matrix\n"
            f"model:\n  onnx_path: {onnx_path}\n"
            f"specification:\n  vnnlib_path: {vnn_path}\n"
            "solver:\n  batch_size: 50  # Number of parallel domains to compute on GPU.\n  "
            "bound_prop_method: forward+backward\n  beta-crown:\n    iteration: 10  # Iterations for intermediate bounds.\n"
            "bab:\n  initial_max_domains: 100000\n  branching:\n    method: naive  # Split on input space.\n    input_split:\n"
            "      enable: True\n      adv_check: .inf\n"
        )

def main(abcrown_path, size=10):
    """
    Main function to run ABCrown verification for Cardinality Estimation models.

    Args:
        abcrown_path (str): Path to the ABCrown verifier executable/script.
        size (int): Number of verification instances to run (default: 10).
    """
    for i in range(size):
        for model in MODEL_NAMES:
            # Construct paths for VNNLIB, ONNX model, and YAML configuration
            vnn_path = f'{vnn_dir_path}/{model}_{i}.vnnlib'
            onnx_path = f'{onnx_dir_path}/{model}.onnx'
            yaml = os.path.join(yaml_path, f'{model}_{i}.yaml')

            # Create YAML configuration
            create_yaml(yaml, vnn_path, onnx_path)

            # Run ABCrown with the generated YAML configuration and save the output
            result_path = os.path.join(running_result_path, f'{model}_{i}.txt')
            command = f"python {abcrown_path} --config {yaml} | tee {result_path}"

            # Print and execute the command
            print(f"Executing: {command}")
            os.system(command)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run ABCrown verification for Cardinality Estimation models.")
    
    # Add argument for the path to the ABCrown verifier
    parser.add_argument("abcrown_path", help="Path to the ABCrown verifier.")
    
    # Add optional argument for specifying the number of verification instances (default: 10)
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.abcrown_path, args.size)
