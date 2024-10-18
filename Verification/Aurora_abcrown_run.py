"""
This script is to verify Aurora with abcrown.
"""

import os
import argparse

# Set MKL threading layer to GNU
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Define model configurations
MODELS = ['small', 'mid', 'big']  # Aurora models (small, mid, big)
MODEL_TYPES = ['simple', 'simple', 'simple', 'parallel', 'concat']  # Model types

# Specifications for Aurora
SPEC_TYPES = [101, 102, 2, 3, 4]  # Different specification types

# Directory paths
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = './aurora_yaml'
running_result_path = './aurora_abcrown_running_result'

# Timeout for each verification task (in seconds)
timeout = 100

# Initialize additional parameters
csv_data = []
total_num = 0
current_gpu = 0
DIMENSION_NUMBERS = [1, 2, 3]  # Different dimensions for the tasks
P_RANGE = [0.8, 1, 1.2, 1.4, 1.6]  # Parameter ranges to test

# Create necessary directories if they do not exist
if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    """
    Create a YAML configuration file for abcrown to run with given model and specification paths.

    Args:
        yaml (str): Path where the YAML file will be created.
        vnn_path (str): Path to the vnnlib specification file.
        onnx_path (str): Path to the onnx model file.
        inputshape (int): Input shape of the model (default is 6).
    """
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 2048\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True"
        )

def main(abcrown_path, size = 10):
    """
    Main function to run abcrown verification on Aurora models.

    Args:
        abcrown_path (str): Path to the abcrown verifier.
        size (int): Number of verification instances to generate for each configuration.
    """
    for i in range(len(SPEC_TYPES)):
        for MODEL in MODELS:
            for instance in range(size):  # Using the size argument to control how many instances to run
                for range_ptr in range(len(P_RANGE)):
                    for d_ptr in range(len(DIMENSION_NUMBERS)):
                        dimension_number = DIMENSION_NUMBERS[d_ptr]
                        
                        # Only continue if dimension number is 3 and range pointer is 1
                        if dimension_number != 3 or range_ptr != 1:
                            continue

                        # Construct paths for vnnlib specification and onnx model
                        vnn_path = f'{vnn_dir_path}/aurora_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{instance}.vnnlib'
                        onnx_path = f'{onnx_dir_path}/aurora_{MODEL}_{MODEL_TYPES[i]}.onnx'
                        yaml = f'{yaml_path}/aurora_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{instance}.yaml'

                        # Create YAML configuration file
                        create_yaml(yaml, vnn_path, onnx_path)

                        # Run abcrown and save the output
                        os.system(f"python {abcrown_path} --config {yaml} | tee {running_result_path}/aurora_{MODEL}_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{instance}.txt")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run abcrown verification on Aurora models.")
    
    # Add required abcrown path argument
    parser.add_argument("abcrown_path", help="Path to the abcrown verifier.")
    
    # Add optional size argument with default value of 10
    parser.add_argument("--size", type=int, default=10, help="Number of verification instances to run (default: 10).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.abcrown_path, args.size)
