"""
This script is to verify Aurora models with abcrown.
"""

import os
import argparse
import subprocess

# Set MKL threading layer to GNU
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Define model configurations
MODELS = ['small', 'mid', 'big']  # Aurora models (small, mid, big)
MODEL_TYPES = ['simple', 'simple', 'simple', 'parallel', 'concat']  # Model types

# Specifications for Aurora
SPEC_TYPES = [101, 102, 2, 3, 4]  # Different specification types

# Directory paths
VNN_DIR_PATH = '../Benchmarks/vnnlib'
ONNX_DIR_PATH = '../Benchmarks/onnx'
YAML_PATH = './aurora_yaml'
RUNNING_RESULT_PATH = './aurora_abcrown_running_result'

# Timeout for each verification task (in seconds)
TIMEOUT = 100

# Initialize additional parameters
CSV_DATA = []
TOTAL_NUM = 0
DIMENSION_NUMBERS = [1, 2, 3]  # Different dimensions for the tasks
P_RANGE = [0.8, 1, 1.2, 1.4, 1.6]  # Parameter ranges to test

# Create necessary directories if they do not exist
os.makedirs(RUNNING_RESULT_PATH, exist_ok=True)
os.makedirs(YAML_PATH, exist_ok=True)

def create_yaml(yaml: str, vnn_path: str, onnx_path: str) -> None:
    """
    Create a YAML configuration file for abcrown to run with given model and specification paths.

    Args:
        yaml (str): Path where the YAML file will be created.
        vnn_path (str): Path to the vnnlib specification file.
        onnx_path (str): Path to the onnx model file.
    """
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 2048\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True"
        )

def main(abcrown_path: str, size: int = 10) -> None:
    """
    Main function to run abcrown verification on Aurora models.

    Args:
        abcrown_path (str): Path to the abcrown verifier.
        size (int): Number of verification instances to generate for each configuration.
    """
    for i in range(len(SPEC_TYPES)):
        for model in MODELS:
            for instance in range(size):
                for range_ptr in range(len(P_RANGE)):
                    for d_ptr in range(len(DIMENSION_NUMBERS)):
                        dimension_number = DIMENSION_NUMBERS[d_ptr]
                        
                        # Only continue if dimension number is 3 and range pointer is 1
                        if dimension_number != 3 or range_ptr != 1:
                            continue

                        # Construct paths for vnnlib specification and onnx model
                        vnn_path = f'{VNN_DIR_PATH}/aurora_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{instance}.vnnlib'
                        onnx_path = f'{ONNX_DIR_PATH}/aurora_{model}_{MODEL_TYPES[i]}.onnx'
                        yaml = f'{YAML_PATH}/aurora_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{instance}.yaml'

                        # Create YAML configuration file
                        create_yaml(yaml, vnn_path, onnx_path)

                        # Run abcrown and save the output
                        output_file = f"{RUNNING_RESULT_PATH}/aurora_{model}_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{instance}.txt"
                        with open(output_file, 'w') as f:
                            subprocess.run(['python', abcrown_path, '--config', yaml], stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Aurora models with abcrown.")
    parser.add_argument('--abcrown_path', type=str, help="Path to the abcrown verifier.")
    parser.add_argument('--size', type=int, default=10, help="Number of verification instances to generate for each configuration (default: 10).")
    args = parser.parse_args()
    main(args.abcrown_path, args.size)