'''
This script is to generate the fixed input pool for the benchmark.
'''

import argparse
import os

def run_script(directory):
    """Helper function to run the script in the specified directory."""
    os.system(f"cd {directory} && python shuyi_gen_upper.py && cd ..")

def print_unavailable_message(model_name):
    """Helper function to print an unavailable message for specific models."""
    print(f"Sorry, but currently we do not provide a generation script for {model_name}. "
          "Feel free to contact us if you need any help.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
                        help="Specify which model to generate: pensieve, decima, lindex, cardinality, bloom_filter, aurora, or all.")

    args = parser.parse_args()

    if args.model == "all":
        # Run the generation script for all available models
        for model in ["Pensieve", "Decima", "Aurora"]:
            run_script(model)
    elif args.model == "pensieve":
        run_script("Pensieve")
    elif args.model == "decima":
        run_script("Decima")
    elif args.model == "aurora":
        run_script("Aurora")
    else:
        # Handle unavailable models
        print_unavailable_message(args.model)

if __name__ == "__main__":
    main()
