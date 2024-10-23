'''
Script to run verification using Marabou for selected models
'''

import argparse
import Aurora_marabou_run
import Pensieve_marabou_run
import Bloom_filter_marabou_run
import Lindex_marabou_run
import Decima_marabou_run

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Marabou verification for selected models.")

    # Add arguments
    parser.add_argument(
        "--model", 
        default="all", 
        choices=["pensieve", "decima", "lindex", "bloom_filter", "aurora", "all"], 
        help="Specify which model to verify: pensieve, decima, lindex, bloom_filter, aurora, or all."
    )
    parser.add_argument(
        "--path", 
        required=True, 
        help="Path to the verifier executable or script."
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=10, 
        help="Number of verification instances to run for each model (default: 10)."
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Map model names to corresponding functions
    model_functions = {
        "pensieve": Pensieve_marabou_run.main,
        "decima": Decima_marabou_run.main,
        "lindex": Lindex_marabou_run.main,
        "bloom_filter": Bloom_filter_marabou_run.main,
        "aurora": Aurora_marabou_run.main
    }

    # Run verification for the specified model(s)
    if args.model == "all":
        # Run all models
        for model, func in model_functions.items():
            print(f"Running Marabou verification for {model} with {args.size} instances...")
            func(args.path, args.size)
            print(f"[Done] Finished running marabou verification for all models")
    else:
        # Run a single specified model
        print(f"Running Marabou verification for {args.model} with {args.size} instances...")
        model_functions[args.model](args.path, args.size)
        print(f"[Done] Finished running marabou verification for {args.model}")


if __name__ == "__main__":
    main()
