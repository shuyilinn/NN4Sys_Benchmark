import argparse
import Aurora_abcrown_run
import Pensieve_abcrown_run
import Bloom_filter_abcrown_run
import Lindex_abcrown_run
import Cardinality_abcrown_run
import Decima_abcrown_run

# This script is used to verify models using abcrown

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run abcrown verification for selected models.")

    # Add arguments
    parser.add_argument(
        "--model", 
        default="all", 
        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
        help="Specify which model to verify: pensieve, decima, lindex, cardinality, bloom_filter, aurora, or all."
    )
    parser.add_argument(
        "--path", 
        required=True, 
        help="Path to the verifier executable or script."
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Function map to associate model names with corresponding function calls
    model_functions = {
        "pensieve": Pensieve_abcrown_run.main,
        "decima": Decima_abcrown_run.main,
        "lindex": Lindex_abcrown_run.main,
        "cardinality": Cardinality_abcrown_run.main,
        "bloom_filter": Bloom_filter_abcrown_run.main,
        "aurora": Aurora_abcrown_run.main
    }

    # Run verification for the specified model(s)
    if args.model == "all":
        # Verify all models
        for model, func in model_functions.items():
            print(f"Running abcrown verification for {model}...")
            func(args.path)
        print(f"[Done] Finished running abcrown verification for all models")
    else:
        # Verify only the specified model
        print(f"Running abcrown verification for {args.model}...")
        model_functions[args.model](args.path)
        print(f"[Done] Finished running abcrown verification for {args.model}")


if __name__ == "__main__":
    main()
