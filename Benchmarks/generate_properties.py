'''
This script generates properties, which are instances, for verifying neural networks. 
These properties include two formats:
- VNNLIB format: A standardized format used in verification tools for defining the constraints and expected outputs of neural networks.
- TXT format: Another format containing similar information for compatibility with other verification tools or processes.
'''

import os
import csv
import argparse
from haoyu_gen import main as card_main
from cheng_gen import main as lindex_main
from decima_gen import main as decima_main
from aurora_gen import main as aurora_main
from pensieve_gen import main as pensieve_main
from bloom_filter_gen import main as bloom_filter_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=2024, type=int,
                        help="Random seed, default is 2024")
    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
                        help="Specify which model to generate: pensieve, decima, lindex, cardinality, bloom_filter, aurora, or all.")

    args = parser.parse_args()
    seed = args.seed

    # Create 'vnnlib' directory if it doesn't exist
    if not os.path.exists('vnnlib'):
        os.makedirs('vnnlib')

    csv_data = []
    
    def generate_and_append(model_main_func, model_name, time_estimate):
        """
        Helper function to generate specifications for a model and append its data to csv_data.
        Prints out the progress and estimated time.
        """
        print(f"Generating {model_name} specifications, this may take {time_estimate}...")
        model_data = model_main_func(seed)
        if model_data:
            csv_data.extend(model_data)
        print(f"Finished generating {model_name} specifications.")

    if args.model == "all" or args.model == "index":
        generate_and_append(index_main, "Index", "several minutes")

    if args.model == "all" or args.model == "cardinality":
        generate_and_append(card_main, "Cardinality", "around ten minutes")

    if args.model == "all" or args.model == "decima":
        generate_and_append(decima_main, "Decima", "around one minute")

    if args.model == "all" or args.model == "aurora":
        generate_and_append(aurora_main, "Aurora", "around one minute")

    if args.model == "all" or args.model == "pensieve":
        generate_and_append(pensieve_main, "Pensieve", "around one minute")

    if args.model == "all" or args.model == "bloom_filter":
        generate_and_append(bloom_filter_main, "Bloom Filter", "around one minute")

    if args.model == "lindex":
        # Generate Lindex specifications
        generate_and_append(lindex_main, "Lindex", "several minutes")

    # Check if csv_data is not empty before processing. This csv file is simply for VNN competition.
    if csv_data:
        # Safely handle the timeout sum calculation by ensuring that the last element is numeric
        total_timeout = sum([int(row[-1]) for row in csv_data if row and isinstance(row[-1], str) and row[-1].isdigit()])
        print(f"Total timeout is {total_timeout}")

        # Write CSV data to file
        with open('instances.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        print(f"Successfully generated {len(csv_data)} files!")
    else:
        print("No data generated or CSV data is empty.")