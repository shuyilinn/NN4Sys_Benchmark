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
from cheng_gen import main as index_main
from decima_gen import main as decima_main
from aurora_gen import main as aurora_main
from pensieve_gen import main as pensieve_main
from bloom_filter_gen import main as bloom_filter_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=2024,
                        help="Random seed, default is 2024")
    
    args = parser.parse_args()
    seed = args.seed

    # Create 'vnnlib' directory if it doesn't exist
    if not os.path.exists('vnnlib'):
        os.makedirs('vnnlib')

    csv_data = []
    
    # Generate Index specifications
    print("Generating Index specifications, this may take several minutes...")
    index_data = index_main(seed)
    if index_data:
        csv_data.extend(index_data)

    # Generate Cardinality specifications
    print("Generating Cardinality specifications, this may take around ten minutes...")
    card_data = card_main(seed)
    if card_data:
        csv_data.extend(card_data)

    # Generate Decima specifications
    print("Generating Decima specifications, this may take around one minute...")
    decima_data = decima_main(seed)
    if decima_data:
        csv_data.extend(decima_data)

    # Generate Aurora specifications
    print("Generating Aurora specifications, this may take around one minute...")
    aurora_data = aurora_main(seed)
    if aurora_data:
        csv_data.extend(aurora_data)

    # Generate Pensieve specifications
    print("Generating Pensieve specifications, this may take around one minute...")
    pensieve_data = pensieve_main(seed)
    if pensieve_data:
        csv_data.extend(pensieve_data)

    # Generate Bloom Filter specifications
    print("Generating Bloom Filter specifications, this may take around one minute...")
    bloom_filter_data = bloom_filter_main(seed)
    if bloom_filter_data:
        csv_data.extend(bloom_filter_data)

    # Check if csv_data is not empty before processing. This csv file is simply for VNN competition.
    if csv_data:
        total_timeout = sum([int(row[-1]) for row in csv_data if row and row[-1].isdigit()])
        print(f"Total timeout is {total_timeout}")

        # Write CSV data to file
        with open('instances.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        print(f"Successfully generated {len(csv_data)} files!")
    else:
        print("No data generated or CSV data is empty.")
