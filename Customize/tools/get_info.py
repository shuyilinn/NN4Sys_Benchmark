import yaml


def get_info():

    # Read the YAML configuration file
    with open('config_file.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Access values
    pt_path = config['paths']['pt_path']
    input_list = config['inputs']['input_list']
    perturbation_list = config['inputs']['perturbation_list']
    output_range = config['outputs']['output_range']
    output_number = config['outputs']['output_number']
    output_path = config['other']['output_path']
    number = config['other']['number']

    # Use the configuration values as needed
    return pt_path, input_list, perturbation_list, output_range, output_path, number
