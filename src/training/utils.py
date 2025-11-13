import yaml

def load_experiment_config(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data
