import sys
sys.path.append('..')

import os
import json
import pandas as pd
import random

from datasets import load_dataset

import requests

def download_file(url, file_path):
    response = requests.get(url)
    response.raise_for_status()

    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "wb") as file:
        file.write(response.content)

def dump_json(data, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

current_dir = os.path.abspath("")
raw_data_dir = os.path.join(current_dir, 'raw')
processed_data_dir = os.path.join(current_dir, 'processed')

def download_advbench():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    raw_file_path = os.path.join(raw_data_dir, 'advbench.csv')
    processed_file_path = os.path.join(processed_data_dir, 'advbench.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)
    instructions = dataset['goal'].to_list()
    targets = dataset['target'].to_list()
    dataset_json = [{'instruction': instruction.strip(), 'target': target, 'category': None} for instruction, target in zip(instructions, targets)]
    dump_json(dataset_json, processed_file_path)

def download_malicious_instruct():
    url = 'https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt'
    raw_file_path = os.path.join(raw_data_dir, 'malicious_instruct.txt')
    processed_file_path = os.path.join(processed_data_dir, 'malicious_instruct.json')

    download_file(url, raw_file_path)
    with open(raw_file_path, 'r') as f:
        instructions = f.readlines()

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_tdc2023():
    urls = [
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json',
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json'
    ]
    raw_file_paths = [
        os.path.join(raw_data_dir, 'tdc2023_dev_behaviors.json'),
        os.path.join(raw_data_dir, 'tdc2023_test_behaviors.json')
    ]
    processed_file_path = os.path.join(processed_data_dir, 'tdc2023.json')

    for url, raw_file_path in zip(urls, raw_file_paths):
        download_file(url, raw_file_path)

    instructions = []
    for raw_file_path in raw_file_paths:
        with open(raw_file_path, 'r') as f:
            dataset = json.load(f)
        instructions.extend(dataset)

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_jailbreakbench():
    processed_file_path = os.path.join(processed_data_dir, 'jailbreakbench.json')

    import jailbreakbench as jbb
    dataset = jbb.read_dataset()

    instructions = dataset.goals
    targets = dataset.targets
    categories = dataset.categories

    dataset_json = [{'instruction': instruction.strip(), 'target': target, 'category': category} for instruction, target, category in zip(instructions, targets, categories)]
    dump_json(dataset_json, processed_file_path)

def download_harmbench(split):
    assert split in ['val', 'test']

    url = f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_{split}.csv'
    raw_file_path = os.path.join(raw_data_dir, f'harmbench_{split}.csv')
    processed_file_path = os.path.join(processed_data_dir, f'harmbench_{split}.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = []
    categories = []
    # filter out instructions with Category=copyright or Tags=context
    filtered_dataset = dataset[
        ~dataset['FunctionalCategory'].str.contains('copyright', case=False, na=False) &
        ~dataset['Tags'].str.contains('context', case=False, na=False)
    ]

    instructions.extend(filtered_dataset['Behavior'].to_list())
    categories.extend(filtered_dataset['SemanticCategory'].to_list())

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_strongreject():
    url = 'https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv'
    raw_file_path = os.path.join(raw_data_dir, 'strongreject.csv')
    processed_file_path = os.path.join(processed_data_dir, 'strongreject.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['forbidden_prompt'].to_list()
    categories = dataset['category'].to_list()

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_alpaca():
    hf_path = 'tatsu-lab/alpaca'
    processed_file_path = os.path.join(processed_data_dir, 'alpaca.json')

    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_circuit_breakers():
    """Download Circuit Breaker datasets from GraySwanAI."""
    base_url = 'https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/main/data'

    files = [
        ('circuit_breakers_train.json', 'circuit_breakers_train.json'),
        ('circuit_breakers_val.json', 'circuit_breakers_val.json'),
        ('harmbench_test.json', 'cb_harmbench_test.json'),  # Rename to avoid collision
    ]

    for remote_name, local_name in files:
        url = f'{base_url}/{remote_name}'
        raw_path = os.path.join(raw_data_dir, local_name)
        processed_path = os.path.join(processed_data_dir, local_name)

        download_file(url, raw_path)

        with open(raw_path, 'r') as f:
            data = json.load(f)

        # Handle dict with 'data' key (harmbench format)
        if isinstance(data, dict) and 'data' in data:
            data = data['data']

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Expected non-empty list in {remote_name}, got {type(data)}")

        # Convert to standard format
        if 'harmbench' in local_name:
            # Test format: use behavior as instruction, default_target as target
            required_keys = {'behavior', 'default_target'}
            if not required_keys.issubset(data[0].keys()):
                raise KeyError(f"Missing keys in {remote_name}: expected {required_keys}, got {data[0].keys()}")
            processed = [
                {'instruction': item['behavior'], 'target': item['default_target'], 'category': None}
                for item in data
            ]
        else:
            # Train/val format: use prompt as instruction, output as target
            required_keys = {'prompt', 'output'}
            if not required_keys.issubset(data[0].keys()):
                raise KeyError(f"Missing keys in {remote_name}: expected {required_keys}, got {data[0].keys()}")
            processed = [
                {'instruction': item['prompt'], 'target': item['output'], 'category': None}
                for item in data
            ]

        dump_json(processed, processed_path)
        print(f"Downloaded {remote_name}: {len(processed)} examples")

# download_advbench()
# download_malicious_instruct()
# download_tdc2023()
# download_jailbreakbench()
# download_harmbench(split='val')
# download_harmbench(split='test')
# download_strongreject()
download_circuit_breakers()

download_alpaca()

current_dir = os.path.abspath("")
splits_data_dir = os.path.join(current_dir, 'splits')

def construct_harmful_dataset_splits():
    """Construct harmful dataset splits using Circuit Breaker data."""
    harmful_train_path = os.path.join(splits_data_dir, 'harmful_train.json')
    harmful_val_path = os.path.join(splits_data_dir, 'harmful_val.json')
    harmful_test_path = os.path.join(splits_data_dir, 'harmful_test.json')

    # Load Circuit Breaker data directly
    with open(os.path.join(processed_data_dir, 'circuit_breakers_train.json'), 'r') as f:
        harmful_train = json.load(f)

    with open(os.path.join(processed_data_dir, 'circuit_breakers_val.json'), 'r') as f:
        harmful_val = json.load(f)

    with open(os.path.join(processed_data_dir, 'cb_harmbench_test.json'), 'r') as f:
        harmful_test = json.load(f)

    dump_json(harmful_train, harmful_train_path)
    dump_json(harmful_val, harmful_val_path)
    dump_json(harmful_test, harmful_test_path)

    print(f'harmful_train: {len(harmful_train)}')
    print(f'harmful_val: {len(harmful_val)}')
    print(f'harmful_test: {len(harmful_test)}')

def construct_harmless_dataset_splits():
    harmless_train_path = os.path.join(splits_data_dir, 'harmless_train.json')
    harmless_val_path = os.path.join(splits_data_dir, 'harmless_val.json')
    harmless_test_path = os.path.join(splits_data_dir, 'harmless_test.json')

    train_p, val_p, test_p = 0.6, 0.20, 0.20

    harmless_instructions = []
    for file in ['alpaca.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmless_instructions.extend(json.load(f))

    random.seed(42)
    random.shuffle(harmless_instructions)

    total_size = len(harmless_instructions)
    train_size = int(train_p * total_size)
    val_size = int(val_p * total_size)

    harmless_train_instructions = harmless_instructions[:train_size]
    harmless_val_instructions = harmless_instructions[train_size:train_size+val_size]
    harmless_test_instructions = harmless_instructions[train_size+val_size:]

    dump_json(harmless_train_instructions, harmless_train_path)
    dump_json(harmless_val_instructions, harmless_val_path)
    dump_json(harmless_test_instructions, harmless_test_path)

construct_harmful_dataset_splits()
construct_harmless_dataset_splits()



