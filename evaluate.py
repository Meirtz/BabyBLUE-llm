import json
import os
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

methods = ['ALL']
data_dir = 'results'
json_list = []

if data_dir == 'results':
    use_babyblue = True

if 'ALL' in methods:
    methods = os.listdir(data_dir)
    methods = [s for s in methods if s != 'text_behaviors.json']
    json_list = [f'{data_dir}/{f}' for f in methods if f.endswith('.json')]
else:
    for method in methods:
        if use_babyblue:
            json_file = f'{data_dir}/{method}_with_babyblue.json'
        else:
            json_file = f'{data_dir}/{method}.json'
        json_list.append(json_file)


def json_element_count(json_data):
    query_count = 0
    case_count = 0
    for key in json_data.keys():
        query_count += 1
        if isinstance(json_data[key], list):
            json_data[key] = {list(datapoint.keys())[0]: datapoint[list(datapoint.keys())[0]] for datapoint in
                              json_data[key]}
        for sub_key in json_data[key].keys():
            case_count += 1
    return query_count, case_count


def get_unique_query(json_data):
    query_set = set()
    for key in json_data.keys():
        query_set.add(key)
    return query_set


def calculate_asr(json_data, models=['llama2_7b', 'llama2_13b']):
    results = {}
    model_query_counts = {}

    if 'ALL' in models:
        models = set()
        for query in json_data.keys():
            models.update(json_data[query].keys())
        models = list(models)

    for model in models:
        results[model] = {'harmbech': 0, 'advbench': 0, 'babyblue': 0, 'harmbech_asr': 0, 'advbench_asr': 0,
                          'babyblue_asr': 0, 'total_queries': 0}
        model_query_counts[model] = 0

    for query in tqdm(json_data.keys()):
        for model in models:
            if model in json_data[query]:
                results[model]['harmbech'] += json_data[query][model]['label']
                results[model]['advbench'] += json_data[query][model].get('advbench_label', 0)
                results[model]['babyblue'] += json_data[query][model].get('babyblue_label', 0)
                model_query_counts[model] += 1

    for model in models:
        if model_query_counts[model] > 0:
            results[model]['harmbech_asr'] = round(results[model]['harmbech'] / model_query_counts[model], 1)
            results[model]['advbench_asr'] = round(results[model]['advbench'] / model_query_counts[model], 1)
            results[model]['babyblue_asr'] = round(results[model]['babyblue'] / model_query_counts[model], 1)
            results[model]['total_queries'] = model_query_counts[model]

    return results


data = []
query_count = 0
case_count = 0
unique_query = set()
overall_results = {}

for json_file in json_list:
    with open(json_file) as f:
        partial_data = json.load(f)
        print(f'Loaded {json_file}')
        data.append(partial_data)

        unique_query = unique_query.union(get_unique_query(partial_data))
        q, c = json_element_count(partial_data)
        query_count += q
        case_count += c

        results = calculate_asr(partial_data, ['ALL'])
        for model in results.keys():
            if model not in overall_results:
                overall_results[model] = {}
            method_name = json_file.split('/')[-1].replace('.json', '').replace('_with_babyblue', '')
            overall_results[model][method_name] = results[model]

print(f'query count: {query_count}, case count: {case_count}')
print(f'unique query count: {len(unique_query)}')

# Create a mapping dictionary
model_mapping = {
    'llama2_7b': 'Llama 2 7B Chat',
    'llama2_13b': 'Llama 2 13B Chat',
    'llama2_70b': 'Llama 2 70B Chat',
    'vicuna_7b_v1_5': 'Vicuna 7B',
    'vicuna_13b_v1_5': 'Vicuna 13B',
    'baichuan2_7b': 'Baichuan 2 7B',
    'baichuan2_13b': 'Baichuan 2 13B',
    'qwen_7b_chat': 'Qwen 7B Chat',
    'qwen_14b_chat': 'Qwen 14B Chat',
    'qwen_72b_chat': 'Qwen 72B Chat',
    'koala_7b': 'Koala 7B',
    'koala_13b': 'Koala 13B',
    'orca_2_7b': 'Orca 2 7B',
    'orca_2_13b': 'Orca 2 13B',
    'solar_10_7b_instruct': 'Solar 10.7b-Instruct',
    'mistral_7b_v2': 'Mistral 7B',
    'mixtral_8x7b': 'Mixtral 8x7B',
    'openchat_3_5_1210': 'OpenChat 3.5 1210',
    'starling_7b': 'Starling 7B',
    'zephyr_7b_robust6_data2': 'Zephyr 7B',
    'gpt-3.5-turbo-0613': 'GPT 3.5 Turbo 0613',
    'gpt-3.5-turbo-1106': 'GPT 3.5 Turbo 1106',
    'gpt-4-0613': 'GPT 4 0613',
    'gpt-4-1106-preview': 'GPT 4 Turbo 1106',
    'claude-instant-1': 'Claude 1',
    'claude-2': 'Claude 2',
    'claude-2.1': 'Claude 2.1',
    'gemini': 'Gemini Pro'
}

method_mapping = {
    'GCG': 'GCG',
    'GCG-M': 'GCG-M',
    'PEZ': 'PEZ',
    'GBDA': 'GBDA',
    'UAT': 'UAT',
    'AutoPrompt': 'AP',
    'FewShot': 'SFS',
    'AutoDAN': 'AutoDAN',
    'GCG-T': 'GCG-T',
    'ZeroShot': 'ZS',
    'PAIR': 'PAIR',
    'TAP': 'TAP',
    'TAP-T': 'TAP-T',
    'PAP': 'PAP-top5',
    'HumanJailbreaks': 'Human',
    'DirectRequest': 'DR'
}

# Apply the mappings to rename the models and methods
renamed_results = {}
for model, methods_scores in overall_results.items():
    new_model_name = model_mapping.get(model, model)
    renamed_results[new_model_name] = {}
    for method, scores in methods_scores.items():
        new_method_name = method_mapping.get(method, method)
        renamed_results[new_model_name][new_method_name] = scores

# Create separate tables for white-box and black-box models
white_box_models = [
    'Llama 2 7B Chat', 'Llama 2 13B Chat', 'Llama 2 70B Chat',
    'Vicuna 7B', 'Vicuna 13B', 'Baichuan 2 7B', 'Baichuan 2 13B',
    'Qwen 7B Chat', 'Qwen 14B Chat', 'Qwen 72B Chat', 'Koala 7B', 'Koala 13B',
    'Orca 2 7B', 'Orca 2 13B', 'Solar 10.7b-Instruct', 'Mistral 7B', 'Mixtral 8x7B',
    'OpenChat 3.5 1210', 'Starling 7B', 'Zephyr 7B'
]
black_box_models = [
    'GPT 3.5 Turbo 0613', 'GPT 3.5 Turbo 1106', 'GPT 4 0613', 'GPT 4 Turbo 1106',
    'Claude 1', 'Claude 2', 'Claude 2.1', 'Gemini Pro'
]

white_box_methods = ['GCG', 'GCG-M', 'PEZ', 'GBDA', 'UAT', 'AP', 'SFS', 'AutoDAN']
black_box_methods = ['GCG-T', 'ZS', 'PAIR', 'TAP', 'TAP-T', 'PAP-top5', 'Human', 'DR']

# Filter results for white-box models
white_box_results = {model: methods_scores for model, methods_scores in renamed_results.items() if model in white_box_models}
black_box_results = {model: methods_scores for model, methods_scores in renamed_results.items() if model in black_box_models}

# Ensure models and methods are sorted according to the specified order
white_box_results_sorted = {model: white_box_results[model] for model in white_box_models if model in white_box_results}
black_box_results_sorted = {model: black_box_results[model] for model in black_box_models if model in black_box_results}

# Create DataFrame for white-box models
white_box_table_data = {}
for model, methods_scores in white_box_results_sorted.items():
    white_box_table_data[model] = []
    for method in white_box_methods:
        if method in methods_scores:
            scores = methods_scores[method]
            white_box_table_data[model].append(f"{scores['advbench_asr']} / {scores['harmbech_asr']} / {scores['babyblue_asr']}")
        else:
            white_box_table_data[model].append("-")

white_box_df = pd.DataFrame.from_dict(white_box_table_data, orient='index', columns=white_box_methods)
white_box_df.index.name = 'Model'
white_box_df.reset_index(inplace=True)

# Create DataFrame for black-box models
black_box_table_data = {}
for model, methods_scores in black_box_results_sorted.items():
    black_box_table_data[model] = []
    for method in black_box_methods:
        if method in methods_scores:
            scores = methods_scores[method]
            black_box_table_data[model].append(f"{scores['advbench_asr']} / {scores['harmbech_asr']} / {scores['babyblue_asr']}")
        else:
            black_box_table_data[model].append("-")

black_box_df = pd.DataFrame.from_dict(black_box_table_data, orient='index', columns=black_box_methods)
black_box_df.index.name = 'Model'
black_box_df.reset_index(inplace=True)

# Save the white-box model DataFrame to CSV
white_box_csv_file = './white_box_model_metrics.csv'
white_box_df.to_csv(white_box_csv_file, index=False)
print(f"White-box CSV file saved to {white_box_csv_file}")

# Save the black-box model DataFrame to CSV
black_box_csv_file = './black_box_model_metrics.csv'
black_box_df.to_csv(black_box_csv_file, index=False)
print(f"Black-box CSV file saved to {black_box_csv_file}")

# Convert the DataFrames to LaTeX tables
white_box_latex_table = tabulate(white_box_df, headers='keys', tablefmt='latex', showindex=False)
black_box_latex_table = tabulate(black_box_df, headers='keys', tablefmt='latex', showindex=False)

# Add table environment and captions
white_box_latex_table = (
    "\\begin{table*}[htbp]\n\\centering\n"
    + white_box_latex_table
    + "\n\\caption{White-Box Model Metrics}\n\\label{tab:white_box_model_metrics}\n\\end{table*}"
)
black_box_latex_table = (
    "\\begin{table*}[htbp]\n\\centering\n"
    + black_box_latex_table
    + "\n\\caption{Black-Box Model Metrics}\n\\label{tab:black_box_model_metrics}\n\\end{table*}"
)

# Save the LaTeX tables to .tex files
white_box_tex_file = './white_box_model_metrics_table.tex'
with open(white_box_tex_file, 'w') as f:
    f.write(white_box_latex_table)
print(f"White-box LaTeX table saved to {white_box_tex_file}")

black_box_tex_file = './black_box_model_metrics_table.tex'
with open(black_box_tex_file, 'w') as f:
    f.write(black_box_latex_table)
print(f"Black-box LaTeX table saved to {black_box_tex_file}")

