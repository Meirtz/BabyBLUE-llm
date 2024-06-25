import json
import os
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

# 定义映射
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

ordered_models = [
    'Llama 2 7B Chat', 'Llama 2 13B Chat', 'Llama 2 70B Chat',
    'Vicuna 7B', 'Vicuna 13B', 'Baichuan 2 7B', 'Baichuan 2 13B',
    'Qwen 7B Chat', 'Qwen 14B Chat', 'Qwen 72B Chat', 'Koala 7B', 'Koala 13B',
    'Orca 2 7B', 'Orca 2 13B', 'Solar 10.7b-Instruct', 'Mistral 7B', 'Mixtral 8x7B',
    'OpenChat 3.5 1210', 'Starling 7B', 'Zephyr 7B', 
    'GPT 3.5 Turbo 0613', 'GPT 3.5 Turbo 1106', 'GPT 4 0613', 'GPT 4 Turbo 1106',
    'Claude 1', 'Claude 2', 'Claude 2.1', 'Gemini Pro'
]

ordered_methods = [
    'GCG', 'GCG-M', 'GCG-T', 'PEZ', 'GBDA', 'UAT', 'AP', 'SFS', 'ZS', 'PAIR', 'TAP', 'TAP-T', 'AutoDAN', 'PAP-top5', 'Human', 'DR'
]

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

# 创建表格数据
table_data = {}
methods = sorted(set(method for scores in overall_results.values() for method in scores.keys()))

for model, methods_scores in overall_results.items():
    mapped_model = model_mapping.get(model, model)
    table_data[mapped_model] = []
    for method in methods:
        mapped_method = method_mapping.get(method, method)
        if method in methods_scores:
            scores = methods_scores[method]
            table_data[mapped_model].append(f"{scores['advbench_asr']} / {scores['harmbech_asr']} / {scores['babyblue_asr']}")
        else:
            table_data[mapped_model].append("-")

# 按照指定顺序排序
ordered_table_data = {model: table_data.get(model, ["-"] * len(methods)) for model in ordered_models}

# 创建DataFrame并保存为CSV
df = pd.DataFrame.from_dict(ordered_table_data, orient='index', columns=ordered_methods)
df.index.name = 'Model'
df.reset_index(inplace=True)

output_csv_file = './model_metrics.csv'
df.to_csv(output_csv_file, index=False)
print(f"CSV file saved to {output_csv_file}")

# 将DataFrame转换为LaTeX表格
latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

# 添加表格环境并设置跨两列
latex_table = (
    "\\begin{table*}[htbp]\n\\centering\n\\resizebox{2\\columnwidth}{!}{%\n"
    "\\begin{tabular}{l" + "l" * len(ordered_methods) + "}\n\\hline\n\\hline\n"
    + latex_table.replace('Model', '\\textbf{Model}') + "\n\\hline\n\\hline\n"
    "\\end{tabular}}\n"
    "\\caption{Model Metrics}\n\\label{tab:model_metrics}\n\\end{table*}"
)

# 保存LaTeX表格到.tex文件
output_tex_file = './model_metrics_table.tex'
with open(output_tex_file, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table saved to {output_tex_file}")
