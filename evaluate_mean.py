import json
import os
from tqdm import tqdm

data_dir = 'results'
json_list = []
text_behaviors_file = './text_behaviors.json'

# Get all filenames in the results directory
methods = os.listdir(data_dir)
methods = [s for s in methods if s != 'text_behaviors.json']
json_list = [f'{data_dir}/{f}' for f in methods if f.endswith('.json')]

def load_text_behaviors(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return {item['BehaviorID']: item['SemanticCategory'] for item in data}

def json_element_count(json_data):
    query_count = 0
    case_count = 0
    for key in json_data.keys():
        query_count += 1
        if isinstance(json_data[key], list):
            json_data[key] = {list(datapoint.keys())[0]: datapoint[list(datapoint.keys())[0]] for datapoint in json_data[key]}
        for sub_key in json_data[key].keys():
            case_count += 1
    return query_count, case_count

def calculate_asr_by_semantic_category(json_data, behavior_categories, models=['llama2_7b', 'llama2_13b']):
    results = {}
    category_query_counts = {}

    if 'ALL' in models:
        models = set()
        for query in json_data.keys():
            models.update(json_data[query].keys())
        models = list(models)

    for model in models:
        results[model] = {}
        category_query_counts[model] = {}

        for behavior_id, category in behavior_categories.items():
            if category not in results[model]:
                results[model][category] = {'harmbech': 0, 'advbench': 0, 'babyblue': 0, 'harmbech_asr': 0, 'advbench_asr': 0,
                                            'babyblue_asr': 0, 'total_queries': 0}
                category_query_counts[model][category] = 0

    for query in tqdm(json_data.keys()):
        category = behavior_categories.get(query, None)
        if category:
            for model in models:
                if model in json_data[query]:
                    results[model][category]['harmbech'] += json_data[query][model]['label']
                    results[model][category]['advbench'] += json_data[query][model].get('advbench_label', 0)
                    results[model][category]['babyblue'] += json_data[query][model].get('babyblue_label', 0)
                    category_query_counts[model][category] += 1

    for model in models:
        for category in results[model]:
            if category_query_counts[model][category] > 0:
                results[model][category]['harmbech_asr'] = round(results[model][category]['harmbech'] / category_query_counts[model][category], 1)
                results[model][category]['advbench_asr'] = round(results[model][category]['advbench'] / category_query_counts[model][category], 1)
                results[model][category]['babyblue_asr'] = round(results[model][category]['babyblue'] / category_query_counts[model][category], 1)
                results[model][category]['total_queries'] = category_query_counts[model][category]

    return results

def calculate_average_asr_by_semantic_category(overall_results):
    average_results = {}

    for model, categories_scores in overall_results.items():
        for category, scores in categories_scores.items():
            if category not in average_results:
                average_results[category] = {'harmbech_asr': 0, 'advbench_asr': 0, 'babyblue_asr': 0, 'total_queries': 0}
            average_results[category]['harmbech_asr'] += scores['harmbech_asr']
            average_results[category]['advbench_asr'] += scores['advbench_asr']
            average_results[category]['babyblue_asr'] += scores['babyblue_asr']
            average_results[category]['total_queries'] += scores['total_queries']

    model_count = len(overall_results)
    for category in average_results:
        average_results[category]['harmbech_asr'] = round(average_results[category]['harmbech_asr'] / model_count, 1)
        average_results[category]['advbench_asr'] = round(average_results[category]['advbench_asr'] / model_count, 1)
        average_results[category]['babyblue_asr'] = round(average_results[category]['babyblue_asr'] / model_count, 1)

    return average_results

# Load behavior categories
behavior_categories = load_text_behaviors(text_behaviors_file)

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

        results = calculate_asr_by_semantic_category(partial_data, behavior_categories, ['ALL'])
        for model in results.keys():
            if model not in overall_results:
                overall_results[model] = {}
            for category, scores in results[model].items():
                if category not in overall_results[model]:
                    overall_results[model][category] = {'harmbech_asr': 0, 'advbench_asr': 0, 'babyblue_asr': 0, 'total_queries': 0}
                overall_results[model][category]['harmbech_asr'] += scores['harmbech_asr']
                overall_results[model][category]['advbench_asr'] += scores['advbench_asr']
                overall_results[model][category]['babyblue_asr'] += scores['babyblue_asr']
                overall_results[model][category]['total_queries'] += scores['total_queries']

print(f'query count: {query_count}, case count: {case_count}')
print(f'unique query count: {len(unique_query)}')

# Calculate and print average ASR by SemanticCategory
average_results = calculate_average_asr_by_semantic_category(overall_results)

print("Average ASR by SemanticCategory:")
for category, scores in average_results.items():
    print(f"SemanticCategory: {category}")
    print(f"  harmbech_asr: {scores['harmbech_asr']}")
    print(f"  advbench_asr: {scores['advbench_asr']}")
    print(f"  babyblue_asr: {scores['babyblue_asr']}")
    print(f"  total_queries: {scores['total_queries']}")
