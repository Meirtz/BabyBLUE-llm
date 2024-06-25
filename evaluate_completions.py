import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import argparse
import os
import csv
from tqdm import tqdm 
import numpy as np

from vllm import LLM, SamplingParams

from eval_utils import compute_results_hashing, compute_results_advbench, compute_results_babyblue_classifier
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='/home/meilingrui/.cache/huggingface/hub/models--NousResearch--Hermes-2-Theta-Llama-3-8B/snapshots/dd2bfa013380639acf9e8fa45ceacbc45fb44081/',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    # ========== load behaviors (for tags and context strings) ========== #
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    
    # convert to dictionary mapping behavior ID field to behavior dict
    behaviors = {b['BehaviorID']: b for b in behaviors}

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side="right"

    # Convert completions to a dictionary mapping behavior ID to a dictionary of model name to completion
    for behavior_id, completion_dict in completions.items():
        # Convert the completion dictionary to a list of completions
        # completion_list = [completion_dict[key] for key in completion_dict.keys()]
        completion_list = []
        for model_name, completion in completion_dict.items():
            completion['model_name'] = model_name
            completion_list.append(completion)
        completions[behavior_id] = completion_list




    # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
    for behavior_id, completion_list in completions.items():
        print(f"Clipping completions for behavior {behavior_id}")
        # print(f"Number of completions: {completion_dict.keys()}")
        # Convert the completion dictionary to a list of completions
        # completion_list = [completion_dict[key] for key in completion_dict.keys()] # model_name : {'generation': '...', 'label': '...}
        # print(len(completion_list))
        for completion in completion_list:
            generation = completion['generation']
            tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
            clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            completion['generation'] = clipped_generation
        # break

    # ========== evaluate completions ========== #
    cls = LLM(model=args.cls_path, tensor_parallel_size=1)
    cls.llm_engine.tokenizer.tokenizer.truncation_side="left"

    cls_params = SamplingParams(temperature=0.0, max_tokens=4096)

    results = {}
    for behavior_id, data in tqdm(completions.items()):
        if behavior_id not in behaviors:
            print(f"Behavior {behavior_id} not found in behaviors file; skipping")
            continue
        behavior_dict = behaviors[behavior_id]
        tags = behavior_dict['Tags'].split(', ')

        if 'hash_check' in tags:
            current_results = compute_results_hashing(behavior_dict, data)
        else:
            current_results = compute_results_babyblue_classifier(behavior_dict, data, cls, cls_params)
        
        if args.include_advbench_metric:
            current_results = compute_results_advbench(current_results, data)

        current_results = [{r['model_name']:r} for r in current_results]
        results[behavior_id] = current_results
        # break

    success_rates = []

    for i, (behavior_id, data) in enumerate(results.items()):


        # print(data)
        total_success = 0
        for d in data:
            for mode_name in d.keys():
                # skip hash_check behaviors for harmbench dataset
                if 'babyblue_label' not in d[mode_name]:
                    d[mode_name]['babyblue_label'] = d[mode_name]['label']
                total_success += d[mode_name]['babyblue_label'] == 1
        success_mean = total_success / len(data)
        # success_mean = np.mean([d['babyblue_label'] == 1 for d in data])
        print(f"{i}. {behavior_id} === average asr: {success_mean}")
        success_rates.append(success_mean)
    
    print("Average ASR: ", np.mean(success_rates))

    # Make dirs to output_file if not yet exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()
