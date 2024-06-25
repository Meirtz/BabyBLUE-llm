# Make sure your environment has enough GPUs for the classifier (Llama 2 13B)
base_save_dir="results"

method_name="GCG"
# Often, the experiment name will be the same as the model name. It can be different for transfer attacks.
experiment_name="llama2_7b" 
model_name="llama2_7b" # from configs/model_configs/models.yaml

cls_path="cais/HarmBench-Llama-2-13b-cls"
behaviors_path="data/behavior_datasets/extra_behavior_datasets/babyblue_behaviors.csv"
completions_path="$base_save_dir/$method/$experiment/completions/$model.json" # output file from Step 2
save_path="$base_save_dir/$method/$experiment/results/$model.json"

./scripts/evaluate_completions.sh $cls_path $behaviors_path $completions_path $save_path
