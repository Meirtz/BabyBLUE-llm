# BabyBLUE Benchmark


![License](https://img.shields.io/badge/License-MIT-red)
![Language](https://img.shields.io/badge/🐍%20Python%20-8A2BE2)
![Pipeline](figs/pipeline.jpg)


## Project Description

The BabyBLUE (**B**enchmark for Reli**ab**ilit**y** and Jail**B**reak hal**LU**cination **E**valuation) is a novel benchmark designed to assess the susceptibility of large language models (LLMs) to hallucinations and jailbreak attempts. Unlike traditional benchmarks that may misinterpret hallucinated outputs as genuine security threats, BabyBLUE focuses on distinguishing between actual harmful outputs and benign hallucinations.

### Key Features:

- **Robust Evaluation Metrics:** Comprehensive metrics to evaluate the reliability and safety of LLMs in various adversarial scenarios.
- **Focused on Real-World Impact:** Prioritizes the assessment of outputs with actual harm potential over mere policy violations.
- **Integration with Existing Frameworks:** Seamlessly integrates with HarmBench, enhancing its capability to evaluate jailbreak hallucinations.
- **Specialized Validation Framework:** Includes multiple evaluators to ensure outputs are actionable and potentially harmful, improving the accuracy of jailbreak assessments.

## Evaluate Completions

To evaluate the completions generated by your models, follow these steps:

1. **Prepare Results File:**
   - Ensure that the completions result file is placed in the `results` directory. The file should contain the output from the models you wish to evaluate.

2. **Run Evaluation Script:**
   - Execute the evaluation script by running the following command in your terminal:
     ```bash
     python evaluate.py
     ```
   - This script will process the results file and provide an evaluation based on the BabyBLUE benchmark metrics.

## Generate Completions

To generate completions using the benchmark, follow these steps:

1. **Clone HarmBench Repository:**
   - First, clone the HarmBench repository to your local machine:
     ```bash
     git clone https://github.com/centerforaisafety/HarmBench
     ```

2. **Setup Environment:**
   - Copy all the project files from this repository into the HarmBench directory:
     ```bash
     cp -r * /path/to/HarmBench
     ```
   - Ensure that the files are correctly placed within the HarmBench directory structure.

3. **Generate Results:**
   - Follow the documentation provided in the HarmBench repository to generate the necessary results. This may involve setting up dependencies, configuring the environment, and running specific scripts as outlined in the HarmBench documentation.

4. **Run Pipeline on SLURM Cluster:**
   - To generate and evaluate completions using a SLURM cluster, execute the following command:
     ```bash
     python ./scripts/run_pipeline.py --methods ZeroShot,PEZ,TAP --models baichuan2_7b,mistral_7b,llama2_70b --step 2_and_3 --mode slurm
     ```
   - This command will run the specified methods and models, generating completions and evaluating them according to the Baby'BLUE benchmark.

## Acknowledgements

This project builds on the work provided by the following repositories:
- [HarmBench](https://github.com/centerforaisafety/HarmBench)
- [Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)

Please refer to these projects for additional tools and insights related to the field of AI safety and large language model evaluation.


