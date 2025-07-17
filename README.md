# ARC-AGI-DL-2025
Project:  Basics to Deep Learning course(2025-1) at SNU

# Llama-3.2-1B-Instruct Fine-Tuning 
Team 25

This repository contains the code for fine-tuning the `meta-llama/Llama-3.2-1B-Instruct` model on a subset of the ARC (Abstraction and Reasoning Challenge) dataset. The training process leverages the Unsloth library for efficient, memory-optimized 4-bit training using QLoRA.

This README provides instructions on setting up the environment, running the training script, and details the hyperparameters used to ensure reproducibility.

**Parts of Code adapted from:** [https://github.com/da-fr/arc-prize-2024/tree/main](https://github.com/da-fr/arc-prize-2024/tree/main) by Daniel Franzen and Jan Disselhoff.

## Project Overview

The goal of this project is to fine-tune a powerful base model, Llama-3.2-1B, to improve its performance on abstract reasoning tasks as defined by the ARC dataset.

- **Model:** `meta-llama/Llama-3.2-1B-Instruct`
- **Technique:** 4-bit QLoRA with Rank-Stabilized LoRA (RSLora)
- **Frameworks:** Unsloth, PEFT, Transformers, PyTorch
- **Key Features:**
    - Efficient 4-bit training to reduce memory footprint.
    - Custom data collator (`InputMaskingDataCollator`) to mask input prompts during loss calculation, focusing training on the model's answers.
    - On-the-fly data augmentation (permutations, rotations, etc.).
    - A two-stage script that first trains and saves the LoRA adapter, then merges it into the base model.

## Prerequisites

### 1. Hardware
- **GPU:** An NVIDIA GPU with CUDA environment.
- **OS:** Linux-based OS (tested on Ubuntu).

### 2. Software & Dependencies
First, clone the repository and install the required Python packages. It is highly recommended to use a virtual environment (e.g., `conda` or `venv`).

```bash
conda run -n $EVAL_ENV pip install diskcache trl unsloth
```

### 3. Dataset
This script uses a custom `ArcDataset` loader that expects the ARC dataset in a specific JSON format.
- Create a directory for your dataset. In the script, this is hardcoded as `/home/student/workspace/dataset`.
- Place your ARC dataset JSON files within this directory. The `ArcDataset.load_from_local_json` function will handle loading them.

## How to Run the Training

The training script is designed to be executed in one go. It performs two main actions: `train` and `merge`.

### Step 1: Run Initial Fine-Tuning Script
```bash
python base_finetune.py
```

This script performs LoRA fine-tuning on the pretrained Llama model and then merges the adapter.

### Step 2: Run Second Fine-Tuning Script
```bash
python base_finetune_final.py
```

This script loads the merged model from the previous step and performs a second round of LoRA fine-tuning (with masking applied), followed by merging the new adapter.

### Step 3: GRPO Final Training Execution

To run the final GRPO training script:

```bash
python train_grpo.py
```

### Script Logic
1. **Train:** The script first checks if a LoRA adapter already exists at `{output_model_base_path}-lora`. If not, it will:
   - Load the 4-bit quantized base model (`Llama-3.2-1B-Instruct`).
   - Prepare the model for LoRA fine-tuning using the specified hyperparameters.
   - Load and augment the ARC dataset.
   - Initialize the Unsloth `Trainer`.
   - Run the training for 1 epoch.
   - Save the resulting LoRA adapter and tokenizer to `{output_model_base_path}-lora`.

2. **Merge:** After the training phase (or if the LoRA adapter was already present), the script checks if a merged model exists at `{output_model_base_path}-merged`. If not, it will:
   - Load the 4-bit quantized base model again.
   - Load the saved LoRA adapter from the `-lora` directory.
   - Merge the adapter weights into the model.
   - Save the final, merged model in 16-bit precision (`float16` or `bfloat16`) to `{output_model_base_path}-merged`.

## Final Output

Upon successful completion (all three scripts), the script will produce two main artifacts in your specified output directory:

1. **`Llama-3.2-1B-Final-RL-lora/`**: Contains the trained PEFT LoRA adapter. This can be used to dynamically load the adapter on top of the base model without merging.
2. **`Llama-3.2-1B-Final-RL-merged/`**: Contains the full model with the LoRA weights merged, saved in 16-bit precision. This is a standalone model ready for inference.
