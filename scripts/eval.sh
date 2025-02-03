#!/bin/sh

# Usage: ./eval.sh <task_config> [agent_config] [additional_hydra_args]
#
# Examples:
# Basic usage:
#   ./eval.sh task=hello_world agent=openhands
#
# With overrides:
#   ./eval.sh task=hello_world agent=openhands num_runs=3
#
# Multirun (parameter sweep):
#   ./eval.sh --multirun task=hello_world,shape_mismatch_train agent=openhands,react
#
# Run all permutations:
#   # Run all debugging tasks with all agents
#   ./eval.sh --multirun task=shape_mismatch_train,output_shape,nan_loss_debug,normalization_bug,training_files_debug agent=openhands,react
#
#   # Run all training tasks with all agents
#   ./eval.sh --multirun task=basic_vision_finetuning,multi_gpu_training,parse_logs,pretrained_bert_base_uncased_load,pretrained_model_load_from_torchvision,small_dataset_overfit agent=openhands,react
#
#   # Run specific tasks with multiple configurations
#   ./eval.sh --multirun task=hello_world,shape_mismatch_train agent=openhands,react num_runs=1,3 debug=true,false
#
# Run all available configurations:
#   # Run all tasks with all agents
#   ./eval.sh --multirun "task=glob(*)" "agent=glob(*)"
#
#   # Run all tasks with specific agent
#   ./eval.sh --multirun "task=glob(*)" agent=openhands
#
#   # Run specific task with all agents
#   ./eval.sh --multirun task=hello_world "agent=glob(*)"
#
# Available tasks:
# - Debugging:
#   - shape_mismatch_train, output_shape     : Shape mismatch debugging
#   - nan_loss_debug                         : NaN losses debugging
#   - normalization_bug                      : Normalization issues
#   - training_files_debug                   : Multi-file fixes
#
# - Dataset:
#   - dataset_download                       : Basic dataset download
#   - noisy_dataset_download                 : Noisy dataset handling
#
# - Training:
#   - basic_vision_finetuning                : Vision model finetuning
#   - multi_gpu_training                     : Multi-GPU setup
#   - parse_logs                             : Training log analysis
#   - pretrained_bert_base_uncased_load      : BERT model loading
#   - pretrained_model_load_from_torchvision : Torchvision model loading
#   - small_dataset_overfit                  : Small dataset training
#
# - Model Modification:
#   - channel_vit_implementation             : Channel ViT implementation
#   - channel_vit_implementation_easy        : Simplified Channel ViT
#   - channel_vit_implementation_no_test     : Channel ViT without tests
#   - var_implementation                     : VAR model implementation
#
# - Model Improvement:
#   - improve_cifar10_baseline              : CIFAR-10 improvement
#   - improve_segmentation_baseline         : Segmentation improvement
#
# - Full Workflow:
#   - cifar_10_lt_performance              : CIFAR-10 long-tail performance
#   - performance_test                     : Performance testing
#   - setup_test                          : Setup verification
#   - setup_test_ambiguous                : Ambiguous setup handling
#
# - Other:
#   - hello_world                         : Basic hello world test
#   - wandb_logging                       : W&B logging setup
#
# Note: When using --multirun, Hydra will run all combinations of the provided values.
# For example, with 2 tasks and 2 agents, it will run 4 combinations (2x2).
# The runs are executed sequentially by default.
#
# Available agents:
#   - openhands : Claude-based agent
#   - react     : React-based agent

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Navigate to the project root (ml_dev_bench)
cd "${SCRIPT_DIR}/.." || exit 1

# Run the evaluation using Hydra through Poetry
python -m calipers.scripts.run_hydra_evaluation $@
