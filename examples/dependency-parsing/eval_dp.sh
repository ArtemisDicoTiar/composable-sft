#!/bin/bash
LANG=bxr # Buryat
TREEBANK=bxr_bdt
LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small
TASK_FT=cambridgeltl/mbert-task-sft-dp

python run_dp.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name $TREEBANK \
  --output_dir results/${LANG} \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --eval_split test
