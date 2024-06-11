#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

python SFT_trainer/train.py \
   --use_lora True \
   --use_qlora False \
   --gradient_checkpointing False \
   --data_name_or_path maywell/ko_wikidata_QA \
   --model_name_or_path EleutherAI/polyglot-ko-1.3b \
   --output_dir ./trained_sft_models \
   --save_strategy epoch \
   --save_steps 500 \
   --num_train_epochs 1 \
   --logging_steps 1 \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 1 \
   --warmup_steps 0 \
   --learning_rate 0.00001 \
   --weight_decay 0.0 \
   --mixed_precision bf16 