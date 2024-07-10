# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

accelerate launch --config_file c2.yaml  train_gen.py \
--local_dir "/LLM-QAT/" \
--input_model_filename "Qwen/Qwen1.5-7B" \
--output_model_filename "Qwen1.5-7B-finetuned_2000_samples" \
--train_data_local_path "/LLM-QAT/gen_data/all_gen.jsonl" \
--do_train True \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 50 \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--qat True \
--w_bits $1 \
--a_bits $2 \
--kv_bits $3 \
--use_kd False \
--w_groupsize 64 \
--quant_strategy_path "quant_strategy.csv"
# --fsdp "full_shard auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
