# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os

print("Loading tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("/auto/worka/aakash/llm/llama/llama_output_dir")
print("Tokenizer loaded!")
print("Loading model")
# model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
# model = model.cuda()
print("Model loaded!")

n_vocab = 500 # number of initial tokens for synthesizing data on each GPU.

i_start = sys.argv[1]
if False and os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")

for j in range(3 + outer_loop, 6):
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(i)
        input_ids = torch.tensor([[i]]).cuda()
        print("generating")
        outputs1 = input_ids
        outputs = input_ids
        # gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text" :i}
        with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')
