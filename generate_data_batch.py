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

i_start = int(sys.argv[1])
j_start = int(sys.argv[2]) 
batch = int(sys.argv[3])
device = torch.device("cuda:1") if i_start % 2 else torch.device("cuda:0")
# device_map = "cuda:0" if i_start%2 else "cuda:1"
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B",trust_remote_code=True)
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B",
    device_map=f"cuda:{i_start%2}",
    torch_dtype="auto")
# model = model.cuda()
print("Model loaded!")

n_vocab = 10000  # number of initial tokens for synthesizing data on each GPU.

if os.path.exists(f"gen_data_{j_start}/gen.chunk." + str(i_start).zfill(2) + ".jsonl"):
    with open(f"gen_data_{j_start}/gen.chunk." + str(i_start).zfill(2) + ".jsonl",
              "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists(f"gen_data_{j_start}"):
    os.mkdir(f"gen_data_{j_start}")

for j in [j_start]:
    for i in range(
            int(i_start) * n_vocab + inner_loop, (int(i_start) + 1) * n_vocab,
            batch):
        lids = [[k] for k in range(i, i + batch)]
        input_ids = torch.tensor(lids).cuda(device)
        # breakpoint()
        print("generating", i)
        outputs1 = model.generate(input_ids, do_sample=False, max_new_tokens=j)
        outputs = model.generate(outputs1,
                                 do_sample=True,
                                 max_new_tokens=2048 - j)
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        with open(f"gen_data_{j_start}/gen.chunk." + str(i_start).zfill(2) + ".jsonl",
                  "a") as f:
            for b in range(batch):
                text_dict = {"text": gen_text[b]}
                f.write(json.dumps(text_dict))
                f.write('\n')
