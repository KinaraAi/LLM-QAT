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

model_path = sys.argv[1]
i_start = int(sys.argv[2])
j_start = int(sys.argv[3]) 
batch = int(sys.argv[4])
sample_start = int(sys.argv[5])
sample_end = int(sys.argv[6])


print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
print("Tokenizer loaded!")
print("Loading model: ", model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=f"auto",
    torch_dtype="auto",
    trust_remote_code=True)
# model = model.cuda()
print("Model loaded!")
n_vocab = sample_end - sample_start  # number of initial tokens for synthesizing data on each GPU.

if os.path.exists(f"gen_data_{j_start}/gen.chunk." + str(i_start).zfill(2) + ".jsonl"):
    with open(f"gen_data_{j_start}/gen.chunk." + str(i_start).zfill(2) + ".jsonl",
              "r") as f:
        lines = f.readlines()
        if len(lines) >= n_vocab:
            inner_loop = sample_start + n_vocab
        else:
            inner_loop = len(lines) % n_vocab
            outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists(f"gen_data_{j_start}"):
    os.mkdir(f"gen_data_{j_start}")

for j in [j_start]:
    for i in range( sample_start + inner_loop,sample_end, batch ):
        lids = [[k] for k in range(i, i + batch)]
        input_ids = torch.tensor(lids).to(model.device)
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
