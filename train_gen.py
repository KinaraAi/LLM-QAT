# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Add quantization and knowledge distillialtion
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
import copy
import torch
import transformers
from utils import utils
from utils import datautils

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch.nn as nn
import random

log = utils.get_logger("clm")


'''Function to evaluate the model: Calculates the perplexity'''
def evaluate_perplexity(model, testenc, nsamples, seqlen, dataset_name):
    model.config.use_cache = False
    model.eval()
    nlls = []
    nsamples = testenc.numel() // seqlen
    for i in tqdm(range(nsamples)):
        with torch.no_grad():
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
            if(batch.shape[-1] == 0):
                nsamples = nsamples - 1
                continue
            logits = model(batch)['logits'].to('cpu')
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            del batch, logits, shift_logits, shift_labels
            torch.cuda.empty_cache()
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print("nsamples: ", nsamples, " ", dataset_name, "_perplexity: ", ppl.item())

'''Function to initialize the wikitext dataset'''
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

'''Function which creates batches of inputs into a Dataset.dict object'''
def get_data_dict(input_ids, nsamples, seqlen):
    batch = []
    nsamples = input_ids.numel() // seqlen
    for i in range(nsamples):
        sample = (input_ids[:, (i * seqlen) : ((i + 1) * seqlen)]).tolist()[0]
        batch.append(sample)

    data = Dataset.from_dict({
        "input_ids": batch,
        "labels": batch
    })
    return data

def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()
    import json
    import pandas as pd

    quant_strategy = {}
    if model_args.quant_strategy_path is not None:
        df = pd.read_csv(model_args.quant_strategy_path, index_col="layer")
        quant_strategy = json.loads(df.to_json(orient="index"))
    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    if training_args.qat:
        config = LlamaConfig.from_pretrained(model_args.input_model_filename)
        student_config = copy.deepcopy(config)
        student_config.w_bits = model_args.w_bits
        student_config.a_bits = model_args.a_bits
        student_config.kv_bits = model_args.kv_bits
        student_config.quant_strategy = quant_strategy
        student_config.group_size = model_args.w_groupsize
        model = LlamaForCausalLMQuant.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model_filename,
            config=student_config,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None #if len(training_args.fsdp) > 0 else "auto",
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model_filename,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None #if len(training_args.fsdp) > 0 else "auto",
        )
    # model.cuda()
    if training_args.use_kd:
        from utils.kd_trainer import KDTrainer
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model_filename,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
        teacher_model.eval()
        teacher_model.cuda()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        model.kd_loss_scale = training_args.kd_loss_scale
        model.teacher = teacher_model
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        pad_token='<|endoftext|>',
    )
    log.info("Complete tokenizer loading...")

    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=data_args.train_data_local_path,
        valid_path=data_args.eval_data_local_path
        if data_args.eval_data_local_path is not None
        else None,
    )
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    valid_data = datautils.CustomJsonDataset(
        valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )
    
    ###################### Testing on wiki Dataset  ###########################
    
    valdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(valdata['text']), return_tensors='pt')
    final_valid_data = get_data_dict(testenc.input_ids, nsamples=128, seqlen=2048)
    
    ############################################################################
    
    model.config.use_cache = False
    if training_args.use_kd:
        from utils.kd_trainer import KDTrainer
        myTrainer = KDTrainer
    else:
        myTrainer = Trainer
        
    def preprocess_logits_for_metrics(logits, labels):
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        ppl_list = []

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        neg_log_likelihood = loss.float()
        return neg_log_likelihood, labels
        

    def compute_metrics(eval_preds):
        neg_log_likelihood, labels = eval_preds
        ppl = torch.exp(torch.mean(torch.tensor(neg_log_likelihood[0])))
        return {"perplexity" : ppl.item()}

    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    print("Path: ",model_args.output_model_local_path)
    if training_args.do_train:
        train_result = trainer.train()
        # trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        # model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    if training_args.do_eval:
        # model.to("cuda")
        metrics = trainer.evaluate(eval_dataset=final_valid_data)
        max_eval_samples = len(final_valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(final_valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval1", metrics)
        trainer.save_metrics("eval1", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
