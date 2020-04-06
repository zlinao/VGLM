# Implemented based on https://github.com/huggingface/transfer-learning-conv-ai

import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, GPT2Config)

from utils import get_dataset, make_logdir




SPECIAL_TOKENS = {"dialogue":["<bos_dial>", "<eos_dial>", "<speaker1>", "<speaker2>", "<persona>"],
                  "qa":["<bos_qa>", "<eos_qa>", "<question>", "<answer>", "<document>"], 
                  "mt":["<bos_mt>", "<eos_mt>", "<source_mt>", "<target_mt>"],
                  "summarization":["<bos_sm>", "<eos_sm>", "<source_sm>", "<target_sm>"],
                  "nlg":["<bos_nlg>", "<eos_nlg>", "<name>", "<eatType>", "<familyFriendly>", "<priceRange>", "<food>", "<near>", "<area>", "<customerRating>", "<response>"],
                  #"smd":["<bos_smd>", "<eos_smd>", "<user>", "<system>", "<KB>"]
                  }
# ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
#                          'additional_special_tokens': ('<speaker1>', '<speaker2>', '<persona>',)}
ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>', 'additional_special_tokens': ("<bos_dial>", "<eos_dial>", "<speaker1>", "<speaker2>", "<persona>",
                  "<bos_qa>", "<eos_qa>", "<question>", "<answer>", "<document>", 
                  "<bos_mt>", "<eos_mt>", "<source_mt>", "<target_mt>",
                  "<bos_sm>", "<eos_sm>", "<source_sm>", "<target_sm>",
                  "<bos_nlg>", "<eos_nlg>", "<name>", "<eatType>", "<familyFriendly>", "<priceRange>", "<food>", "<near>", "<area>", "<customerRating>", "<response>",
                  #"<bos_smd>", "<eos_smd>", "<user>", "<system>", "<KB>",
                  "<pad>",)}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

MAXLEN_MAP = {"summarization":{'src':400, 'tgt':130}, "mt":{'src':100, 'tgt':100}, "qa":{'document':400, 'qa':50}}

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # returns 0 and doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments_dialqa(persona, history, reply, tokenizer, lm_labels=False, with_eos=True, task=None):
    """ Build a sequence of input from 3 segments for dialogue and coqa task: persona(document), history(question) and last reply(answer). """
    bos, eos, speaker1, speaker2, persona_token= tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
    instance = {}
    sequence = [[bos] + [persona_token] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [persona_token for _ in sequence[0]] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
    instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    # print("input:", tokenizer.decode(instance["input_ids"]))
    # print("type:", tokenizer.decode(instance["token_type_ids"]))
    # print("lm_label:", instance["lm_labels"])
    # print("==================================================")
    return instance, sequence  # TODO: second arg is never used, delete it


def build_input_from_segments_nlg(source, target, tokenizer, lm_labels=False, with_eos=True, task=None):
    """ Build a sequence of input from 2 segments: source and target for natrual langugage generation task. """
    bos, eos, name, eatType, familyFriendly, priceRange, food, near, area, customerRating, tgt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
    nlg_map = {"name":name, "eatType":eatType, "familyFriendly":familyFriendly, "priceRange":priceRange, "food":food, "near":near, "area":area, "customer rating":customerRating}
    instance = {}
    sequence = [[bos]] + [ [nlg_map[k]] + v for k, v in source.items()]+ [[tgt] + target + ([eos] if with_eos else [])]
    instance["input_ids"] = list(chain(*sequence))
    tokenizer.decode(instance["input_ids"])
    instance["token_type_ids"] = [bos] + list(chain(*[[s[0]]*len(s) for s in sequence[1:-1]])) + [tgt]* len(sequence[-1])
    instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    # print("input:", tokenizer.decode(instance["input_ids"]))
    # print("type:", tokenizer.decode(instance["token_type_ids"]))
    # print("lm_label:", instance["lm_labels"])
    # print("==================================================")
    return instance, sequence  # TODO: second arg is never used, delete it

def build_input_from_segments_mtsm(source, target, tokenizer, lm_labels=False, with_eos=True, task=None):
    """ Build a sequence of input from 2 segments: source and target for machine translation and summarization task. """
    bos, eos, src, tgt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
    instance = {}
    sequence = [[bos]] + [[src] + source] + [[tgt] + target + ([eos] if with_eos else [])]
    instance["input_ids"] = list(chain(*sequence))
    if len(instance["input_ids"])==0:
        print("yes")
    instance["token_type_ids"] = [bos] + [src]* len(sequence[1]) + [tgt]* len(sequence[2])
    instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

    return instance, sequence  # TODO: second arg is never used, delete it


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    loaded_dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.task)
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
    def load_dial():
        for dataset_name, dataset in loaded_dataset.items():
            tgt = "candidates"
            if (args.distillation and dataset_name=="train"):
                tgt = "distillated_candidates"
            for pair in dataset:
                persona = pair["personality"].copy()
                for utterance in pair["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    candidate = utterance[tgt][-1] #only last one candidate which is gold response
                    lm_labels = True # always train language model
                    instance, _ = build_input_from_segments_dialqa(persona, history, candidate, tokenizer, lm_labels, task=args.task)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                    persona = [persona[-1]] + persona[:-1]  # permuted personalities

    def load_smd():
        count=0
        for dataset_name, dataset in loaded_dataset.items():
            for pair in dataset:
                kb = pair["KB"].copy()
                if len(list(chain(*kb)))>800:
                    continue
                #print("Long KB:", count)
                for utterance in pair["utterances"]:
                    history = utterance["history"]

                    candidate = utterance["candidates"][-1] #only last one candidate which is gold response
                    lm_labels = True # always train language model
                    instance, _ = build_input_from_segments_dialqa(kb, history, candidate, tokenizer, lm_labels, task=args.task)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
               
    
    def load_qa():
        for dataset_name, dataset in loaded_dataset.items():
            tgt = "candidates"
            if (args.distillation and dataset_name=="train"):
                tgt = "distillated_candidates"
            for pair in dataset:
                evidence = pair["document"].copy()
                evidence = [evidence[0][:MAXLEN_MAP[args.task]['document']]]
                for utterance in pair["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    candidate = utterance[tgt][-1] #only last one candidate which is gold response
                    lm_labels = True # always train language model
                    instance, _ = build_input_from_segments_dialqa(evidence, history, candidate, tokenizer, lm_labels, task=args.task)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
    def load_nlg():
        for dataset_name, dataset in loaded_dataset.items():
            for pair in dataset:
                source = pair["src"]
                target = pair["tgt"]
                if (args.distillation and dataset_name=="train"):
                    target = pair["distillated_tgt"]
                instance, _ =  build_input_from_segments_nlg(source, target, tokenizer, lm_labels=True, with_eos=True, task=args.task)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
    def load_mtsm():
        for dataset_name, dataset in loaded_dataset.items():
            if dataset_name=="val":
                dataset_name="valid" # fix dataset key error
            tgt = "tgt"
            if (args.distillation and dataset_name=="train"):
                tgt = "distillated_tgt"
            for pair in dataset:
                #print(pair)
                if len(pair["src"])<2 or len(pair[tgt])<2: # filter the shot sentence
                    continue
                source = pair["src"][:MAXLEN_MAP[args.task]['src']]
                target = pair[tgt][:MAXLEN_MAP[args.task]['tgt']]
                instance, _ =  build_input_from_segments_mtsm(source, target, tokenizer, lm_labels=True, with_eos=True, task=args.task)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
    logger.info("Pad inputs and convert to Tensor")
    if args.task =="dialogue":
        load_dial()
        _ = datasets.pop("test", None)
        tensor_datasets = {"train": [], "valid": []}
    # elif args.task =="":
    elif args.task =="qa":
        load_qa()
        _ = datasets.pop("test", None)
        tensor_datasets = {"train": [], "valid": []}
    elif args.task =="nlg":
        load_nlg()
        tensor_datasets = {"train": [], "valid": [], "test": []}
    elif args.task =="mt":
        load_mtsm()
        datasets["valid"] = datasets["test"]
        _ = datasets.pop("test", None)
        tensor_datasets = {"train": [], "valid": []}
    else:
        load_mtsm()
        tensor_datasets = {"train": [], "valid": [], "test": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['pad_token']))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])

            tensor = tensor.view((-1, 1) + tensor.shape[1:]) #always one candidate

            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler
    
def count_parameters(list_param):
    return sum(p.numel() for p in list_param)

def count_parameters_model(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--task", type=str, default="dialogue", help="one of task from [dialogue, qa, mt, nlg, summarization]")
    parser.add_argument("--emb_only", action='store_true', help="fine tune only task embeddings")
    parser.add_argument("--linear_perturb", action='store_true', help="fine tune only task embeddings")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--perturbation_layers", type=int, default=0, help="number of perturbation layers")
    parser.add_argument("--self_copy", action='store_true', help="add self copy ")
    parser.add_argument("--adapter_bottleneck", type=int, default=0, help="adapter layer bottleneck")
    parser.add_argument("--random_init", action='store_true', help="don't use GPT-2 pre-trained model ")
    parser.add_argument("--distillation", action='store_true')
    parser.add_argument("--outputlayer_only", action='store_true')
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = GPT2LMHeadModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    if not args.random_init:
        model = model_class.from_pretrained(args.model_checkpoint, perturbation_layers=args.perturbation_layers, self_copy=args.self_copy, adapter_bottleneck=args.adapter_bottleneck)
    else:
        config = GPT2Config()
        model = model_class(config, perturbation_layers=args.perturbation_layers, self_copy=args.self_copy, adapter_bottleneck=args.adapter_bottleneck)
    model.to(args.device)

    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)

    if args.adapter_bottleneck>0:
        parameters_to_update = [p for n, p in model.named_parameters() if "adapter" in str(n)] + [model.transformer.wte.weight]
        optimizer = AdamW(parameters_to_update, lr=args.lr, correct_bias=True)
    elif args.outputlayer_only:
        parameters_to_update = [model.transformer.wte.weight]
        optimizer = AdamW(parameters_to_update, lr=args.lr, correct_bias=True)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update_emb(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, lm_labels, token_type_ids = batch
        (lm_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, labels=lm_labels
        )
        loss = lm_loss/ args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            param_to_check = []
            for n, p in model.named_parameters():
                if(n!="transformer.wte.weight"):
                    param_to_check.append(p)
            a = list(param_to_check)[0].clone()
            model.transformer.wte.weight.grad[:50257,:] = 0
            model.transformer.wte.weight.data.add_(-args.lr,model.transformer.wte.weight.grad.data)
            optimizer.zero_grad()
            param_to_check = []
            for n, p in model.named_parameters():
                if(n!="transformer.wte.weight"):
                    param_to_check.append(p)

            b = list(param_to_check)[0].clone()
            assert torch.equal(a.data, b.data)
        return loss.item()

    # Training function and trainer
    def update_linear_perturbation(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, lm_labels, token_type_ids = batch
        (lm_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, labels=lm_labels
        )
        loss = lm_loss/ args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:

            model.transformer.wte.weight.grad[:50257,:] = 0
            # model.transformer.wte.weight.data.add_(-args.lr,model.transformer.wte.weight.grad.data)
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    # Training function and trainer
    def update_all(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, lm_labels, token_type_ids = batch
        (lm_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, labels=lm_labels, self_copy=args.self_copy
        )
        loss = lm_loss/ args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    if args.emb_only:
        trainer = Engine(update_emb)
    elif (args.linear_perturb or args.adapter_bottleneck>0):
        trainer = Engine(update_linear_perturbation)
    else:
        trainer = Engine(update_all)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, lm_labels, token_type_ids = batch
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids,
            )
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, ), (lm_labels_flat_shifted, )
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint, task= args.task, lr=args.lr, layer=args.perturbation_layers, self_copy = args.self_copy, n_epochs = args.n_epochs, adapter=args.adapter_bottleneck, random_init=args.random_init)
        if args.distillation:
            log_dir+="_distillation"
        if args.outputlayer_only:
            log_dir+="_outputlayer_only"
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
