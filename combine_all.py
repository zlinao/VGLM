
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import os

import torch
import torch.nn.functional as F
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, VLM
from train import SPECIAL_TOKENS, build_input_from_segments_dialqa, build_input_from_segments_nlg, build_input_from_segments_mtsm, add_special_tokens_, MAXLEN_MAP
from utils import get_dataset_personalities, download_pretrained_model, get_dataset
import numpy as np
from util.eval_metrics import moses_multi_bleu, rouge #, ent_score
import json
from copy import deepcopy

BOTTLEBECK_MAP = {"mt":300, "summarization":100, "dialogue":100, "qa": 300, "nlg": 10}
COMMAND2ID = {"mt":0, "summarization":1, "dialogue":2, "qa": 3, "nlg":4}
EOSLIST = {"mt":50269, "summarization":50273, "dialogue":50259, "qa":50264, "nlg":50277}
ID2COMMAND = ["mt", "summarization", "dialogue", "qa", "nlg"]
#COMMAND2ID = {"translate":0, "summarize":1, "chat":2, "qa": 3, "nlg":4}
ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>', 'additional_special_tokens': ("<bos_dial>", "<eos_dial>", "<speaker1>", "<speaker2>", "<persona>",
                  "<bos_qa>", "<eos_qa>", "<question>", "<answer>", "<document>", 
                  "<bos_mt>", "<eos_mt>", "<source_mt>", "<target_mt>",
                  "<bos_sm>", "<eos_sm>", "<source_sm>", "<target_sm>",
                  "<bos_nlg>", "<eos_nlg>", "<name>", "<eatType>", "<familyFriendly>", "<priceRange>", "<food>", "<near>", "<area>", "<customerRating>", "<response>",
                  "<pad>",)}


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--task", type=str, default="dialogue", help="one of task from [dialogue, qa, mt, nlg, summarization]")
    parser.add_argument("--self_copy", action='store_true', help="add self copy")
    parser.add_argument("--perturbation_layers", type=int, default=0, help="number of perturbation layers")
    parser.add_argument("--adapter_bottleneck", type=int, default=0, help="adapter layer bottleneck")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    
  
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #path_list = ["Oct31_20-10-36_black-rack-1_gpt2_mt_0.0005_0_False_epoch8adapter300", "Oct28_13-02-05_black-rack-1_gpt2_summarization_0.0005_0_False_epoch10adapter100", "Nov18_11-06-15_black-rack-1_gpt2_dialogue_0.001_0_False_epoch3adapter100randomFalse_distillation", "Nov03_21-09-43_black-rack-1_gpt2_qa_0.0005_0_False_epoch5adapter300", "Oct28_02-34-33_black-rack-1_gpt2_nlg_0.005_0_False_epoch10adapter10"]

    # path_list: Give the list of checkpoints of [mt, summarization, dialogue, qa, nlg] to combine all adapters into VLM
    path_list = ["Nov15_10-44-19_black-rack-1_gpt2_mt_0.0005_0_False_epoch8adapter300randomFalse_distillation", "Nov24_21-54-28_black-rack-1_gpt2_summarization_0.0005_0_False_epoch5adapter100randomFalse_distillation", "Nov18_11-06-15_black-rack-1_gpt2_dialogue_0.001_0_False_epoch3adapter100randomFalse_distillation", "Nov03_21-09-43_black-rack-1_gpt2_qa_0.0005_0_False_epoch5adapter300", "Oct28_02-34-33_black-rack-1_gpt2_nlg_0.005_0_False_epoch10adapter10"]

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = VLM
    vlm_model = model_class.from_pretrained(args.model_checkpoint, bottleneck_map=BOTTLEBECK_MAP)
    add_special_tokens_(vlm_model, tokenizer)
    print([tokenizer.encode(word) for word in ATTR_TO_SPECIAL_TOKEN['additional_special_tokens']])
    vlm_weights = deepcopy(vlm_model.state_dict())
    model_class_sub = GPT2LMHeadModel
    bottlenecklist = [300, 100, 100, 300, 10]
    emblist = [[50268, 50272], [50272, 50276], [50258, 50263], [50263, 50268], [50276, 50287]]

    for i, path in enumerate(path_list):
        sub_model = model_class_sub.from_pretrained("runs/"+path, perturbation_layers=0, self_copy=False, adapter_bottleneck=bottlenecklist[i])
        #print(vlm_model.transformer.wte.weight.data[emblist[i][0]:emblist[i][1],:].shape, sub_model.transformer.wte.weight.data[emblist[i][0]:emblist[i][1],:].shape)
        vlm_model.transformer.wte.weight.data[emblist[i][0]:emblist[i][1],:] = sub_model.transformer.wte.weight.data[emblist[i][0]:emblist[i][1],:]
        #vlm_model.transformer.wte.weight.data = sub_model.transformer.wte.weight.data
        #assert torch.equal(vlm_model.transformer.wte.weight.data[:50257,:], sub_model.transformer.wte.weight.data[:50257,:])
        # print(ID2COMMAND[i])
        #print(vlm_model.transformer.h[10].adapter_block.mixadapter[4].project_up.weight.data)
        #print(sub_model.transformer.h[10].mlp.c_proj.weight.data)
        # print([n for n, p in sub_model.named_parameters()])
        weights = deepcopy(sub_model.state_dict())
        check_str = "mixadapter.{}.".format(i)
        model_dict = vlm_model.state_dict()
        # for name in vlm_weights:
        #     if ("mixadapter" not in name) and ("wte" not in name):
        #         try:
        #             assert torch.equal(vlm_weights[name], weights[name])
        #         except:
        #             print(name)
        #             print(vlm_weights[name])
        #             print(weights[name])
 
        #print({ name: name.replace(check_str, "") for name in vlm_weights if check_str in name })
        model_dict.update({ name: weights[name.replace(check_str, "")] for name in vlm_weights if check_str in name })
        vlm_model.load_state_dict(model_dict)
        #print(vlm_model.transformer.h[10].adapter_block.mixadapter[4].project_up.weight.data)
        # print([n for n, p in sub_model.named_parameters() if "adapter" in n])
        # exit(0)
        # model.to(args.device)
        # add_special_tokens_(model, tokenizer)
    torch.save(vlm_model.state_dict(), "runs/VLM/pytorch_model.bin")
    


if __name__ == "__main__":
    run()
