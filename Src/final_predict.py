import json
import argparse
import os
import sys
import random
from evaluation_utils import *
from models.openai_models import OpenAIModel

def load_prompt_final_predict(dataset_name):
    file_path = os.path.join('./prompts', dataset_name, 'final_predict.txt')
    with open(file_path) as f:
        final_predict_prompt = f.read()

    if dataset_name == "proofwriter":
        system_input = ""
    return system_input, final_predict_prompt

def load_raw_dataset(data_path, dataset_name, split):
    print(os.getcwd())
    with open(os.path.join(data_path, dataset_name, f'{split}.json')) as f:
        raw_dataset = json.load(f)
    print(f"Loaded {len(raw_dataset)} examples from {dataset_name} {split} split.\n")
    return raw_dataset

def final_predict(args):
    openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
    examples = load_raw_dataset(args.data_path , args.dataset_name, args.split)

    result_file = os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/facts/{args.split}_{args.model_name}_implement.json')
    print("Result file: ", result_file)
    with open(result_file, 'r') as f:
        fact_memory = json.load(f)
        print(f"Loaded {len(fact_memory)} predictions from {result_file}.\n")

    system_input, final_predict_prompt = load_prompt_final_predict(args.dataset_name)

    if os.path.exists(os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/final/{args.split}_{args.model_name}.json')):
        with open(os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/final/{args.split}_{args.model_name}.json'), "r") as f:
            output = json.load(f)
            print(len(output))
    else:
        output = {}

    for example in examples:
        if args.dataset_name == "proofwriter":
            key = " ".join(example['facts']) + " Is it true that " + example["question"][:-1] + "?"
            question = "Is it true that " + example["question"][:-1] + "?"

            rules = " ".join(example['rules'])
            facts_verb = [each[0] if type(each) == list else each for each in fact_memory[key]]
        
        prompt = final_predict_prompt.format(rules=rules, fact_list=" ".join(facts_verb), query=question)
        response = openai_api.model_generate(system_input, prompt)
        output[key] = response
    
    with open(os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/final/{args.split}_{args.model_name}.json'), "w") as f:
        print(len(output))
        json.dump(output, f, indent=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='../Output/')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    final_predict(args)
