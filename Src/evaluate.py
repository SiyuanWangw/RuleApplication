import json
import argparse
import os
from evaluation_utils import *
import random


def evaluate_performance(examples, dataset_name, predictions, backup_predications=None):    
    hit_acc_num = 0
    no_hit_acc_num = 0
    total_num = 0
    hit_num = 0

    for each_instance in examples:
        total_num += 1

        if dataset_name == "clutrr":
            key = each_instance['query']
            qeury_objects = each_instance["query"].split(". ")[-1].split("How is")[1][:-1].strip().split(" related to ")
            for each_pred in predictions[key]:
                hit = True
                for each_obj in qeury_objects:
                    if each_obj not in each_pred[1]:
                        hit = False
                        break
                if hit:
                    hit_num += 1
                    pred_answer = each_pred[0].split(". ")[1].strip()
                    break
            if not hit:
                # continue 
                response = backup_predications[key]        
                assert "Answer:" in response
                pred_answer = response[response.find("Answer:")+7:].strip().split("#END#")[0]   
            
            if parse_answer(each_instance['answer'], pred_answer):
                if hit:
                    hit_acc_num += 1
                else:
                    no_hit_acc_num += 1
        
        elif dataset_name == "proofwriter":
            key = " ".join(each_instance['facts']) + " Is it true that " + each_instance["question"][:-1] + "?"
            pred = predictions[key]

            if each_instance['answer'] != "Unknown":
                if "Yes" in pred and "No" not in pred:
                    hit = True
                else:
                    hit = False
            else:
                if "No" in pred and "Yes" not in pred:
                    hit = True
                else:
                    hit = False
            
            if hit:
                hit_num += 1
            
            if "Answer:" in pred:
                pred_answer = pred[pred.find("Answer:")+7:].strip().split("#END#")[0]
                if str(each_instance['answer']).lower() in pred_answer.lower():
                    if hit:
                        hit_acc_num += 1
                    else:
                        no_hit_acc_num += 1
                    
        elif dataset_name == "boxes":
            key = each_instance["context"]
            query_object = each_instance["question"].split("contained in ")[1][:-1].strip()

            if query_object not in predictions[key]:
                query_object = query_object.replace("Box ", "Box_").strip()
            pred_answers = [_.replace("the ", "").strip() for _ in predictions[key][query_object][1][2:]]
            ref_answers = [_.replace("the ", "").strip() for _ in each_instance["answers"]]

            hit_num += 1
            if set(pred_answers) == set(ref_answers):
                hit_acc_num += 1
            # else:
            #     print(pred_answers, ref_answers)
            #     print(each_instance['context'])

        elif dataset_name == "lsat-ar":
            key = each_instance["question"] + " " + each_instance['context']
            negative_question = "EXCEPT" in each_instance['question'] or "CANNOT" in each_instance['question']
            conflict_options = []
            for each_op in predictions[key][0]:
                conflict_options.append(each_op.split(") ")[0])
            
            hit = False
            if not negative_question:
                if len(conflict_options) == 4:
                    hit = True
                    pred_ans = set(["A", "B", "C", "D", "E"]) - set(conflict_options)
                else:
                    backup_pred = backup_predications[key][backup_predications[key].find("Answer:")+7:].strip().split("#END#")[0]
                    if backup_pred not in conflict_options:
                        pred_ans = backup_pred
                    else:
                        pred_ans = random.choice(set(["A", "B", "C", "D", "E"]) - set(conflict_options))
            else:
                if len(conflict_options) == 1:
                    hit = True
                    pred_ans = conflict_options[0]
                else:
                    pred_ans = backup_predications[key][backup_predications[key].find("Answer:")+7:].strip().split("#END#")[0]
            
            if str(each_instance['answer']) in pred_ans:
                if hit:
                    hit_num += 1
                    hit_acc_num += 1
                else:
                    no_hit_acc_num += 1 
            
    print(hit_num, total_num)
    return total_num, (hit_acc_num+no_hit_acc_num)/total_num*100, hit_num, hit_acc_num/hit_num*100 if hit_num>0 else 0

def load_raw_dataset(data_path, dataset_name, split):
    print(os.getcwd())
    with open(os.path.join(data_path, dataset_name, f'{split}.json')) as f:
        raw_dataset = json.load(f)
    print(f"Loaded {len(raw_dataset)} examples from {dataset_name} {split} split.\n")
    return raw_dataset

def evaluate(args):
    examples = load_raw_dataset(args.data_path, args.dataset_name, args.split)

    if args.dataset_name == "clutrr":
        result_file = os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/facts/{args.split}_{args.model_name}_implement.json')
        print("Result file: ", result_file)
        with open(result_file, 'r') as f:
            predictions = json.load(f)
            print(f"Loaded {len(predictions)} predictions from {result_file}.\n")

        backup_file = os.path.join(args.save_path, f'{args.dataset_name}/baselines/CoT/{args.split}_{args.model_name}.json')
        print("Backup file: ", backup_file)
        with open(backup_file, 'r') as f:
            backup_predictions = json.load(f)
            print(f"Loaded {len(backup_predictions)} predictions from {backup_file}.\n")

        total_records, accuracy, hit_num, hit_acc = evaluate_performance(examples, args.dataset_name, predictions, backup_predictions)

    elif args.dataset_name == "proofwriter":
        result_file = os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/final/{args.split}_{args.model_name}.json')
        print("Result file: ", result_file)
        with open(result_file, 'r') as f:
            predictions = json.load(f)
            print(f"Loaded {len(predictions)} predictions from {result_file}.\n")
        
        total_records, accuracy, hit_num, hit_acc = evaluate_performance(examples, args.dataset_name, predictions)

    elif args.dataset_name == "lsat-ar":
        result_file = os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/implementation/{args.split}_{args.model_name}_response.json')
        print("Result file: ", result_file)
        with open(result_file, 'r') as f:
            predictions = json.load(f)
            print(f"Loaded {len(predictions)} predictions from {result_file}.\n")

        backup_file = os.path.join(args.save_path, f'{args.dataset_name}/baselines/CoT/{args.split}_{args.model_name}.json')
        print("Backup file: ", backup_file)
        with open(backup_file, 'r') as f:
            backup_predictions = json.load(f)
            print(f"Loaded {len(backup_predictions)} predictions from {backup_file}.\n")

        total_records, accuracy, hit_num, hit_acc = evaluate_performance(examples, args.dataset_name, predictions, backup_predictions)
    
    elif args.dataset_name == "boxes":
        result_file = os.path.join(args.save_path, f'{args.dataset_name}/symbolic_memory/implementation/{args.split}_{args.model_name}_response.json')
        print("Result file: ", result_file)
        with open(result_file, 'r') as f:
            predictions = json.load(f)
            print(f"Loaded {len(predictions)} predictions from {result_file}.\n")
        
        total_records, accuracy, hit_num, hit_acc = evaluate_performance(examples, args.dataset_name, predictions)

    print(f"Total records: {total_records}")
    print(f"Total Accuracy: {accuracy:.2f}%")
    print(f"Hit records: {hit_num}")
    print(f"Hit accuracy: {hit_acc:.2f}%")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='../Output/')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--result_path', type=str, default='./results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
