import json
import os
from tqdm import tqdm
from models.openai_models import OpenAIModel
from models.llama_models import LLamaGenerator
from utils import *
from response_postprocess import *
import argparse
from itertools import combinations, permutations


class WM_Neurosymbolic:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.application_steps = args.application_steps
        self.use_memory_schema = args.use_memory_schema
        if "gpt" in self.model_name.lower():
            self.model = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        elif "llama" in self.model_name.lower():
            self.model = LLamaGenerator(args.model_name, args.stop_words, args.max_new_tokens)
    
    def load_raw_dataset(self, split):
        print(os.getcwd())
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)[:]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name} {self.split} split.\n")
        return raw_dataset

    def update_schema(self, object_schema, predicate_schema, new_facts, is_fact=True):
        variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'X', 'Y', 'Z']
        if self.dataset_name == "lsat-ar":
            fact_variables = new_facts[1]
            if len(fact_variables) >= 2:
                for _ in fact_variables:
                    if _ not in object_schema:
                        object_schema.append(_)
            return object_schema, predicate_schema
        if is_fact:
            for each in new_facts:
                new_predicate = each[1][0]
                new_objects = each[1][1:]
                # print(new_predicate, new_objects)
                if new_predicate not in predicate_schema:
                    predicate_schema.append(new_predicate)
                for _ in new_objects:
                    if _ not in object_schema:
                        object_schema.append(_)
            return object_schema, predicate_schema
        else:
            conclusion, premises = new_facts
            conclusion_predicate, conclusion_args = conclusion
            if conclusion_predicate not in predicate_schema:
                predicate_schema.append(conclusion_predicate)
            for each in conclusion_args:
                if each not in object_schema and each not in variables:
                    object_schema.append(each)
            for each_premise in premises:
                if each_premise[0] not in predicate_schema:
                    predicate_schema.append(each_premise[0])
                for _ in each_premise[1]:
                    if _ not in object_schema and _ not in variables:
                        object_schema.append(_)
            return object_schema, predicate_schema
    
    def fact_initialization(self, examples):
        print("Start fact initialization.")
        if self.dataset_name == "boxes":
            system_input, fact_init_prompt, op_fact_init_prompt = load_prompt_fact_init(self.args)
        else:
            system_input, fact_init_prompt = load_prompt_fact_init(self.args)
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}_response.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}_response.json'), "r") as f:
                output = json.load(f)
        else:
            output = {}
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}.json'), "r") as f:
                all_example_fact = json.load(f)
        else:
            all_example_fact = {}
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json'), "r") as f:
                all_example_memory_schema = json.load(f)
        else:
            all_example_memory_schema = {}

        for example in tqdm(examples):
            if self.dataset_name == "clutrr":
                key = example["query"]
                contexts = [_+"." if _[-1] != "." else _ for _ in example["query"].split(". ")[:-1]]

                split_contexts = []
                for i in range(len(contexts)):
                    uppercase_count = sum(1 for char in contexts[i] if char.isupper())
                    if len(contexts[i].split(" ")) <= 5 or uppercase_count < 2:
                        if len(split_contexts) > 0:
                            split_contexts[-1] = split_contexts[-1] + " " + contexts[i]
                        else:
                            split_contexts.append(contexts[i])
                    else:
                        split_contexts.append(contexts[i])
            elif self.dataset_name == "proofwriter":
                key = " ".join(example['facts']) + " Is it true that " + example["question"][:-1] + "?"
                split_contexts = example['facts']
            elif self.dataset_name == "boxes":
                key = example['context']
                split_contexts = [_+"." for _ in example['context'].split(". ")[0].split(", ")]
                split_contexts_op = [_+"." if _[-1]!="." else _ for _ in example['context'].split(". ")[1:]]
            elif self.dataset_name == "lsat-ar":
                key = example["question"] + " " + example['context']
                split_contexts = example['options']

            output[key] = []
            if key in all_example_memory_schema:
                object_schema_list, predicate_schema_list = all_example_memory_schema[key]
            else:
                object_schema_list = []
                predicate_schema_list = []
            cur_exmaples_facts = []
            cur_exmaples_facts_verb = []
            for each_context in split_contexts:
                if len(object_schema_list) == 0:
                    object_schema = "null"
                else:
                    object_schema = ", ".join(object_schema_list)
                if len(predicate_schema_list) == 0:
                    predicate_schema = "null"   
                else:
                    predicate_schema = ", ".join(predicate_schema_list)

                if self.dataset_name == "lsat-ar":
                    prompt = fact_init_prompt.format(context=example['context'], query=example['question'], option=each_context, objects=object_schema, predicates=predicate_schema)
                else:
                    prompt = fact_init_prompt.format(context=each_context, objects=object_schema, predicates=predicate_schema)
                response = self.model.model_generate(system_input, prompt)
                output[key].append(response)

                cur_context_facts, cur_context_facts_verb = post_process_fact(args, response, cur_exmaples_facts_verb, cur_verb=each_context)
                object_schema_list, predicate_schema_list = self.update_schema(object_schema_list, predicate_schema_list, cur_context_facts)
                
                if self.dataset_name == "lsat-ar":
                    cur_exmaples_facts.append(cur_context_facts)
                else:
                    cur_exmaples_facts += cur_context_facts[:4]
                    cur_exmaples_facts_verb += cur_context_facts_verb[:4]

            if self.dataset_name == "boxes":
                cur_exmaples_op_facts = []
                for each_op_context in split_contexts_op:
                    prompt = op_fact_init_prompt.format(context=each_op_context, objects=object_schema, predicates=predicate_schema)
                    response = self.model.model_generate(system_input, prompt)
                    output[key].append(response) 

                    cur_context_facts, cur_context_facts_verb = post_process_fact(args, response, [_[0] for _ in cur_exmaples_op_facts], cur_verb=each_op_context)
                    cur_exmaples_op_facts += cur_context_facts
                all_example_fact[key] = [cur_exmaples_facts, cur_exmaples_op_facts]
            else:
                all_example_fact[key] = cur_exmaples_facts

            if key not in all_example_memory_schema or all_example_memory_schema[key] != [object_schema_list, predicate_schema_list]:
                all_example_memory_schema[key] = [object_schema_list, predicate_schema_list]

            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}_response.json'), "w") as f:
                json.dump(output, f, indent=1)
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}.json'), "w") as f:
                json.dump(all_example_fact, f, indent=1)
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json'), "w") as f:
                json.dump(all_example_memory_schema, f, indent=1)

    def rule_initialization(self, examples):
        print("Start rule initialization.")
        system_input, rule_init_prompt = load_prompt_rule_init(self.args)

        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}_response.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}_response.json'), "r") as f:
                output = json.load(f)
        else:
            output = {}
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}.json'), "r") as f:
                all_rule_dict = json.load(f)
        else:
            all_rule_dict = {}
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json'), "r") as f:
                all_example_memory_schema = json.load(f)
        else:
            all_example_memory_schema = {}  

        for example in tqdm(examples):
            if self.dataset_name == "clutrr":
                key = example["query"] 
                rules = example['rule']
            elif self.dataset_name == "proofwriter":
                key = " ".join(example['facts']) + " Is it true that " + example["question"][:-1] + "?"
                rules = example['rules']
            elif self.dataset_name == "lsat-ar":
                key = example["question"] + " " + example['context']
                split_idx = example['context'].rfind(": ")
                context_background = example['context'][:split_idx+1].strip()
                rules = example['context'][split_idx+1:].strip().split(". ")
                rules = [_+"." if _[-1] != "." else _ for _ in rules]
            
            output[key] = []
            if key in all_example_memory_schema:
                object_schema_list, predicate_schema_list = all_example_memory_schema[key]
            else:
                object_schema_list = []
                predicate_schema_list = []
            for each_rule in rules:
                if each_rule not in all_rule_dict:
                    if len(object_schema_list) == 0:
                        object_schema = "null"
                    else:
                        object_schema = ", ".join(object_schema_list)
                    if len(predicate_schema_list) == 0:
                        predicate_schema = "null"   
                    else:
                        predicate_schema = ", ".join(predicate_schema_list)

                    if self.dataset_name == "lsat-ar":
                        prompt = rule_init_prompt.format(context=context_background, rule=each_rule, objects=object_schema, predicates=predicate_schema)
                    else:
                        prompt = rule_init_prompt.format(rule=each_rule, objects=object_schema, predicates=predicate_schema)
                    response = self.model.model_generate(system_input, prompt)

                    all_rule_dict[each_rule] = post_process_rule(args, response)
                    if all_rule_dict[each_rule] is None:
                        response = self.model.model_generate(system_input, prompt)
                        all_rule_dict[each_rule] = post_process_rule(args, response)
                    object_schema_list, predicate_schema_list = self.update_schema(object_schema_list, predicate_schema_list, all_rule_dict[each_rule], is_fact=False)
                    output[key].append(response)

            if key not in all_example_memory_schema or all_example_memory_schema[key] != [object_schema_list, predicate_schema_list]:
                all_example_memory_schema[key] = [object_schema_list, predicate_schema_list]
            
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}_response.json'), "w") as f:
                json.dump(output, f, indent=1)
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}.json'), "w") as f:
                json.dump(all_rule_dict, f, indent=1)
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json'), "w") as f:
                json.dump(all_example_memory_schema, f, indent=1)

    def query_symbolize(self, examples):
        print("Start query symbolization.")
        system_input, fact_init_prompt = load_prompt_fact_init(self.args)

        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/queries/{self.split}_{self.model_name}.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/queries/{self.split}_{self.model_name}.json'), "r") as f:
                all_example_queries = json.load(f)
        else:
            all_example_queries = {}
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json'), "r") as f:
                all_example_memory_schema = json.load(f)
        else:
            all_example_memory_schema = {}

        for example in tqdm(examples):
            assert self.dataset_name == "proofwriter"
            key = " ".join(example['facts']) + " Is it true that " + example["question"][:-1] + "?"

            if key in all_example_memory_schema:
                object_schema_list, predicate_schema_list = all_example_memory_schema[key]
            else:
                object_schema_list = []
                predicate_schema_list = []

            each_context = example["question"]
            if len(object_schema_list) == 0:
                object_schema = "null"
            else:
                object_schema = ", ".join(object_schema_list)
            if len(predicate_schema_list) == 0:
                predicate_schema = "null"   
            else:
                predicate_schema = ", ".join(predicate_schema_list)

            prompt = fact_init_prompt.format(context=each_context, objects=object_schema, predicates=predicate_schema)
            response = self.model.model_generate(system_input, prompt)

            cur_context_facts, cur_context_facts_verb = post_process_fact(args, response, [], cur_verb=each_context)
            
            all_example_queries[key] = cur_context_facts
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/queries/{self.split}_{self.model_name}.json'), "w") as f:
                json.dump(all_example_queries, f, indent=1)

    def symbolic_rule_grounding(self, rules, facts, all_rule_dict, query_variables=None, state_facts_dict=None, thres=0):
        if state_facts_dict is not None:
            # chronological operational tasks 
            predicates = set([_[0] for _ in facts])
            target_objects = []
            for _ in facts:
                for each_var in _[1:]:
                    if "box" in each_var.lower():
                        target_objects.append(each_var)
            target_objects = set(target_objects)

            candidate_rules = []
            for each_rule in rules:
                rule_predicates = set([_[0] for _ in all_rule_dict[each_rule][1]])
                if len(predicates & rule_predicates) > 0:
                    candidate_rules.append(each_rule)
            candidate_state_facts = []
            for each_target in target_objects:
                if each_target in state_facts_dict:
                    candidate_state_facts.append(state_facts_dict[each_target][0])
            return " ".join(candidate_rules), " ".join(candidate_state_facts)
        else:
            # static reasoning tasks
            if type(thres) == list:
                # constraint satisfaction
                max_overlap_num = -1
                for rule_id, each_rule in enumerate(rules):
                    if rule_id not in thres:
                        if len(set(facts[1]) & set(all_rule_dict[each_rule][0])) > max_overlap_num:
                            max_rule_id = rule_id
                            max_overlap_num = len(set(facts[1]) & set(all_rule_dict[each_rule][0]))
                thres.append(max_rule_id)
                return rules[max_rule_id], thres
            else:
                # logical reasoning
                variables_strs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'X', 'Y', 'Z']
                if self.dataset_name == "clutrr":
                    facts = [_ for _ in facts if len(_) == 3]

                valid_rule_facts = []
                valid_facts = []
                for rule_id in range(len(rules)):
                    each_rule_verb = rules[rule_id]
                    each_rule_sym = all_rule_dict[each_rule_verb]
                    premises = each_rule_sym[1]

                    # predicate satisfication check
                    premise_predicate_list = []
                    for each_prem in premises:
                        premise_predicate_list.append(each_prem[0])

                    comb_choice = list(combinations(list(range(len(facts))), len(premises)))
                    for iths in comb_choice:
                        if max(iths) < thres:
                            continue
                        cur_fact_predicates = [facts[n][0] for n in iths]
                        cur_facts = [facts[n][1:] for n in iths]
                        if self.dataset_name == "proofwriter" and query_variables is not None:
                            cur_fact_variables = sum([_ for _ in cur_facts], [])
                            cur_fact_variables = [_.lower() for _ in cur_fact_variables]
                            if len(set(cur_fact_variables) & set(query_variables)) == 0:
                                continue
                        
                        if (self.dataset_name == "clutrr" and hard_predicate_match(cur_fact_predicates, premise_predicate_list)) or \
                            (self.dataset_name == "proofwriter" and soft_predicate_match(premise_predicate_list, cur_fact_predicates)):
                            ### variable statisfication check
                            if len(set(premise_predicate_list)) == len(premise_predicate_list):
                                ### reorder selected facts
                                reordered_cur_facts = []
                                for each_predicate in premise_predicate_list:
                                    if each_predicate in cur_fact_predicates:
                                        fact_id = cur_fact_predicates.index(each_predicate)
                                    else:
                                        fact_id = get_predicate_id(each_predicate, cur_fact_predicates)
                                    assert fact_id > -1
                                    reordered_cur_facts.append(cur_facts[fact_id])

                                cur_satify = True
                                variables_instance = {}
                                for i in range(len(premises)):
                                    for n in range(len(premises[i][1])):
                                        if premises[i][1][n] not in variables_instance:
                                            if n < len(reordered_cur_facts[i]):
                                                variables_instance[premises[i][1][n]] = [reordered_cur_facts[i][n].lower()]
                                            else:
                                                cur_satify = False
                                        else:
                                            if n < len(reordered_cur_facts[i]):
                                                if reordered_cur_facts[i][n].lower() not in variables_instance[premises[i][1][n]]:
                                                    variables_instance[premises[i][1][n]].append(reordered_cur_facts[i][n].lower())
                                            else:
                                                cur_satify = False
                                if cur_satify:
                                    for each_variable in variables_instance:
                                        if len(variables_instance[each_variable]) > 1:
                                            cur_satify = False
                                            break
                                        if each_variable not in variables_strs:
                                            if len(variables_instance[each_variable]) > 1 or variables_instance[each_variable][0] != each_variable.lower():
                                                cur_satify = False
                                                break
                                    if len(set(list(variables_instance.keys()))) != len(set(sum(list(variables_instance.values()), []))):
                                        cur_satify = False
                                    if cur_satify:
                                        if [rule_id, iths] not in valid_rule_facts:
                                            valid_facts.append(iths)
                                            valid_rule_facts.append([rule_id, iths])
                            else:
                                perm_choice = list(permutations(list(range(len(cur_facts))), len(cur_facts)))
                                for each_perm in perm_choice:
                                    new_cur_fact_predicates = [cur_fact_predicates[n] for n in each_perm]
                                    predicate_satisfy = True
                                    for j, each_predicate in enumerate(new_cur_fact_predicates):
                                        if each_predicate != premise_predicate_list[j] and (not predicate_match(each_predicate, premise_predicate_list[j])):
                                            predicate_satisfy = False
                                            break
                                    if not predicate_satisfy:
                                        continue
                                    reordered_cur_facts = [cur_facts[n] for n in each_perm]

                                    cur_satify = True
                                    variables_instance = {}
                                    for i in range(len(premises)):
                                        for n in range(len(premises[i][1])):
                                            if premises[i][1][n] not in variables_instance:
                                                if n < len(reordered_cur_facts[i]):
                                                    variables_instance[premises[i][1][n]] = [reordered_cur_facts[i][n].lower()]
                                                else:
                                                    cur_satify = False
                                            else:
                                                if n < len(reordered_cur_facts[i]):
                                                    if reordered_cur_facts[i][n].lower() not in variables_instance[premises[i][1][n]]:
                                                        variables_instance[premises[i][1][n]].append(reordered_cur_facts[i][n].lower())
                                                else:
                                                    cur_satify = False
                                    if cur_satify:
                                        for each_variable in variables_instance:
                                            if len(variables_instance[each_variable]) > 1:
                                                cur_satify = False
                                                break
                                            if each_variable not in variables_strs:
                                                if len(variables_instance[each_variable]) > 1 or variables_instance[each_variable][0] != each_variable.lower():
                                                    cur_satify = False
                                                    break
                                        if len(set(list(variables_instance.keys()))) != len(set(sum(list(variables_instance.values()), []))):
                                            cur_satify = False
                                        if cur_satify:
                                            if [rule_id, iths] not in valid_rule_facts:
                                                valid_facts.append(iths)
                                                valid_rule_facts.append([rule_id, iths])
                                                break

            return valid_rule_facts, len(facts)

    def rule_implementation(self, examples):
        print("Start rule implementation.")
        system_input, rule_implement_prompt = load_prompt_rule_implementation(self.args)

        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}_implement.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}_implement.json'), "r") as f:
                all_example_fact_implement = json.load(f)
        else:
            all_example_fact_implement = {}
        with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}.json'), "r") as f:
            all_example_fact = json.load(f)
        if self.dataset_name != "boxes":
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/rules/{self.split}_{self.model_name}.json'), "r") as f:
                all_rule_dict = json.load(f)
                print("rule base size", len(all_rule_dict))
        with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/memory_schema/{self.split}_{self.model_name}.json'), "r") as f:
            all_example_memory_schema = json.load(f)
        if os.path.exists(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/implementation/{self.split}_{self.model_name}_response.json')):
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/implementation/{self.split}_{self.model_name}_response.json'), "r") as f:
                output = json.load(f)
        else:
            output = {}
        if self.dataset_name == "proofwriter":
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/queries/{self.split}_{self.model_name}.json'), "r") as f:
                all_example_queries = json.load(f)

        for example in tqdm(examples):
            if self.dataset_name == "clutrr":
                key = example["query"]
                question = "How is " + example["query"].split(" How is ")[1]
                qeury_objects = example["query"].split(". ")[-1].split("How is")[1][:-1].strip().split(" related to ")
                rules = []
                for _ in example['rule']:
                    if _ not in rules:
                        rules.append(_)
            elif self.dataset_name == "proofwriter":
                key = " ".join(example['facts']) + " Is it true that " + example["question"][:-1] + "?"
                question = "Is it true that " + example["question"][:-1] + "?"
                qeury_sym = [_.lower() for _ in all_example_queries[key][0][1]]
                qeury_objects = qeury_sym[1:]
                rules = []
                for _ in example['rules']:
                    if _ not in rules:
                        rules.append(_)
            elif self.dataset_name == "boxes":
                question = example["question"]
                key = example['context']
                state_facts, operational_facts = all_example_fact[key]
                state_facts_dict = {}
                for each_state in state_facts:
                    assert len(each_state) == 2
                    position = each_state[1][1]
                    state_facts_dict[position] = each_state
            elif self.dataset_name == "lsat-ar":
                key = example["question"] + " " + example['context']
                question = example["question"]
                split_idx = example['context'].rfind(": ")
                context_background = example['context'][:split_idx+1].strip()
                rules = example['context'][split_idx+1:].strip().split(". ")
                rules = [_+"." if _[-1] != "." else _ for _ in rules]

            object_schema_list, predicate_schema_list = all_example_memory_schema[key]
            if len(object_schema_list) == 0:
                object_schema = "null"
            else:
                object_schema = ", ".join(object_schema_list)
            if len(predicate_schema_list) == 0:
                predicate_schema = "null"   
            else:
                predicate_schema = ", ".join(predicate_schema_list)

            if self.dataset_name == "boxes":
                for each_operation in operational_facts:
                    each_operation_str = each_operation[0]
                    all_sym_operations = each_operation[1:]
                    cand_rules, cand_facts = self.symbolic_rule_grounding(rules_boxes, all_sym_operations, rules_dict_boxes, state_facts_dict=state_facts_dict)

                    prompt = rule_implement_prompt.format(state_facts=cand_facts, op_facts=each_operation_str, rule=cand_rules, objects=object_schema, predicates=predicate_schema)
                    response = self.model.model_generate(system_input, prompt)
                    state_facts_dict = post_process_implement_box(response, state_facts_dict)
                output[key] = state_facts_dict
            elif self.dataset_name == "lsat-ar":
                option_conflit = {}
                option_facts = {}
                for option_id, each_fact in enumerate(example['options']):
                    thres = []
                    epoch = 0
                    cur_fact_list = [each_fact]
                    while epoch < len(rules):
                        cur_facts = "\n".join(["- "+_ for _ in cur_fact_list])
                        valid_rule, thres = self.symbolic_rule_grounding(rules, all_example_fact[key][option_id], all_rule_dict, thres=thres)
                        prompt = rule_implement_prompt.format(context=context_background, rule=valid_rule, query=question, facts=cur_facts, objects=object_schema, predicates=predicate_schema)
                        response = self.model.model_generate(system_input, prompt)
                        cur_implement_output = post_process_implement(args, response)
                        if cur_implement_output is not None:
                            judgement, new_fact = cur_implement_output
                            if judgement:
                                option_conflit[each_fact] = True
                                break
                            elif new_fact is not None:
                                new_added_var = set(new_fact[1]) -  set(all_example_fact[key][option_id][1])
                                if len(new_added_var) > 0:
                                    cur_fact_list.append(f'{len(cur_fact_list)+1}. {new_fact[0]}')
                                    all_example_fact[key][option_id][1] += list(new_added_var)
                        epoch += 1   
                    option_facts[each_fact] = cur_fact_list
                output[key] = [option_conflit, option_facts]                
            else:
                epoch = 0
                thres = 0
                jump = False
                output[key] = []
                while epoch < args.application_steps:
                    listed_facts = all_example_fact[key]
                    listed_symbolic_facts = [each[1] for each in listed_facts]
                    valid_rule_facts, thres = self.symbolic_rule_grounding(rules, listed_symbolic_facts, all_rule_dict, query_variables=qeury_objects, thres=thres)

                    if len(valid_rule_facts) == 0:
                        print("No valid rule facts.")
                        break
                    for each_valid_rule_fact in valid_rule_facts[:7]:
                        rule_id, fact_ids = each_valid_rule_fact
                        cur_rule = rules[rule_id]
                        cur_fact_list = " ".join([listed_facts[i][0] for i in fact_ids])

                        prompt = rule_implement_prompt.format(query=question, fact_list=cur_fact_list, rule=cur_rule, objects=object_schema, predicates=predicate_schema)
                        response = self.model.model_generate(system_input, prompt)
                        output[key].append(response)
                        cur_implement_output = post_process_implement(args, response)
                        if cur_implement_output is not None:
                            judgement, new_fact = cur_implement_output
                            if new_fact[1] not in listed_symbolic_facts:
                                all_example_fact[key].append([f'{len(all_example_fact[key])+1}. {new_fact[0]}.', new_fact[1]])

                            if self.dataset_name == "clutrr" and judgement and qeury_objects[0] in new_fact[1] and qeury_objects[1] in new_fact[1]:
                                jump = True
                                break
                            elif self.dataset_name == "proofwriter" and judgement:
                                all_exist = True
                                new_fact_lower = [_.lower() for _ in new_fact[1]]
                                for _ in qeury_sym:
                                    if _ in new_fact_lower or "not_" + _ in new_fact_lower:
                                        continue
                                    elif len(_) > 1 and _[-1] == "s" and _[:-1] in new_fact_lower:
                                        continue
                                    else:
                                        all_exist = False
                                        break
                                if all_exist:      
                                    jump = True
                                    break
                    epoch += 1
                    if jump:
                        break
            
                all_example_fact_implement[key] = all_example_fact[key]
                with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/facts/{self.split}_{self.model_name}_implement.json'), "w") as f:
                    json.dump(all_example_fact_implement, f, indent=1)
            with open(os.path.join(self.save_path, f'{self.dataset_name}/symbolic_memory/implementation/{self.split}_{self.model_name}_response.json'), "w") as f:
                json.dump(output, f, indent=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='../Output/')
    parser.add_argument('--application_steps', type=int, default=6)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--use_memory_schema', type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    wm_neurosymbolic_reasoner = WM_Neurosymbolic(args)
    examples = wm_neurosymbolic_reasoner.load_raw_dataset(args.split)

    if args.dataset_name == "clutrr":
        wm_neurosymbolic_reasoner.rule_initialization(examples=examples)
        wm_neurosymbolic_reasoner.fact_initialization(examples=examples)
    if args.dataset_name == "proofwriter":
        wm_neurosymbolic_reasoner.rule_initialization(examples=examples)
        wm_neurosymbolic_reasoner.fact_initialization(examples=examples)
        wm_neurosymbolic_reasoner.query_symbolize(examples=examples)
    if args.dataset_name == "boxes":
        wm_neurosymbolic_reasoner.fact_initialization(examples=examples)
    if args.dataset_name == "lsat-ar":
        wm_neurosymbolic_reasoner.fact_initialization(examples=examples)
        wm_neurosymbolic_reasoner.rule_initialization(examples=examples)

    wm_neurosymbolic_reasoner.rule_implementation(examples=examples)
