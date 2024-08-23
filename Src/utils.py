import os


def load_prompt_fact_init(args):
    file_path = os.path.join('./prompts', args.dataset_name, 'fact_initialization.txt')
    with open(file_path) as f:
        fact_init_prompt = f.read()
    
    if args.dataset_name == "clutrr":
        system_input = "You are an expert in determining kinship relationships."
    else:
        system_input = "You are a helpful assistant."

    if args.dataset_name == "boxes":
        file_path_op = os.path.join('./prompts', args.dataset_name, 'op_fact_initialization.txt')
        with open(file_path_op) as f:
            op_fact_init_prompt = f.read()
        return system_input, fact_init_prompt, op_fact_init_prompt
    else:
        return system_input, fact_init_prompt


def load_prompt_rule_init(args):
    if args.dataset_name == "boxes" and ("gpt-3.5" in args.model_name or "llama" in args.model_name):
        file_path = os.path.join('./prompts', args.dataset_name, 'rule_initialization_moreshots.txt')
    else:
        file_path = os.path.join('./prompts', args.dataset_name, 'rule_initialization.txt')
    with open(file_path) as f:
        rule_init_prompt = f.read()

    if args.dataset_name == "clutrr":
        system_input = "You are an expert in determining kinship relationships."
    else:
        system_input = "You are a helpful assistant."
    return system_input, rule_init_prompt


def load_prompt_rule_implementation(args):
    file_path = os.path.join('./prompts', args.dataset_name, 'rule_implementation.txt')
    with open(file_path) as f:
        rule_init_implement = f.read()
    
    if args.dataset_name == "clutrr":
        system_input = "You are an expert in determining kinship relationships. You will receive a query about the kinship between two individuals, and your task is to answer this query."
    elif args.dataset_name == "proofwriter":
        system_input = "You are an expert in logiacl reasoning. You will receive a context including a list of facts and inference rules, and a specific query. Your task is to answer this query following the provided rule."
    elif args.dataset_name == "boxes":
        system_input = "You are an expert in logical reasoning. You will receive a context including a list of state facts and operational facts, a list of rules and a specific query. Your task is to answer this query following the provided rule."
    elif args.dataset_name == "lsat-ar":
        system_input = 'You are an expert in logical reasoning. You will receive a context including background information followed by a list of constraint rules, and a specific query with five candidate options (A, B, C, D, E). Your task is to accurately select the answer that satisfies the provided rule.'
    return system_input, rule_init_implement


def hard_predicate_match(predicates_1, predicates_2):
    return set(predicates_1) == set(predicates_2) and len(predicates_1) == len(predicates_2)

def soft_predicate_match(predicates_1, predicates_2):
    if len(predicates_1) == len(predicates_2):
        for each_1 in predicates_1:
            each_splits_1 = each_1.split("_")
            has_supp_predicate = False
            for each_2 in predicates_2:
                each_splits_2 = each_2.split("_")
                if ("not" in each_splits_1 and "not" in each_splits_2) or ("not" not in each_splits_1 and "not" not in each_splits_2):
                    joint_split = set(each_splits_1) & set(each_splits_2)
                    if len(joint_split) > 0 and joint_split != {"not"}:
                        has_supp_predicate = True
                        break
            if not has_supp_predicate:
                return False
        return True
    else:
        return False

def get_predicate_id(cur_pred, all_predicates):
    cur_splits = cur_pred.split("_")
    for i, each in enumerate(all_predicates):
        each_splits = each.split("_")
        if ("not" in cur_splits and "not" in each_splits) or ("not" not in cur_splits and "not" not in each_splits):
            joint_split = set(cur_splits) & set(each_splits)
            if len(joint_split) > 0 and joint_split != {"not"}:
                return i
    return -1

def predicate_match(pred_1, pred_2):
    pred_1_splits = pred_1.split("_")
    pred_2_splits = pred_2.split("_")
    if ("not" in pred_1_splits and "not" in pred_2_splits) or ("not" not in pred_1_splits and "not" not in pred_2_splits):
        joint_split = set(pred_1_splits) & set(pred_2_splits)
        if len(joint_split) > 0 and joint_split != {"not"} and joint_split != {"of"}:
            return True
        else:
            return False
    else:
        return False
    

rules_boxes = [
    "If remove the contents X from Box A, then X are not in Box A.",
    "If move the contents X from Box A to Box B, then X are not in Box A and X are in Box B.",
    "If put the contents X into Box A, then X are in Box A."
]

rules_dict_boxes = {
    "If remove the contents X from Box A, then X are not in Box A.": [
        [
            "not_in",
            [
                "X",
                "Box A"
            ]
        ],
        [
            [
                "remove_from",
                [
                    "X",
                    "Box A"
                ]
            ]
        ],
    ],
    "If move the contents X from Box A to Box B, then X are not in Box A and X are in Box B.": [
        [
            [
                "not_in",
                [
                    "X",
                    "Box A"
                ]
            ],
            [
                "in",
                [
                    "X",
                    "Box B"
                ]
            ],
        ],
        [
            [
                "move_from_to",
                [
                    "X",
                    "Box A",
                    "Box B"
                ]
            ]
        ],
    ],
    "If put the contents X into Box A, then X are in Box A.": [
        [
            "in",
            [
                "X",
                "Box A"
            ]
        ],
        [
            [
                "put_into",
                [
                    "X",
                    "Box A"
                ]
            ]
        ],
    ],
}