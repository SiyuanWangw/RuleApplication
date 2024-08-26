import re

def post_process_fact(args, response, cur_exmaples_facts_verb, cur_verb=None):
    new_facts = []
    new_facts_verb = []

    if args.dataset_name == "clutrr":
        if "Reverse Facts:" in response:
            response = response.split("Reverse Facts:")[0].strip()
    all_facts = response.split("\n")
    for each_fact in all_facts:
        if args.dataset_name == "clutrr":
            if "Fact" in each_fact:
                continue
            if len(each_fact) < 20 or " [" not in each_fact:
                continue
            if each_fact.startswith("- "):
                each_fact = each_fact[2:].strip()
            cur_verb = each_fact.split(" [")[0]
            cur_sym_str = each_fact.split(" [")[1].split("]")[0].strip()
        elif args.dataset_name == "lsat-ar":
            pattern = r'assign\s*\(.*?\)'
            matches = re.findall(pattern, each_fact)
            if len(matches) == 0:
                continue
            cur_sym_str = matches[0]
        else:
            if "Fact:" in each_fact:
                cur_sym_str = each_fact.split("Fact:")[1].strip()
            else:
                cur_sym_str = each_fact.strip()

        if "(" in cur_sym_str and ")" in cur_sym_str:
            predicate = cur_sym_str.split("(")[0]
            variables = cur_sym_str.split("(")[1].split(")")[0].strip().split(", ")

            if args.dataset_name == "lsat-ar":
                for each_var in variables:
                    if len(each_var) > 0 and each_var not in new_facts:
                        new_facts.append(each_var)
            else:
                cur_syms = tuple([predicate] + variables)
                if cur_verb in new_facts_verb or cur_verb in cur_exmaples_facts_verb:
                    continue
                if args.dataset_name == "clutrr" and len(cur_syms) < 3:
                    continue
                new_facts.append([f"{len(new_facts)+len(cur_exmaples_facts_verb)+1}. {cur_verb}", cur_syms])
                new_facts_verb.append(cur_verb)
    if args.dataset_name == "lsat-ar":
        return [f"1. {cur_verb}", new_facts], cur_verb
    else:
        return new_facts, new_facts_verb

def post_process_rule(args, response):
    all_rules_text = response.split("Symbolic Rule:")[-1].strip()
    pattern = r'constraint\s*\(.*?\)'

    if args.dataset_name == "lsat-ar":
        all_rules_variables = []
        matches = re.findall(pattern, all_rules_text)
        for each_match in matches:
            each_constraint_vars = each_match.split("(")[1].split(")")[0].strip().split(", ")
            all_rules_variables += each_constraint_vars
        all_rules_variables = list(set(all_rules_variables))
        return [all_rules_variables, all_rules_variables]
    else: 
        all_rules = all_rules_text.split("\n")
        for cur_symbolic_text in all_rules:
            cur_symbolic_text = cur_symbolic_text.strip()
            if " :- " in cur_symbolic_text and cur_symbolic_text.count(" :- ") == 1:
                conclusion, premises = cur_symbolic_text.split(" :- ")
                conclusion_predicate = conclusion.split("\n")[-1].strip().split("(")[0]
                conclusion_args = conclusion.split("(")[1].split(")")[0].strip().split(", ")
                conclusion_triplet = [conclusion_predicate, conclusion_args]

                premises = premises.strip()
                premise_list = premises[:-1].split("), ") if premises[-1] == "." else premises.split("), ")
                premise_list = [each if each[-1]==")" else each+")" for each in premise_list]
                premise_triplet_list = []
                for each_premise in premise_list:
                    if "(" in each_premise and ")" in each_premise:
                        each_premise_predicate = each_premise.split("(")[0]
                        each_premise_args = each_premise.split("(")[1].split(")")[0].strip().split(", ")
                        premise_triplet_list.append([each_premise_predicate, each_premise_args])
                if len(premise_triplet_list) > 0:
                    return [conclusion_triplet, premise_triplet_list]
        
        return None

def post_process_implement(args, response):
    response = response.replace("\n\n", "\n")

    if args.dataset_name != "lsat-ar":
        if "New fact:" in response and "Judgement:" in response:
            new_fact = response.split("New fact:")[1].split("Judgement:")[0].strip()
            judgement = response.split("Judgement:")[1].strip()
            if ". [" not in new_fact:
                return None
            else:
                new_verbalized_fact = new_fact.split(". [")[0].strip()
                new_symbolic_fact = new_fact.split(". [")[1].split("]")[0].strip()

                assert "(" in new_symbolic_fact and ")" in new_symbolic_fact
                predicate = new_symbolic_fact.split("(")[0]
                variables = new_symbolic_fact.split("(")[1].split(")")[0].strip().split(", ")
                cur_syms = [predicate] + variables

                new_fact = [new_verbalized_fact, cur_syms]
            if "Yes" in judgement:
                return True, new_fact
            else:
                assert "No" in judgement, judgement
                return False, new_fact
        else:
            return None
    else:
        if "Judgement:" not in response:
            return None
        judgement = response.split("Judgement:")[1].split("\n")[0].strip()
        if "Yes" in judgement:
            return True, None
        if "New fact:" not in response:
            return None
        new_fact = response.split("New fact:")[1].strip()
        if ". [" not in new_fact:
            return False, None
        else:
            new_verbalized_fact = new_fact.split(". [")[0].strip()
            all_new_fact_vars = []
            pattern = r'assign\s*\(.*?\)'
            matches = re.findall(pattern, new_fact.split(". [")[1].split("]")[0].strip())
            for each_match in matches:
                new_fact_vars = each_match.split("(")[1].split(")")[0].strip().split(", ")
                all_new_fact_vars += new_fact_vars
            all_new_fact_vars = list(set(all_new_fact_vars))
            new_fact = [new_verbalized_fact, all_new_fact_vars]
            return False, new_fact

def post_process_implement_box(response, state_facts_dict):
    if "New facts:" in response:
        new_facts = response.split("New facts:")[1].strip()
    else:
        new_facts = response.split("Rule Implementation:")[1].strip()
    response = response.replace("Box_", "Box ")
    all_new_facts = new_facts.split("\n")
    for each_new_fact in all_new_facts:
        if "[" in each_new_fact and "]" in each_new_fact:
            new_fact_verb = each_new_fact.split("[")[0].strip()
            new_facts_sym = each_new_fact.split("[")[1].split("]")[0].strip()

            cur_sentence_facts = new_facts_sym.split("), ")
            parsed_cur_state_facts = []
            new_target = []
            for each_fact in cur_sentence_facts:
                if each_fact[-1] == ")":
                    each_fact = each_fact[:-1]
                assert "(" in each_fact, each_fact
                predicate = each_fact.split("(")[0]
                variables = each_fact.split("(")[1].split(", ")
                parsed_cur_state_facts.append(tuple([predicate] + variables))
                new_target.append(variables[0])
            state_facts_dict[new_target[0]] = [new_fact_verb] + parsed_cur_state_facts
    return state_facts_dict
