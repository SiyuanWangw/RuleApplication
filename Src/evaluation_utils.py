all_relations = ['daughter-in-law', 'father-in-law', 'son-in-law', 'mother-in-law', 'granddaughter', 'grandson', 'grandmother', 'grandfather', 'grandchild',
   'son', 'daughter', 'niece', 'nephew', 'mother', 'father', 'brother', 'sister', 'aunt', 'uncle', 'cousin', 'wife', 'husband']
relation_reverse_dict = {
    'son': ['father', 'mother'],
    'daughter': ['father', 'mother'],
    'niece': ['uncle', 'aunt'],
    'nephew': ['uncle', 'aunt'],
    'mother': ['son', 'daughter'],
    'father': ['son', 'daughter'],
    'granddaughter': ['grandfather', 'grandmother'],
    'grandson': ['grandfather', 'grandmother'],
    'grandmother': ['grandson', 'granddaughter'],
    'grandfather': ['grandson', 'granddaughter'],
    'grandchild': ['grandson', 'granddaughter', 'grandparent'],
    'daughter-in-law': ['father-in-law', 'mother-in-law', 'father', 'mother'],
    'father-in-law': ['son-in-law', 'daughter-in-law', 'son', 'daughter'],
    'son-in-law': ['father-in-law', 'mother-in-law', 'father', 'mother'],
    'mother-in-law': ['son-in-law', 'daughter-in-law', 'son', 'daughter'],
    'brother': ['brother', 'sister'],
    'sister': ['brother', 'sister'],
    'aunt': ['nephew', 'niece'],
    'uncle': ['nephew', 'niece'],
    'cousin': ['cousin'],
    'wife': ['husband'],
    'husband': ['wife']
}

def find_individual(text):
    i = 0
    start = -1
    end = 0
    while i < len(text):
        if text[i].isupper():
            start = i
            break
        i += 1
    i += 1
    # print(start, i, text[start:])
        
    while i < len(text):
        if not text[i].isalpha() and start >= 0:
            end = i
            break
        i += 1
    # print(end, i, text[end:])
    return text[start:end], end

def parse_answer(reference, prediction, verbose=False):
    prediction = prediction.replace("Answer", "")
    # print(reference)
    # print(prediction)

    if type(reference) == str:
        for each in all_relations:
            if each in reference:
                relation_ref = each
                name1_ref, end = find_individual(reference.strip())
                name2_ref = find_individual(reference[end:].strip())[0]
                break
        if reference.find(relation_ref) > reference.find(name1_ref) and reference.find(relation_ref) > reference.find(name2_ref):
            pass
        else:
            if reference[reference.find(relation_ref)+len(relation_ref)+1:reference.find(relation_ref)+len(relation_ref)+3] == "of":
                pass
            else:
                temp = name1_ref
                name1_ref = name2_ref
                name2_ref = temp
    elif type(reference) == list:
        relation_ref = reference[1]
        name1_ref = reference[2]
        name2_ref = reference[0]
    if verbose:
        print("ref", name1_ref, relation_ref, name2_ref)
    
    pred_has_rela = False
    for each in all_relations:
        if each in prediction:
            pred_has_rela = True
            relation_pred = each
            name1_pred, end = find_individual(prediction.strip())
            name2_pred = find_individual(prediction[end:].strip())[0]
            break
    if not pred_has_rela:
        return False
    if prediction.find(relation_pred) > prediction.find(name1_pred) and prediction.find(relation_pred) > prediction.find(name2_pred):
        pass
    else:
        if prediction[prediction.find(relation_pred)+len(relation_pred)+1:prediction.find(relation_pred)+len(relation_pred)+3] == "of":
            pass
        else:
            temp = name1_pred
            name1_pred = name2_pred
            name2_pred = temp
    if verbose:
        print("pred", name1_pred, relation_pred, name2_pred)

    if (name1_pred == name1_ref or name1_pred == name2_ref) and (name2_pred == name1_ref or name2_pred == name2_ref):
        if name1_pred == name1_ref and name2_pred == name2_ref:
            if relation_pred.split("-")[0] == relation_ref.split("-")[0]:
                return True
        elif name1_pred == name2_ref and name2_pred == name1_ref:
            if relation_pred.split("-")[0] in relation_reverse_dict[relation_ref]:
                return True
        return False
    else:
        return False