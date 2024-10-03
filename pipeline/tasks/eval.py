import json
import argparse
from difflib import SequenceMatcher
import logging
import os
import random
import numpy as np
import pandas as pd
import pyalign
from sklearn.metrics import classification_report
from word2number import w2n
from nltk.translate.bleu_score import sentence_bleu


num_txt={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5}

def is_number(s):
    try:
        s = s.replace(",", "")
        float(s)
        return True
    except ValueError:
        return False

q_bi=[ 'Is it known whether this drug is administered parenterally?',
 'Is it known whether this drug is applied topically?',
  'Is this compound a small molecule polymer, such as polystyrene sulfonate?',
 'Is this molecule characterized by a small molecular structure or a protein sequence?',
 'Does this compound satisfy the rule-of-three criteria?',
 'Determine if this molecule is inorganic, meaning it contains only metal atoms and fewer than two carbon atoms.',
 'Is there a black box warning associated with this drug?',
 'Is this drug used for therapeutic purposes, rather than for imaging, additives, or other non-therapeutic applications?',
 'Has this approved drug been withdrawn due to toxicity reasons for all indications, populations, and doses in at least one country (not necessarily the US)?',
 'Is it known if this drug is the first approved in its class, regardless of the indication or route of administration, acting on a specific target?',
 'Is it known whether this drug is taken orally?',
 'Is the drug administered in this specific form, such as a particular salt?',
 'Determine if this compound is a prodrug.',
 ]
q_clf=['What is the highest development stage achieved for this compound across all indications? Please respond with Approved, Phase 3 Clinical Trials, Phase 2 Clinical Trials, Phase 1 Clinical Trials, Early Phase 1 Clinical Trials, or Clinical Phase Unknown.',
 'Determine if this drug is administered as a racemic mixture, a single stereoisomer, an achiral molecule, or has an unknown chirality.',
 'Determine the type of availability for this drug.',
 'Is this compound an acid, a base, or neutral?',
 'What is the classification of this molecule? Please respond with Small Molecule, Protein, Antibody, Oligosaccharide, Oligonucleotide, Cell, Enzyme, Gene, or Unknown.',
 ]
q_num=['What is the polar surface area (PSA) value of this compound?',
 "How many violations of Lipinski's Rule of Five are there for this compound, using the HBA_LIPINSKI and HBD_LIPINSKI counts?",
 'What is the calculated ALogP value for this compound?',
 'How many heavy (non-hydrogen) atoms does this compound have?',
 'How many rotatable bonds does this compound have?',
 'How many aromatic rings does this compound have?',
 "How many hydrogen bond acceptors are there in this compound, calculated according to Lipinski's original rules (i.e., counting N and O atoms)?",
 "How many violations of Lipinski's Rule of Five (using HBA and HBD definitions) are there for this compound?",
 'How many hydrogen bond acceptors does this compound have?',
 "How many hydrogen bond donors are there in this compound, calculated according to Lipinski's original rules (i.e., counting NH and OH groups)?",
 'How many hydrogen bond donors does this compound have?',
 'What is the molecular weight of this compound\'s parent molecule?',
 ]
q_sen=[#'What is the first recorded year of approval for this drug?',
 "What is the definition of this compound's USAN stem?",
 'Which USAN substem can this drug or clinical candidate name be matched with?',
 "Please provide a description of this drug's mechanism of action.",
 'What is the molecular formula of this compound, including any salt that it may have?',]

q_all=list(set(q_bi+q_clf+q_num+q_sen))
q_structured = list(set(q_bi+q_clf+q_num))

# answer set for multiclass classification
q_multicls_ans = [
    ['Approved', 'Phase 1 Clinical Trials', 'Phase 3 Clinical Trials', 'Phase 2 Clinical Trials', 'Clinical Phase unknown'],
    ['An achiral molecule', 'Racemic mixture', 'Single stereoisomer', 'Unknown chirality'],
    ['prescription only', 'withdrawn', 'over the counter', 'discontinued', 'unknown'],
    ['NEUTRAL', 'ZWITTERION', 'BASE', 'ACID'],
    ['Protein', 'Oligosaccharide', 'Small molecule', 'Unknown'],
]

q_multi2ans = {q_clf[ii]: [a.lower() for a in q_multicls_ans[ii]] for ii in range(len(q_clf))}

q_idx = {'Is this molecule characterized by a small molecular structure or a protein sequence?': [
    "It has a small molecule structure.".lower().strip(" \n."),
    "It has both.".lower().strip(" \n.")
]}

def get_bi_label(q, ans):
    if isinstance(ans, list):
        assert (len(ans) == 1)
        ans = ans[0]
    if q in q_idx:
        ans_list = q_idx[q]
        ans = ans.lower().strip(" \n.")
        # if ans not in ans_list:
        #     ans_list.append(ans)
        for i, ref in enumerate(ans_list):
            if ref in ans or ans in ref:
                return i
        if "small molecule" in ans:
            return 0
        else:
            return 1
        # logging.warning(f"Didn't find {ans=} for {ans_list=}")
        # return random.choice([0, 1])
        # return ans_list.index(ans)
    ans = ans.lower().strip(" .")
    # if ans == "yes":
    if 'yes' in ans:
        return 1
    # if ans == "no":
    if 'no' in ans or "doesn't" in ans or "isn't" in ans:
        return 0
    # return random.choice([0, 1])
    return random.randint(0, 1)


def string_similarity(a, b):
    """Computer similarity of two strings."""
    return SequenceMatcher(None, a, b).ratio()


def match_score(labels, ans):
    ans = ans.lower()
    scores = []
    for label in labels:
        label = label.lower()
        score = pyalign.local_alignment(ans, label).score
        scores.append(score / min(len(ans), len(label)))
    # s = max(scores)
    return scores


def get_multicls_label(q, ans):
    """Get answer label. If not found in the answer set, return -1."""
    if isinstance(ans, list):
        assert (len(ans) == 1)
        ans = ans[0]
    ans_set = q_multi2ans[q]
    if not ans:
        return random.randint(0, len(ans_set) - 1)
    # for i, a in enumerate(ans_set):
    #     if a in ans.lower():
    #         return i
    # match_scores = [string_similarity(a.lower(), ans.lower()) for a in ans_set]
    match_scores = match_score(ans_set, ans)
    max_score = np.max(match_scores)
    return np.argmax(match_scores).item()


def get_cls_report(pred, labels):
    report = classification_report(
        labels, pred, labels=list(set(labels)), output_dict=True
    )
    # report is a dict {'0': {}, '1': {}, 'macro avg': {}, 'weighted avg': {}}
    # return report['macro avg']
    return report


def write_to_file(this_result, output_file):
    """Append to output csv file if it exits, otherwise write to it."""
    if os.path.exists(output_file):
        with open(output_file) as f:
            old_result = pd.read_csv(f)
        this_result = pd.merge(old_result, this_result, how='outer')
        this_result = this_result.drop(columns=['Unnamed: 0'])
        # this_result.reset_index()

    with open(output_file, 'w') as f:
        this_result.to_csv(f)


def eval_generation(results, model_name, output_file0):
    q_res_bi={}

    q_lst={}
    q_cnt=[]

    if isinstance(results, dict):
        for ans_lst in results.values():

            for ans in ans_lst:
                if ans[0] in q_lst:
                    q_lst[ans[0]].append((ans[1],ans[2]))
                    #q_lst[ans[0]].append((ans[1],ans[2][0].split(':')[-1]))
                else:
                    q_lst[ans[0]]=[(ans[1],ans[2])]
                    #q_lst[ans[0]]=[(ans[1],ans[2][0].split(':')[-1])]

    # results could be a list, so convert it to a dict
    elif isinstance(results, list):
        # results = [(question, gt, pred)]

        for ans in results:
            for q in q_all:
                if q in ans[0]:
                    if q in q_lst:
                        q_lst[q].append((ans[1],ans[2]))
                        #q_lst[ans[0]].append((ans[1],ans[2][0].split(':')[-1]))
                    else:
                        q_lst[q]=[(ans[1],ans[2])]

    for a_lst in q_lst.values():
        q_cnt.append(len(a_lst))

    for q in q_bi:
        num_t,num_f=0,0
        pred = []
        labels = []
        for yt,yp in q_lst[q]:
            if yt in yp:num_t+=1
            else:num_f+=1
            pred.append(get_bi_label(q, yp))
            labels.append(get_bi_label(q, yt))
        rep = get_cls_report(pred, labels)
        # q_res_bi[q]= [(num_t/(num_t+num_f),num_t,num_f), rep]
        # change to use macro average of F1-score
        q_res_bi[q] = [(rep['macro avg']['f1-score'],num_t,num_f), rep]

    q_res_clf={}

    for q in q_clf:
        num_t,num_f=0,0
        pred = []
        labels = []
        for yt,yp in q_lst[q]:
            if yt in yp:num_t+=1
            else:num_f+=1
            pred.append(get_multicls_label(q, yp))
            labels.append(get_multicls_label(q, yt))
        rep = get_cls_report(pred, labels)
        q_res_clf[q] = [(rep['macro avg']['f1-score'],num_t,num_f), rep]

    q_res_num={}

    for q in q_num:
        sum_sq,cnt=0,0
        for yt,yp in q_lst[q]:
            if isinstance(yp, list):
                assert (len(yp) == 1)
                yp = yp[0]
            yp_lst=yp.split()
            found = False
            for y in yp_lst:
                y = y.strip(' #,.\n')
                if is_number(y):
                    cnt+=1
                    sum_sq+=(yt-float(y.replace(",", "")))**2
                    found = True
                    break
            if not found:
                try:
                    y = w2n.word_to_num(yp)
                    cnt+=1
                    sum_sq+=(yt-float(y))**2
                except ValueError:
                    cnt += 1
                    sum_sq += yt ** 2  # as if the default prediction is 0
        # the metric is the rooted MSE
        try:
            q_res_num[q]=(np.sqrt(sum_sq/cnt),cnt/len(q_lst[q]), cnt)
        except ZeroDivisionError:
            pass

    other_q = list(set(q_lst.keys()) - set(q_structured))

    # use bleu score for other questions
    bleu_results = {}
    for q in other_q:
        bleu = 0
        this_q = q_lst[q]
        for gt, pred in this_q:
            s = sentence_bleu([gt.split()], pred.split(), weights=(1, 0, 0, 0))
            bleu += s
        avg_bleu = bleu / len(this_q)
        bleu_results[q] = avg_bleu

    print(f'Writing to files {output_file0} for model {model_name}...')
    # binary questions
    res = [model_name] + [q_res_bi[q][0][0] for q in q_bi]
    columns = ['method'] + q_bi
    this_result = pd.DataFrame([res], columns=columns)
    output_file = output_file0 + '_binary.csv'
    write_to_file(this_result, output_file)
    bi_f1 = np.mean([q_res_bi[q][0][0] for q in q_bi])

    # multiclass questions
    res = [model_name] + [q_res_clf[q][0][0] for q in q_clf]
    columns = ['method'] + q_clf
    this_result = pd.DataFrame([res], columns=columns)
    output_file = output_file0 + '_multicls.csv'
    write_to_file(this_result, output_file)
    multi_f1 = np.mean([q_res_clf[q][0][0] for q in q_clf])

    # regression questions
    res = [model_name] + [q_res_num[q][0] for q in q_num]
    columns = ['method'] + q_num
    this_result = pd.DataFrame([res], columns=columns)
    output_file = output_file0 + '_regression.csv'
    write_to_file(this_result, output_file)
    avg_rmse = np.mean([q_res_num[q][0] for q in q_num])

    # regression questions: the propotion of finding a number in the answers
    res = [model_name] + [q_res_num[q][1] for q in q_num]
    columns = ['method'] + q_num
    this_result = pd.DataFrame([res], columns=columns)
    output_file = output_file0 + '_regression_answer_rate.csv'
    write_to_file(this_result, output_file)

    # general questions
    res = [model_name] + [bleu_results[q] for q in other_q]
    columns = ['method'] + other_q
    this_result = pd.DataFrame([res], columns=columns)
    output_file = output_file0 + '_general.csv'
    write_to_file(this_result, output_file)
    mean_bleu = np.mean([bleu_results[q] for q in other_q])

    return bi_f1, multi_f1, avg_rmse, mean_bleu
