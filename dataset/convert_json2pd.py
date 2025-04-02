import os
import json
import pandas as pd
import argparse
from collections import OrderedDict


parser = argparse.ArgumentParser(description="Converts json dataset to pandas dataset (parquet)")
parser.add_argument("--input_path", default="data/ChEMBL_QA_test.json", type=str, help="path to json file.")
parser.add_argument("--input_paths", nargs='+', default=None, help="Convert and combine multiple json files (for each split below).")
parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='split names to use')
parser.add_argument("--save_dir", type=str, help="path to save output.")
args = parser.parse_args()

with open(args.input_path) as f:
    js = json.load(f)

compounds2idx = OrderedDict()
questions2idx = OrderedDict()
label2idx = OrderedDict()


def convert(file, split=None):
    with open(file) as f:
        js = json.load(f)
    data = []
    for smi, qas in js.items():
        if isinstance(qas, dict):
            # For json file with the cv fold information
            split = qas['cv_fold']
            qas = qas['QAs']
        if smi not in compounds2idx:
            cidx = len(compounds2idx)
            compounds2idx[smi] = cidx
        else:
            cidx = compounds2idx[smi]
        for q, a in qas:
            if q not in questions2idx:
                qidx = len(questions2idx)
                questions2idx[q] = qidx
            else:
                qidx = questions2idx[q]
            if qidx is None:
                continue
            if a not in label2idx:
                aidx = len(label2idx)
                label2idx[a] = aidx
            else:
                aidx = label2idx[a]
            if split is not None:
                data.append([cidx, qidx, aidx, split])
            else:
                data.append([cidx, qidx, aidx])

    return data


def to_pd(X2idx, column_names, filename):
    df = pd.DataFrame(X2idx.items(), columns=column_names)
    assert (df[column_names[-1]] == df.index).all()
    df.to_parquet(os.path.join(args.save_dir, filename))
    print(df)
    return df


all_activity = []

os.makedirs(args.save_dir, exist_ok=True)

if args.input_paths is not None:
    for split, file in zip(args.splits, args.input_paths):
        # if split == 'val':
        #     split = 'valid'
        data = convert(file, split)
        all_activity.extend(data)
else:
    data = convert(args.input_path)
    all_activity.extend(data)

if len(all_activity[0]) == 4:
    df = pd.DataFrame(all_activity, columns=['compound_idx', 'question_idx', 'label', 'split'])
else:
    df = pd.DataFrame(all_activity, columns=['compound_idx', 'question_idx', 'label'])
df.to_parquet(os.path.join(args.save_dir, 'QA.parquet'))
print(df)


to_pd(compounds2idx, column_names=['smiles', 'compound_idx'], filename="compound_smiles.parquet")
to_pd(questions2idx, column_names=['question', 'question_idx'], filename="question_idx.parquet")
to_pd(label2idx, column_names=['text', 'label'], filename="label2text.parquet")