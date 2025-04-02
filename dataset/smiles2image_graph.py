import argparse
import json
import os
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm.contrib.concurrent import process_map
from itertools import islice, chain
from smiles2graph import smiles2graph


def Smiles2ImgGraph(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    # try:
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    if savePath is not None:
        img.save(savePath)
    return smiles2graph(mol)
    # except:
    #     return None


def process_smi(args):
    smis, save_dir, start_index = args
    i = 0
    out = []
    for smi in smis:
        idx = i + start_index
        save_path = os.path.join(save_dir, f"img_{idx}.png")
        try:
            g = Smiles2ImgGraph(smi, savePath=save_path)
            out.append((1, g))
        except:
            # print(f"smiles: {smi}")
            out.append((0, None))
        i += 1
    return out

def chunked_iterator(iterable, chunk_size):
    """Yield successive chunks from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, chunk_size)):
        yield chunk

def main(smiles_path, save_dir, start_index, n_processes=48):
    with open(smiles_path, "rt") as f:
        js = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    chunk_sz = len(js) // (n_processes * 5)
    print(f"chunk_sz={chunk_sz}")

    smiles_ = [x for x in chunked_iterator(js.keys(), chunk_sz)]
    args = []
    st_idx = 0
    for x in smiles_:
        args.append((x, save_dir, st_idx))
        st_idx += len(x)
    outputs = process_map(process_smi, args, max_workers=n_processes)

    # res = [(x, o) for x, o in zip(smiles_, outputs)]

    out_js = {}
    out_graphs = {}
    idx = 0
    for smi, (success, g) in zip(chain(*smiles_), chain(*outputs)):
        if success:
            out_js[idx] = [smi, js[smi]]
            out_graphs[smi] = {"graph": g}
        else:
            out_js[idx] = []
        idx += 1

    out_js_path = os.path.join(save_dir, "smiles_img_qa.json")
    with open(out_js_path, "wt") as f:
        json.dump(out_js, f)

    outfile = os.path.join(save_dir, 'graph_smi.pkl')
    with open(outfile, "wb") as f:
        pickle.dump(out_graphs, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Converts SMILES dataset to images and graphs")
    parser.add_argument("--smiles_path", default="data/ChEMBL_QA_test.json", type=str, help="path to json file.")
    parser.add_argument("--save_dir", default="data/ChEMBL_QA_test_image/", type=str, help="path to save output.")
    parser.add_argument("--start_index", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_proc", type=int, default=48, help="number of processes to process the data.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.smiles_path, args.save_dir, args.start_index, args.num_proc)
