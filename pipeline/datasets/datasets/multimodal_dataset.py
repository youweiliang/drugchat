import os
import json
import pickle
import torch
import pandas as pd
import random
import yaml

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import Draw
from dataset.smiles2graph import smiles2graph
from transformers import LlamaTokenizer


def Smiles2ImgGraph(smis, size=224):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
    '''
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    g = smiles2graph(mol)
    graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
    return img, graph


class MultimodalDataset(Dataset):
    def __init__(self, datapath, use_image=True, use_graph=False, image_size=224, is_train=False,
                 max_token=150) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_graph = use_graph
        self.max_token = max_token
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        if is_train:
            self.transforms = transforms.Compose([
                transforms.RandomRotation((0, 180), fill=255),
                transforms.RandomResizedCrop(image_size, (0.8, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop(image_size), 
                transforms.ToTensor(),
                normalize,
            ])

        with open('pipeline/configs/models/drugchat.yaml') as f:
            cfg = yaml.safe_load(f)
        llama_model = cfg['model'].get("llama_model")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=True)

        self.init_online(datapath)
        self.init_offline(datapath)

        print("Dataset size:", len(self))

    def init_online(self, datapath):
        # Check if the split info is appended to datapath
        base = os.path.basename(datapath)
        splits = None
        if base.startswith('__split__'):
            # A hack to put the split indices un the datapath
            # For example, if the datapath = /a/b/__split__1,3
            # then we read the files under /a/b/ and extract the split = 1 and split = 3
            datapath = os.path.dirname(datapath)
            splits = base.replace('__split__', '').split(',')
        # QApath = os.path.join(datapath, "QA_2.parquet")
        QApath = os.path.join(datapath, "QA.parquet")
        self.bioassay = False
        if not os.path.exists(QApath):
            QApath = os.path.join(datapath, "QA_2.parquet")
            if not os.path.exists(QApath):
                return
            self.bioassay = True
        self.online_data = True
        print(f"Using {QApath=}")
        self.meta = pd.read_parquet(QApath)
        if splits is not None:
            self.meta['split'] = self.meta['split'].astype(str)
            mask = self.meta['split'].isin(splits)
            self.meta = self.meta[mask]
        smi_path = os.path.join(datapath, "compound_smiles.parquet")
        self.smiles = pd.read_parquet(smi_path)
        assert (self.smiles['compound_idx'] == self.smiles.index).all()
        q_path = os.path.join(datapath, "question_idx.parquet")
        self.questions = pd.read_parquet(q_path)
        assert (self.questions['question_idx'] == self.questions.index).all()
        l_path = os.path.join(datapath, "label2text.parquet")
        self.label2text = pd.read_parquet(l_path)
        assert (self.label2text['label'] == self.label2text.index).all()

    def init_offline(self, datapath):
        jsonpath = os.path.join(datapath, "smiles_img_qa.json")
        if not os.path.exists(jsonpath):
            return
        self.online_data = False
        print(f"Using {jsonpath=}")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)
        if self.use_graph:
            with open(os.path.join(datapath, "graph_smi.pkl"), "rb") as f:
                graphs = pickle.load(f)
        self.images = {}
        self.data = []
        self.graphs = {}
        self.smi = {}
        real_idx = 0
        for idx, rec in meta.items():
            if not rec:
                continue
            if self.use_image:
                img_file = 'img_{}.png'.format(idx)
                image_path = os.path.join(datapath, img_file)
                image = Image.open(image_path).convert("RGB")
                # img = self.transforms(image)
                self.images[real_idx] = image
            smi, qa = rec
            self.smi[real_idx] = smi
            if self.use_graph:
                g = graphs[smi]["graph"]
                graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
                self.graphs[real_idx] = graph
            qa = [(real_idx, qa_pair) for qa_pair in qa]
            self.data.extend(qa)
            real_idx += 1

    def __len__(self):
        if self.online_data:
            return len(self.meta)
        return len(self.data)
    
    def truncate_text(self, question):
        # truncate long questions
        # text with 150 tokens can use a batch size of 24 on 80G GPU
        tokens = self.llama_tokenizer.encode(question, add_special_tokens=False)
        if len(tokens) > self.max_token:
            question_trunc = self.llama_tokenizer.decode(tokens[:self.max_token])
            question = question_trunc + '...'
        return question

    def __getitem__(self, index):
        if self.online_data:
            return self.get_online_item(index)
        return self.get_offline_item(index)

    def get_offline_item(self, index):
        idx, qa_pair = self.data[index]
        out = {"question": qa_pair[0], "text_input": str(qa_pair[1]), "idx": index, 'smiles': self.smi[idx]}
        if self.use_image:
            img = self.images[idx]
            img = self.transforms(img)
            out.update({"img": img})
        if self.use_graph:
            out.update({"graph": self.graphs[idx]})
        return out

    def get_online_item(self, index):
        row = self.meta.iloc[index]
        question = self.questions.iloc[row['question_idx']]['question']

        if self.max_token:
            question = self.truncate_text(question)

        smi = self.smiles.iloc[row['compound_idx']]['smiles']
        label = row['label']
        row2 = self.label2text.iloc[label]
        ans = row2['text']
        if self.bioassay:
            question = 'Is the compound active in this bioassay: ' + question
        out = {"question": question, "text_input": ans, "idx": index, 'smiles': smi}

        if 'weight' in row:
            weight = row['weight']
            out.update({'weight': weight})
        elif 'weight' in row2:
            weight = row2['weight']
            out.update({'weight': weight})

        img, graph = Smiles2ImgGraph(smi)

        if self.use_image:
            img = self.transforms(img)
            out.update({"img": img})
        if self.use_graph:
            out.update({"graph": graph})
        return out
    
    @staticmethod
    def collater(samples):
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        idx = [x['idx'] for x in samples]
        smi = [x['smiles'] for x in samples]

        out = {"question": qq, "text_input": aa, "idx": idx, 'smiles': smi}
        
        if 'weight' in samples[0]:
            weight = torch.Tensor([x['weight'] for x in samples])
            out.update({'weight': weight})

        if "img" in samples[0]:
            imgs = default_collate([x["img"] for x in samples])
            out.update({"image": imgs})
        if "graph" in samples[0]:
            g = Batch.from_data_list([x["graph"] for x in samples])
            out.update({"graph": g})
        return out