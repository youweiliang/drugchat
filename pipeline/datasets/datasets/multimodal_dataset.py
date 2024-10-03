import os
import json
import pickle
import torch

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torch_geometric.data import Data, Batch


class MultimodalDataset(Dataset):
    def __init__(self, datapath, use_image=True, use_graph=False, image_size=224, is_train=False) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_graph = use_graph
        jsonpath = os.path.join(datapath, "smiles_img_qa.json")
        print(f"Using {jsonpath=}")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        if is_train:
            self.transforms = transforms.Compose([
                transforms.RandomRotation((0, 180), fill=255),
                transforms.RandomResizedCrop(image_size, (0.5, 1)),
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
        if use_graph:
            with open(os.path.join(datapath, "graph_smi.pkl"), "rb") as f:
                graphs = pickle.load(f)
        self.images = {}
        self.data = []
        self.graphs = {}
        for idx, rec in meta.items():
            if use_image:
                img_file = 'img_{}.png'.format(idx)
                image_path = os.path.join(datapath, img_file)
                image = Image.open(image_path).convert("RGB")
                # img = self.transforms(image)
                self.images[idx] = image
            smi, qa = rec
            if use_graph:
                g = graphs[smi]["graph"]
                graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
                self.graphs[idx] = graph
            qa = [(idx, qa_pair) for qa_pair in qa]
            self.data.extend(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        out = {"question": qa_pair[0], "text_input": str(qa_pair[1])}
        if self.use_image:
            img = self.images[idx]
            img = self.transforms(img)
            out.update({"img": img})
        if self.use_graph:
            out.update({"graph": self.graphs[idx]})
        return out
    
    @staticmethod
    def collater(samples):
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"question": qq, "text_input": aa}
        if "img" in samples[0]:
            imgs = default_collate([x["img"] for x in samples])
            out.update({"image": imgs})
        if "graph" in samples[0]:
            g = Batch.from_data_list([x["graph"] for x in samples])
            out.update({"graph": g})
        return out
