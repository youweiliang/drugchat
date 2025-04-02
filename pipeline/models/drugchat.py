import logging
import random
import re

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from pipeline.common.registry import registry
from pipeline.models.utils import disabled_train, Mlp
from pipeline.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from pipeline.models.gnn import GNN, GNN_graphpred
import contextlib
from pipeline.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

from pipeline.models.image_mol import ImageMol
from peft import LoraConfig, get_peft_model, LoraModel


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


@registry.register_model("drugchat")
class DrugChat(BaseModel):
    """
    GNN GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/drugchat.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        gnn_ckpt=None,
        image_mol_ckpt=None,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_image_mol=True,
        freeze_gnn=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        use_graph_agg=True,
        encoder_names=["gnn"],
        feat_dims=None,
        prompt_tuning=0,
        use_mlp=False,
        lora_rank=0,  # this is for setting it up for TRAINING
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.encoder_names = encoder_names

        self.feat_dims = feat_dims
        if "gnn" in self.encoder_names:
            self.use_graph_agg = use_graph_agg
            self.create_gnn(gnn_ckpt, freeze_gnn)
        if "image_mol" in self.encoder_names:
            self.create_image_mol(image_mol_ckpt, freeze_image_mol)

        self.ln_vision = nn.Identity()

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.token_dtype = self.llama_tokenizer(
            'foo',
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_txt_len,
            add_special_tokens=False
        ).input_ids.dtype

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        def get_projector(indim, outdim):
            if use_mlp:
                return Mlp(indim, outdim, outdim)
            else:
                return nn.Linear(indim, outdim)

        if self.feat_dims is not None:
            self.llama_proj = nn.ModuleDict()
            for kk, dim in self.feat_dims.items():
                self.llama_proj.add_module(
                    kk,
                    get_projector(dim, self.llama_model.config.hidden_size)
                    )
                # if 'image_mol' in kk and freeze_image_mol:
                #     for p in self.llama_proj[kk].parameters():
                #         p.requires_grad = False
                #     self.llama_proj[kk].eval()
        else:
            self.llama_proj = get_projector(
                self.encoder_out_dim, self.llama_model.config.hidden_size
            )

        self.lora_rank = lora_rank

        if lora_rank:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        if prompt_tuning:
            self.register_parameter("soft_prompt", nn.Parameter(torch.randn(1, prompt_tuning, self.llama_model.config.hidden_size)))
            nn.init.uniform_(self.soft_prompt, -0.5, 0.5)
        else:
            self.soft_prompt = None
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<compoundHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            # self.prompt_list = ["###Human: <compound><compoundHere></compound>###Human: Can you describe the mechanism of this drug?###Assistant:"]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def llama_embed_tokens(self, *args):
        """
        Without LoRA: llama_model.model.embed_tokens
        With LoRA: llama_model.base_model.model.model.embed_tokens
        """
        if self.lora_rank:
            return self.llama_model.base_model.model.model.embed_tokens(*args)
        return self.llama_model.model.embed_tokens(*args)

    def create_gnn(self, model_path, freeze):
        print('Loading GNN')
        print(f"use_graph_agg={self.use_graph_agg}")

        emb_dim = 300

        self.gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling='attention', gnn_type='gin')
        # self.gnn.from_pretrained('ckpt/gin_contextpred.pth')
        if model_path:
            self.gnn.from_pretrained(model_path)

        self.encoder_out_dim = emb_dim

        if freeze:
            for name, param in self.gnn.named_parameters():
                param.requires_grad = False
            self.gnn = self.gnn.eval()
            # self.gnn.train = disabled_train
            print("freezed GNN")

        pt = None
        if not self.use_graph_agg:
            pt = nn.Parameter(torch.zeros(1, self.gnn.out_dim))
        self.register_parameter("pad_token", pt)
        
        print('Loaded GNN')

    def create_image_mol(self, model_path, freeze):
        model = ImageMol()
        if model_path:
            model.load_from_pretrained(url_or_filename=model_path)
        emb_dim = model.emb_dim
        self.image_mol = model
        self.encoder_out_dim = emb_dim

        if freeze:
            for name, param in self.image_mol.named_parameters():
                param.requires_grad = False
            self.image_mol = self.image_mol.eval()
            # self.image_mol.train = disabled_train
            print("freezed image_mol")
        print('Loaded image_mol')

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        if "gnn" in self.encoder_names:
            self.gnn.to("cpu")
            self.gnn.float()
        if "image_mol" in self.encoder_names:
            self.image_mol.to("cpu")
            self.image_mol.float()

    def encode_img(self, inputs, device, do_proj=True):
        """
        Args:
            inputs (dict)
        """
        if "gnn" in self.encoder_names:
            graph = inputs['graph']
            device = graph.x.device
            if self.low_resource:
                self.vit_to_cpu()
                graph = graph.to("cpu")

            graph_feat = self.gnn(graph).to(device)
            if not self.use_graph_agg:
                graph_feat = self.pad_node(graph, graph_feat)
            if graph_feat.ndim == 2:
                graph_feat = graph_feat.unsqueeze(1)
            feat = graph_feat
            inputs["feat"] = feat
            inputs["graph_feat"] = feat
        if "image_mol" in self.encoder_names:
            image = inputs['image']
            device = image.device
            if self.low_resource:
                self.vit_to_cpu()
                image = image.to("cpu")
            feat = self.image_mol(image).to(device)
            img_feat = feat.unsqueeze(1)

            inputs["feat"] = img_feat
            inputs["image_feat"] = img_feat
        
        if do_proj:
            inputs_llama, atts_llama = self.proj_feat(inputs, device)
            return inputs_llama, atts_llama
        return inputs

    def encode_img_infer(self, inputs, device, autocast=False, autocast_proj=False):
        """
        Need this function to fix the inference data casting issues
        """
        with torch.cuda.amp.autocast(autocast):
            features = self.encode_img(inputs, device, do_proj=False)
        with torch.cuda.amp.autocast(autocast_proj):
            out = self.proj_feat(features, device)
        return out

    def proj_feat(self, features, device):
        """
        Args:
            features (dict)
        """
        if self.feat_dims is not None:
            feats = []
            for kk, dim in self.feat_dims.items():
                feat = features[kk]
                feat = F.dropout(feat, p=0.2, training=self.training)
                inputs_tokens = self.llama_proj[kk](feat)
                feats.append(inputs_tokens)
            img_embeds = torch.cat(feats, dim=1)
            atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(device)
            return img_embeds, atts_img

        embeds = self.ln_vision(features["feat"]).to(device)

        inputs_llama = self.llama_proj(embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama

    def pad_node(self, data, node_representation):
        # pad the repr so that each graph has some number of node repr
        ptr = data.ptr.tolist()
        nodes = [node_representation[ptr[i]: ptr[i+1]] for i in range(data.num_graphs)]
        nnodes = [ptr[i+1] - ptr[i] for i in range(data.num_graphs)]
        max_len = max(nnodes)
        pad_size = [max_len - x_ for x_ in nnodes]
        pad = self.pad_token.to(device=node_representation.device)
        node_repr = torch.stack([torch.cat([node, pad.expand(pz, -1)]) for pz, node in zip(pad_size, nodes)])
        return node_repr

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            batch_size = img_embeds.shape[0]
            ps = [prompt.split('<compoundHere>') for prompt in prompts]
            p_before, p_after = list(zip(*ps))
            p_before_tokens = self.llama_tokenizer(
                p_before, padding="longest", return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, padding="longest", return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_embed_tokens(p_before_tokens.input_ids)#.expand(batch_size, -1, -1)
            p_after_embeds = self.llama_embed_tokens(p_after_tokens.input_ids)#.expand(batch_size, -1, -1)
            if self.soft_prompt is not None:
                img_embeds = torch.cat([img_embeds, self.soft_prompt.expand(batch_size, -1, -1)], 1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples, generate=False, generate_prob=False):
        if "gnn" in self.encoder_names:
            inputs = samples["graph"]
            device = inputs.x.device
        if "image_mol" in self.encoder_names:
            inputs = samples["image"]
            device = inputs.device
        if "feat" in self.encoder_names:
            # no encoder
            device = list(v for v in samples.values() if isinstance(v, torch.Tensor))[0].device

        img_embeds, atts_img = self.encode_img(samples, device)

        assert 'question' in samples
        if 'question' in samples:
            # assert len(samples['question']) == 1, "not supporting batch mode yet"
            vqa_prompt = ['###Human: <compound><compoundHere></compound> ' + qq + "###Assistant: " for qq in samples['question']]
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, [prompt])

        self.llama_tokenizer.padding_side = "right"

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=self.token_dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_embed_tokens(bos)
        img_embeds = img_embeds.to(dtype=bos_embeds.dtype)
        embs = torch.cat([bos_embeds, img_embeds], dim=1)

        out = {}
        if generate:
            # Generate output
            tt = self.gen_(embs, generate_prob)
            out = {'gen': tt}
            if "text_input" not in samples:
                return out

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        weight = None
        if 'weight' in samples:
            # Use sample-wise weight in the teacher-forcing cross entropy loss
            weight = samples['weight'].to(device)
            weight = weight.reshape(batch_size, 1).expand_as(targets)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                weight=weight,
            )
        loss = outputs.loss

        return {"loss": loss} | out

    def gen_(self, embs, prob):
        """
        Generate text.
        """
        stop_words_ids = [torch.tensor([835]).to(embs.device),
                          torch.tensor([2277, 29937]).to(embs.device)]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=300,
            stopping_criteria=stopping_criteria,
            num_beams=1,
            do_sample=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.,
            length_penalty=1.,
            temperature=1,
            return_dict_in_generate=prob,
            output_scores=prob
        )

        output_texts = []
        output_probs = []
        for i in range(embs.shape[0]):
            if prob:
                # Get the probability of generating Yes/No
                output_token = outputs.sequences[i]

                # Get the logits for the first generated token
                logits = outputs.scores[0][i]  # Logits for the next token
                probs = torch.nn.functional.softmax(logits, dim=-1)  # Apply softmax to get probabilities

                # Get token IDs for "Yes" and "No"
                yes_token_id = self.llama_tokenizer("Yes", add_special_tokens=False).input_ids[0]
                no_token_id = self.llama_tokenizer("No", add_special_tokens=False).input_ids[0]

                # Extract probabilities for "Yes" and "No"
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()
                output_prob = (yes_prob, no_prob)
                output_probs.append(output_prob)
            else:
                output_token = outputs[i]
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_texts.append(output_text)
        return output_texts, output_probs

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        encoder_ckpt = cfg.get("encoder_ckpt", None)
        assert encoder_ckpt is None, "encoder_ckpt is no longer used, check image_mol_ckpt/gnn_ckpt"
        image_mol_ckpt = cfg.get("image_mol_ckpt", "ckpt/ImageMol.pth.tar")
        gnn_ckpt = cfg.get("gnn_ckpt", "ckpt/gin_contextpred.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_image_mol = cfg.get("freeze_image_mol", True)
        freeze_gnn = cfg.get("freeze_gnn", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 160)
        end_sym = cfg.get("end_sym", '\n')
        use_graph_agg = cfg.get("use_graph_agg", True)
        encoder_name = cfg.get("encoder_name", "gnn")
        encoder_names = cfg.get("encoder_names", [encoder_name])
        feat_dims = cfg.get("feat_dims", None)  # a dict that controls the name of llama_proj
        prompt_tuning = cfg.get("prompt_tuning", 0)
        use_mlp = cfg.get("use_mlp", False)
        lora_rank = cfg.get("lora_rank", 0)

        model = cls(
            vit_model=vit_model,
            image_mol_ckpt=image_mol_ckpt,
            gnn_ckpt=gnn_ckpt,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_image_mol=freeze_image_mol,
            freeze_gnn=freeze_gnn,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            use_graph_agg=use_graph_agg,
            encoder_names=encoder_names,
            feat_dims=feat_dims,
            prompt_tuning=prompt_tuning,
            use_mlp=use_mlp,
            lora_rank=lora_rank,
        )

        ckpt_path = cfg.get("ckpt", None)  # load weights of DrugChat
        if isinstance(ckpt_path, str):
            ckpt_path = [ckpt_path]
        if ckpt_path:
            # could have multiple ckpt, as in pretraining and finetuning (where different model components are trained)
            for path in ckpt_path:
                ckpt = torch.load(path, map_location="cpu")
                model_dict = ckpt['model']

                if gnn_ckpt != "ckpt/gin_contextpred.pth":
                    # not to overwrite gnn
                    new_dict = {}
                    for name, p in model_dict.items():
                        if 'gnns' in name:
                           continue
                        new_dict[name] = p
                    model_dict = new_dict
                
                if image_mol_ckpt != "ckpt/ImageMol.pth.tar":
                    # not to overwrite image mol
                    new_dict = {}
                    for name, p in model_dict.items():
                        if 'image_mol' in name:
                           continue
                        new_dict[name] = p
                    model_dict = new_dict
                
                if lora_rank:
                    # convert names for lora model
                    # in case the saved model is not lora, but the current model is lora
                    new_dict = {}
                    for name, p in model_dict.items():
                        if name.startswith('llama_model.model'):
                            name = name.replace('llama_model.model', 'llama_model.base_model.model.model')
                        new_dict[name] = p
                    model_dict = new_dict

                msg = model.load_state_dict(model_dict, strict=False)
                print("Loaded checkpoint from {}: {}".format(path, msg))

        return model