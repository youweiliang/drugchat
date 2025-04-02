"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch
import torch.distributed as dist
from pipeline.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from pipeline.common.logger import MetricLogger, SmoothedValue
from pipeline.common.registry import registry
from pipeline.datasets.data_utils import prepare_sample
from pipeline.tasks.eval import eval_generation
from pipeline.models.sentence_similarity import SimilarityModel


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.test_yaml_config = None

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        self.val_iters = cfg.run_cfg.get('val_iters', False)
        self.config = cfg
        return model_cls.from_config(model_config)

    @property
    def generation(self):
        return self.config.run_cfg.get('generation', False)

    @property
    def generate_prob(self):
        return self.config.run_cfg.get('generate_prob', False)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)
            if builder is None:
                builder = registry.get_builder_class('base_dataset')(dataset_config)
            else:
                builder = builder(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def generation_step(self, model, samples, generate_prob):
        gen = model(samples, generate=True, generate_prob=generate_prob)
        return gen

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))
        if "test_yaml_config" in kwargs:
            self.test_yaml_config = kwargs['test_yaml_config']
            self._save_path_ = kwargs['save_path']

    def after_evaluation(
            self, val_result, split_name, epoch, **kwargs,
        ):
        """val_result is a list of dict or loss"""
        if isinstance(val_result[0], (int, float, torch.Tensor)):
            # val_result is a list of loss
            mean_loss = 0
            if val_result:
                mean_loss = sum(val_result) / len(val_result)
                mean_loss = mean_loss.item()
            out = {"agg_metrics": 1000 - mean_loss, "loss": mean_loss}  # convert loss to metric and make it positive
            return out

        results = val_result
        output_dir = kwargs['output_dir']
        results_dir = kwargs['results_dir']
        job_id = kwargs['job_id']
        metrics = eval_generation(results, f"{job_id}_{epoch}", os.path.join(results_dir, "val"))
        with open(os.path.join(results_dir, f"{epoch}.json"), "w") as f:
            json.dump(results, f, indent=2)
        bi_f1, multi_f1, avg_rmse, mean_bleu = metrics
        agg_metrics = bi_f1 + multi_f1 + mean_bleu - avg_rmse * 0.001
        out = {"agg_metrics": agg_metrics, "bi_f1": bi_f1, "multi_f1": multi_f1, "avg_rmse": avg_rmse, "mean_bleu": mean_bleu}
        return out

    def inference_step(self):
        raise NotImplementedError
    
    def inference_(self, file=None):
        """This is mainly for structured QA evaluation."""
        if self.test_yaml_config is None or file is None:
            return
        raise DeprecationWarning("This inference is no longer used.")
        with open(self.test_yaml_config) as f:
            config = yaml.load(f, Loader=Loader)
        config['model']['ckpt'] = self._save_path_
        with open("eval_configs/tmp.yaml", "w") as f:
            yaml.dump(config, f)

        torch.cuda.empty_cache()

        logging.info(f"Evaluating with file {file}")
        os.system(
            f"python inference.py --temperature 1 --cfg-path eval_configs/tmp.yaml --gpu-id 0 --max_new_tokens 50 --in_file {file} --out_file eval_results/ChEMBL_QA_val_tmp.json"
        )

        with open("eval_results/ChEMBL_QA_val_tmp.json") as f:
            js = json.load(f)
        results = sum(js.values(), [])
        return results

    @torch.no_grad()
    def evaluation(self, model, data_loader, cuda_enabled=True, file=None, test=False):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{global_avg:.4f}"))
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 500
        niters = len(data_loader)
        if self.val_iters and not test:
            niters = self.val_iters

        results = []
        res = self.inference_(file=file)
        if res is not None:
            return res
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        logging.info("No eval file found. Evaluating with teacher forcing loss.")
        gens = []
        for i in metric_logger.log_every(range(niters), print_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            if self.generation and test:
                ress = self.generation_step(model=model, samples=samples, generate_prob=self.generate_prob)
                output_texts, output_probs = ress['gen']
                eval_output = ress['loss']

                if not self.generate_prob:
                    for smi, question, gt, out in zip(samples['smiles'], samples['question'], samples["text_input"], output_texts):
                        gens.append([smi, question, gt, out, eval_output.item()])
                else:
                    for smi, question, gt, out, (yes_p, no_p) in zip(samples['smiles'], samples['question'], samples["text_input"], output_texts, output_probs):
                        gens.append([smi, question, gt, out, yes_p, no_p, eval_output.item()])
            else:
                eval_output = self.valid_step(model=model, samples=samples)
                results.append(eval_output)
            metric_logger.update(loss=eval_output.item())
            if i >= niters and not test:
                break

        if is_dist_avail_and_initialized():
            dist.barrier()

        if self.generate_prob and test:
            df = pd.DataFrame(gens, columns=['smiles', 'question', 'gt', 'output', 'yes_p', 'no_p', 'loss'])
            df['logit'] = df['yes_p'] / df['no_p']
            def to_binary(x):
                x = x.lower()
                if 'yes' in x:
                    return 1
                else:
                    return 0
            df['label'] = df['gt'].apply(to_binary)
            scores = []
            for question, group in df.groupby("question"):
                if len(group["label"].unique()) > 1:  # Ensure both classes are present for AUROC
                    auroc = roc_auc_score(group["label"], group["logit"])
                    auprc = average_precision_score(group["label"], group["logit"])
                    delta_ap = auprc - np.mean(group["label"]).item()
                else:
                    auroc = None  # AUROC is undefined when only one class is present
                    auprc = None  # AUPRC might still be computable but depends on class distribution
                    delta_ap = None

                scores.append({"question": question, "AUROC": auroc, "AUPRC": auprc, 'delta_ap': delta_ap})

            scores_df = pd.DataFrame(scores)
            print(scores_df)
            df.to_csv(self.generate_prob)
            scores_df.to_csv(self.generate_prob.replace('.', '_agg.'))
            scores_df = scores_df.drop(columns=['question'])
            print(scores_df.mean(0))
            return
        
        if self.generation and test:
            df = pd.DataFrame(gens, columns=['smiles', 'question', 'gt', 'output', 'loss'])
            print(df)
            df.to_csv(self.generation)
            sent_model = SimilarityModel(device=eval_output.device)
            scores = []
            for gt, pred in zip(df['gt'], df['output']):
                scores.append(sent_model.calculate_similarity(gt, pred))
            scores = pd.DataFrame(scores)
            print(scores.mean(0))
            df = pd.concat([df, scores], axis=1)
            df.to_csv(self.generation)
            return

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        print("use_amp:", use_amp)

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{global_avg:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)
        log_freq = iters_per_epoch // 3 + 1
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file