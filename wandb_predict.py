import os
import argparse
import json
import copy
import sys

sys.path.append('../pykt-new')

import torch
import pandas as pd

from pykt.datasets import init_test_datasets
from pykt.models.evaluate_model import evaluate, evaluate_question, evaluate_doa_multi_sample
from pykt.models import train_model, init_model
from pykt.models.init_model import load_model
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



import ast
import pandas as pd
import numpy as np


def main(params):
      params_str = "user/dataset_name_plkt_xxx"
    save_dir, fusion_type = params.save_dir, params.fusion_type.split(",")
    batch_size = 256

    ckpt_path = os.path.join(save_dir, params_str)

    with open(os.path.join(ckpt_path, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], \
        trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt", "vkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len

    with open("./configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]

    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(
            data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(
            data_config, model_name, batch_size, diff_level=diff_level)

    print(
        f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    if model_name in ['ptkt','autoptkt']:
        model = load_model(model_name, params, data_config, emb_type, ckpt_path, device)
    else:
        model = load_model(model_name, model_config, data_config, emb_type, ckpt_path, device)

    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))


    use_cuda = str(device).startswith("cuda")

    num_test_batches = len(test_loader)
    num_test_samples = len(test_loader.dataset) if hasattr(test_loader, "dataset") else None

    if use_cuda:
        torch.cuda.synchronize()
    test_start_time = time.perf_counter()

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    elif model.model_name in ['ptkt', 'autoptkt']:
        testauc, testacc = evaluate(model.to(device), test_loader, model_name, device, new=1)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, device)

    if use_cuda:
        torch.cuda.synchronize()
    test_end_time = time.perf_counter()

    test_infer_time = test_end_time - test_start_time
    avg_batch_infer_time = test_infer_time / num_test_batches

    print(f"testauc: {testauc}, testacc: {testacc}")
    print(f"test inference time: {test_infer_time:.6f} s")
    print(f"avg batch inference time: {avg_batch_infer_time:.6f} s")


    doa_res = None

    if getattr(params, "use_doa", 0) == 1:
        if params.q_matrix_path == "":
            raise ValueError("开启 DOA 评估时，必须提供 --q_matrix_path")

        q_matrix = np.load(params.q_matrix_path)

        doa_seeds = tuple(int(x) for x in params.doa_seeds.split(","))

        use_new_for_doa = model.model_name in ["ptkt", "autoptkt"]

        if use_cuda:
            torch.cuda.synchronize()
        doa_start_time = time.perf_counter()

        doa_res = evaluate_doa_multi_sample(
            model=model,
            test_loader=test_loader,
            model_name=model_name,
            device=device,
            q_matrix=q_matrix,
            max_students=params.doa_num_students,
            seeds=doa_seeds,
            use_new=use_new_for_doa,
        )

        if use_cuda:
            torch.cuda.synchronize()
        doa_end_time = time.perf_counter()
        doa_eval_time = doa_end_time - doa_start_time

        print(f"DOA mean: {doa_res['doa_mean']:.6f}")
        print(f"DOA std: {doa_res['doa_std']:.6f}")
        print(f"DOA list: {doa_res['doa_list']}")
        print(f"DOA evaluation time: {doa_eval_time:.6f} s")

    if num_test_samples is not None:
        avg_sample_infer_time = test_infer_time / num_test_samples
        throughput = num_test_samples / test_infer_time
        print(f"num test samples: {num_test_samples}")
        print(f"avg sample inference time: {avg_sample_infer_time * 1000:.6f} ms")
        print(f"throughput: {throughput:.2f} samples/s")

        dres = {
        "testauc": testauc,
        "testacc": testacc,
        "test_infer_time_sec": test_infer_time,
        "avg_batch_infer_time_sec": avg_batch_infer_time,
    }

    if num_test_samples is not None:
        dres["num_test_samples"] = num_test_samples
        dres["avg_sample_infer_time_ms"] = avg_sample_infer_time * 1000
        dres["throughput_samples_per_sec"] = throughput

    if doa_res is not None:
        dres["doa_mean"] = doa_res["doa_mean"]
        dres["doa_std"] = doa_res["doa_std"]
        dres["doa_list"] = doa_res["doa_list"]
        dres["doa_num_students"] = params.doa_num_students
        dres["doa_seeds"] = params.doa_seeds
  

    raw_config = json.load(open(os.path.join(ckpt_path, "config.json")))
    dres.update(raw_config['params'])

    save_result_path = os.path.join(ckpt_path, f"{model.emb_type}_test_result_with_doa.json")
    with open(save_result_path, "w", encoding="utf-8") as f:
        json.dump(dres, f, indent=4, ensure_ascii=False)

    print(f"Saved test result to: {save_result_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--dataset_name", type=str, default="bridge2algebra2006")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument('--log_dir', type=str, default="./log/test")
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument('--gamma', type=float, default=2., help='use to initialize embedding')
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_en", type=int, default=1)
    parser.add_argument('--bias_weight', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--pattern_type', type=str, default='multi_level_bias')
    parser.add_argument('--reg_lambda', type=float, default=0.0)
    parser.add_argument('--pattern_num', type=int, default=100)
    parser.add_argument('--diversity_lambda', type=float, default=0.0)
    parser.add_argument('--pattern_level', type=int, default=2, help='maximum value of sliding window')
    parser.add_argument('--emb_type', type=str, default='beta', help='general, beta, gamma')

    parser.add_argument('--max_segment_length', type=int, default=20)
    parser.add_argument('--max_num_segments', type=int, default=99)

    parser.add_argument('--weight_decay', type=float, default=2.e-4)

    parser.add_argument('--attention_enhancement', type=bool, default=True)
    parser.add_argument('--attention_weight', type=float, default=0.5)

    parser.add_argument("--use_doa", type=int, default=1)
    parser.add_argument("--q_matrix_path",type=str,default="/home/user/data/bridge2algebra2006/q_matrix.npy")
    parser.add_argument("--doa_num_students", type=int, default=1000)
    parser.add_argument("--doa_seeds", type=str, default="2026,2027,2028")



    args = parser.parse_args()
    print(args)
    params = args
    main(params)
