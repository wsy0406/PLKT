import os
import argparse
import json
import sys

sys.path.append('../pykt-new')
import torch
from torch.utils.tensorboard import SummaryWriter
# # torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy
from pykt.models import train_model, evaluate, init_model
from pykt.utils import debug_print, set_seed
from pykt.datasets import init_dataset4train, init_test_datasets
import datetime
import time
import wandb


def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": vars(params)}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)



def main(params):

    if "use_wandb" not in params:
        params.use_wandb = 1


    initial_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp = time.strftime("%m%d_%H_%M_%S", time.localtime())


    set_seed(params.seed)
    model_name, dataset_name, fold, emb_type, save_dir = params.model_name, params.dataset_name, \
        params.fold, params.emb_type, params.save_dir

    log_dir = f"./log/{model_name}/{dataset_name}/{initial_time}_tune/{timestamp}"

    debug_print(text="load config files.", fuc_name="main")

    with open("configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]

        if model_name in ["dkvmn", "deep_irt", "sakt", "saint", "saint++", "akt", "folibikt", "atkt", "lpkt", "skvmn",
                          "dimkt", "vkt"]:
            train_config["batch_size"] = 512  ## because of OOM
        if model_name in ["simplekt", "stablekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64  ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16
        if model_name in ["qdkt", "qikt"] and dataset_name in ['algebra2005', 'bridge2algebra2006']:
            train_config["batch_size"] = 32
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32  ## because of OOM

        if model_name in ["ptkt"]:
            train_config["batch_size"] = 512
        if model_name in ["autoptkt"]:
            train_config["batch_size"] = 256
        model_config = vars(copy.deepcopy(params))

        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config[
        "optimizer"]

    with open("./configs/data_config.json") as fin:
        data_config = json.load(fin)



    if 'maxlen' in data_config[dataset_name]:  # prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)

    debug_print(text="init_dataset", fuc_name="main")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size,
                                                            diff_level=diff_level)

    if model_name == 'ptkt':
        # params_str = "_".join([str(v) for k, v in vars(params).items() if not k in ['other_config']])
        params_str = str(params.dataset_name) + '_' + str(params.model_name) + '_' + str(params.seed) + '_' + str(params.fold) + '_' \
                     + str(params.dropout) + '_' + str(params.learning_rate) + '_' + str(params.emb_size) + '_' + str(params.emb_type) \
                     + '_' + str(params.gamma) + '_' + str(params.pattern_type) + '_' + str(params.pattern_level) + '_' \
                     + str(params.bias_weight) + '_' + str(timestamp) + '_' +'subseq_pos-neg_bath'+str(batch_size)+'_' + str(params.max_num_segments) +\
                     '_' + 'left_dedup_diff_r_pattern_id+1_arxiv_mask0_noboth_len80_targetemb_qdiff_new' + '_'\
                     + str(params.weight_decay)
    elif model_name == 'autoptkt':
        params_str = str(params.dataset_name) + '_' + str(params.model_name) + '_' + str(params.seed) + '_' + str(
            params.fold) + '_' \
                     + str(params.dropout) + '_' + str(params.learning_rate) + '_' + str(params.emb_size) + '_' + str(
            params.emb_type) \
                     + '_' + str(params.gamma) + '_' + str(params.pattern_type) + '_' + str(params.pattern_level) +'_' + str(params.bias_weight) + '_' \
                     + str(params.reg_lambda)+ '_' + str(params.diversity_lambda) + '_' + str(params.min_tau) + '_'+ str(params.pattern_num) + '_' + str(timestamp) + '_' \
                     + 'subseq_rshft_pos-neg_bath'+str(batch_size)+'_' + '_' + 'left_neg_dedup_qzero_nnnormal_new'
    else:
        params_dict = vars(params) 
        params_str = "_".join([str(v) for k, v in params_dict.items() if k not in ['other_config']])
        params_str = params_str + str(timestamp) + '_dedup_len80'


    print(f"params: {params}, params_str: {params_str}")
    if params.add_uuid == 1 and params.use_wandb == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str + f"_{str(uuid.uuid4())}"

    ckpt_path = os.path.join(save_dir, params_str)

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(
        f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params.learning_rate
    for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint", "saint++", "sakt", "atdkt", "simplekt", "stablekt", "bakt_time", "folibikt","ptkt","vkt","autoptkt"]:
        model_config["seq_len"] = seq_len
        params.seq_len = seq_len

    debug_print(text="init_model", fuc_name="main")
    # model_config['log_dir'] = log_dir
    params.log_dir = log_dir

    print(f"model_name:{model_name}")


    writer = SummaryWriter(log_dir=log_dir)
    # writer2 = None
    if model_name == 'ptkt':
        writer2 = SummaryWriter(log_dir=f"/home/usr/wrokspace/Knowledge_tracing/runs/{model_name}_{params.dataset_name}_{params.seed}_{params.dropout}_{params.learning_rate}_{params.emb_type}_"
                                        f"{params.gamma}_{params.pattern_type}_{params.pattern_level}_{params.bias_weight}_{timestamp}_"
                                        f"batch{batch_size}_{params.max_num_segments}_{params.weight_decay}"
                                        )
    elif model_name == 'autoptkt':
        writer2 = SummaryWriter(log_dir=f"/home/usr/wrokspace/Knowledge_tracing/runs/{model_name}_{params.seed}_{params.dropout}_{params.learning_rate}_{params.emb_type}_"
                                        f"{params.gamma}_{params.pattern_type}_{params.pattern_level}_{params.bias_weight}_{params.reg_lambda}_{params.diversity_lambda}_{params.min_tau}_{params.pattern_num}_{timestamp}_"
                                        f"batch{batch_size}"
                                        )
    else:
        writer2 = SummaryWriter(log_dir=f"/home/usr/wrokspace/Knowledge_tracing/runs/{model_name}_{params.dropout}_{learning_rate}_{timestamp}")



    if model_name in  ["ptkt", "autoptkt"]:
        model = init_model(model_name, params, data_config[dataset_name], emb_type, params.device, writer)
    else:
        model = init_model(model_name, model_config, data_config[dataset_name], emb_type, params.device, writer)


    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        print(f"dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            # opt = Adam(model.parameters(), learning_rate, weight_decay=params.weight_decay)
            opt = Adam(model.parameters(), learning_rate, weight_decay=2.e-4)
            # opt = Adam(model.parameters(), learning_rate)

    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    # save_model = False
    debug_print(text="train model", fuc_name="main")


    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
            train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model,
                        data_config[dataset_name], fold,writer=writer2)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model,
                                                                                                       train_loader,
                                                                                                       valid_loader,
                                                                                                       num_epochs, opt,
                                                                                                       ckpt_path, params.device, params, None,
                                                                                                       None, save_model,writer=writer2)

    if save_model:
        # best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type, writer)
        if model_name in ['ptkt', "autoptkt"]:
            best_model = init_model(model_name, params, data_config[dataset_name], emb_type, params.device, writer)
        else:
            best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type, params.device, writer)
        net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(
        round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(
        validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    model_save_path = os.path.join(ckpt_path, emb_type + "_model.ckpt")
    print(f"end:{datetime.datetime.now()}")

    if params.use_wandb == 1:
        wandb.log({
            "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch, "model_save_path": model_save_path})

    writer.close()