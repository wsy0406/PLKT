import argparse
import sys

from wandb_train import main
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="bridge2algebra2006")
    parser.add_argument("--model_name", type=str, default="ptkt")
    # parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--num_en", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
                        help="选择运行设备，如 'cpu' 或 'cuda:0'")

    parser.add_argument('--emb_type', type=str, default='gamma', help='general, beta, gamma')
    parser.add_argument('--base_add', type=float, default=1,
                        help='make sure the parameters of Beta embedding is positive')
    parser.add_argument('--min_val', type=float, default=0.05,
                        help='make sure the parameters of Beta embedding is positive')
    parser.add_argument('--max_val', type=float, default=1e9,
                        help='make sure the parameters of Beta embedding is positive')
    parser.add_argument('--gamma', type=float, default=2., help='use to initialize embedding')
    parser.add_argument('--pattern_level', type=int, default=2, help='maximum value of sliding window')
    parser.add_argument('--bias_weight', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--pattern_type', type=str, default='multi_level_bias')
    parser.add_argument('--reg_lambda', type=float, default=0.0)
    parser.add_argument('--pattern_num', type=int, default=100)
    parser.add_argument('--diversity_lambda', type=float, default=0.0)

    parser.add_argument('--max_segment_length', type=int, default=20)
    parser.add_argument('--max_num_segments', type=int, default=99)

    parser.add_argument('--weight_decay', type=float, default=2.e-4)


    # auto PTKT
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=int, default=1000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--anneal_epoch', type=float, default=0)
   
    args = parser.parse_args()

    # params = vars(args)
    params = args
    main(params)
