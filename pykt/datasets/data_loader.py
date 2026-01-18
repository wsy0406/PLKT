#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from multiprocessing import Pool, cpu_count
import tqdm
import pickle

# if torch.cuda.is_available():
#     from torch.cuda import FloatTensor, LongTensor
# else:
#     from torch import FloatTensor, LongTensor
from torch import FloatTensor, LongTensor

def right_pad_to_left_pad(seq: torch.Tensor, mask: torch.Tensor):
    valid_seq = seq[mask]
    valid_len = valid_seq.size(0)
    total_len = seq.size(0)

    left_padded_seq = torch.zeros_like(seq)
    left_padded_seq[-valid_len:] = valid_seq

    left_padded_mask = torch.zeros_like(mask, dtype=torch.bool)
    left_padded_mask[-valid_len:] = True

    return left_padded_seq, left_padded_mask

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False, new= None):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.new = new
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")


    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        if self.new != None:
        # return len(self.dori["rseqs"])
            return len(self.dori["new_rseqs"])
        else:
            return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **new_q_seqs (torch.tensor)**: question id sequence of the interacting subsequences
            - **new_c_seqs (torch.tensor)**: knowledge concept id sequence of the interacting subsequences
            - **new_r_seqs (torch.tensor)**: response id sequence of the interacting subsequences
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **pos_qseqs (torch.tensor)**: Target interactions for question id
            - **pos_cseqs (torch.tensor)**: Target interactions for concept id
            - **pos_rseqs (torch.tensor)**: Target interactions for response
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        # dcur = dict()
        # if self.new == None:
        #     mseqs = self.dori["masks"][index]   # seq_len-1
        #
        # new_mseqs = self.dori["new_masks"][index]
        #
        #
        # for key in self.dori:
        #     # if key in ["masks", "smasks"]:
        #     if key in ["masks", "smasks", "new_masks", "new_smasks"]:
        #         continue
        #     if len(self.dori[key]) == 0:    # 如果当前建对应数据为空，则直接存入dcur并跳过后续处理
        #         dcur[key] = self.dori[key]
        #         dcur["shft_"+key] = self.dori[key]
        #         continue
        #     # print(f"key: {key}, len: {len(self.dori[key])}")
        #     if key in ["new_qseqs","new_cseqs","new_rseqs"]:
        #         new_seqs = self.dori[key][index][:-1] * new_mseqs  # 0~seq_len-2时刻（当前时刻）
        #         new_shft_seqs = self.dori[key][index][1:] * new_mseqs  # 1~seq_len-1时刻（下一时刻）
        #         dcur[key] = new_seqs
        #         dcur["shft_" + key] = new_shft_seqs
        #
        #         # 额外添加目标值（pos_key）：shft_seq 中最后一个有效值
        #         valid_indices = (new_mseqs == 1).nonzero(as_tuple=True)[0]
        #         if len(valid_indices) > 0:
        #             last_valid_idx = valid_indices[-1]
        #             dcur["pos_" + key] = new_shft_seqs[last_valid_idx]
        #         else:
        #             dcur["pos_" + key] = torch.tensor(0, device=new_shft_seqs.device)
        #     if self.new == None:
        #     # else:
        #         seqs = self.dori[key][index][:-1] * mseqs   # 0~seq_len-2时刻（当前时刻）
        #         shft_seqs = self.dori[key][index][1:] * mseqs   # 1~seq_len-1时刻（下一时刻）
        #         dcur[key] = seqs
        #         dcur["shft_"+key] = shft_seqs
        #
        #         # 额外添加目标值（pos_key）：shft_seq 中最后一个有效值
        #         valid_indices = (mseqs == 1).nonzero(as_tuple=True)[0]
        #         if len(valid_indices) > 0:
        #             last_valid_idx = valid_indices[-1]
        #             dcur["pos_" + key] = shft_seqs[last_valid_idx]
        #         else:
        #             dcur["pos_" + key] = torch.tensor(0, device=shft_seqs.device)
        # if self.new == None:
        #     dcur["masks"] = mseqs
        #     dcur["smasks"] = self.dori["smasks"][index]
        # dcur["new_masks"] = new_mseqs
        # dcur["new_smasks"] = self.dori["new_smasks"][index]
        # # print("tseqs", dcur["tseqs"])
        # if not self.qtest:
        #     return dcur
        # else:
        #     dqtest = dict()
        #     for key in self.dqtest:
        #         dqtest[key] = self.dqtest[key][index]
        #     return dcur, dqtest


        dcur = dict()
        if self.new == None:
            mseqs = self.dori["masks"][index]  # seq_len-1
            smasks = self.dori["smasks"][index]


        new_mseqs = self.dori["new_masks"][index]
        new_smasks = self.dori["new_smasks"][index]


        for key in self.dori:

            if len(self.dori[key]) == 0:  
                dcur[key] = self.dori[key]
                dcur["shft_" + key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")

            if key in ["new_qseqs", "new_cseqs", "new_rseqs"]:
                old_new_seqs = self.dori[key][index][:-1]
                raw_new_seqs = self.dori[key][index][:-1][new_mseqs]
                new_seqs = torch.zeros_like(old_new_seqs)
                raw_new_seq_len = raw_new_seqs.size(0)
                new_seqs[-raw_new_seq_len:] = raw_new_seqs

                old_new_shft_seqs = self.dori[key][index][1:]
                raw_new_shft_seqs = self.dori[key][index][1:][new_mseqs]  
                new_shft_seqs = torch.zeros_like(old_new_shft_seqs)
                raw_new_shft_seq_len = raw_new_shft_seqs.size(0)
                new_shft_seqs[-raw_new_shft_seq_len:] = raw_new_shft_seqs
                dcur[key] = new_seqs
                dcur["shft_" + key] = new_shft_seqs

                

                
                if key in ["new_qseqs", "new_cseqs"]:
                    device = new_seqs.device
                    valid_mask = torch.zeros_like(new_seqs, dtype=torch.bool, device=device)
                    if raw_new_seq_len > 0:
                        valid_mask[-raw_new_seq_len:] = True

                    valid_mask_shft = torch.zeros_like(new_shft_seqs, dtype=torch.bool, device=new_shft_seqs.device)
                    if raw_new_shft_seq_len > 0:
                        valid_mask_shft[-raw_new_shft_seq_len:] = True

                  
                    new_seqs = new_seqs.long()
                    new_seqs[valid_mask] = new_seqs[valid_mask] + 1

                    new_shft_seqs = new_shft_seqs.long()
                    new_shft_seqs[valid_mask_shft] = new_shft_seqs[valid_mask_shft] + 1

                    dcur[key] = new_seqs
                    dcur["shft_" + key] = new_shft_seqs
                else:
                    dcur[key] = new_seqs
                    dcur["shft_" + key] = new_shft_seqs



            if self.new == None:
                if key in ["qseqs","cseqs","rseqs"]:
                    # else:
                    old_seqs = self.dori[key][index][:-1]
                    raw_seqs = self.dori[key][index][:-1][mseqs]
                    seqs = torch.zeros_like(old_seqs)
                    raw_seq_len = raw_seqs.size(0)
                    seqs[-raw_seq_len:] = raw_seqs

                    old_shft_seq = self.dori[key][index][1:]
                    raw_shft_seqs = self.dori[key][index][1:][mseqs]
                    shft_seqs = torch.zeros_like(old_shft_seq)
                    raw_shft_seq_len = raw_shft_seqs.size(0)
                    shft_seqs[-raw_shft_seq_len:] = raw_shft_seqs

                    
                    dcur[key] = seqs
                    dcur["shft_" + key] = shft_seqs

            if key in ["masks", "smasks", "new_masks", "new_smasks"]:
                if self.new == None:
                    left_pad_mseqs = torch.zeros_like(mseqs, dtype=torch.bool)
                    mseqs_len = mseqs.sum().item()
                    left_pad_mseqs[-mseqs_len:] = True

                    lef_pad_smasks = torch.zeros_like(smasks, dtype=torch.bool)
                    smasks_len = smasks.sum().item()
                    lef_pad_smasks[-smasks_len:] = True

                else:
                    left_pad_new_mseqs = torch.zeros_like(new_mseqs, dtype=torch.bool)
                    new_mseqs_len = new_mseqs.sum().item()
                    left_pad_new_mseqs[-new_mseqs_len:] = True

                    left_pad_new_smasks = torch.zeros_like(new_smasks, dtype=torch.bool)
                    new_smasks_len = new_smasks.sum().item()
                    left_pad_new_smasks[-new_smasks_len:] = True
                # continue
        if self.new == None:
            dcur["masks"] = left_pad_mseqs
            dcur["smasks"] = lef_pad_smasks
        else:
            dcur["new_masks"] = left_pad_new_mseqs
            dcur["new_smasks"] = left_pad_new_smasks

        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

       

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
     
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "new_qseqs": [], "new_cseqs": [], "new_rseqs":[], "new_smasks": []}

       
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]    
        interaction_num = 0
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            
            qseq = [int(x) for x in row["questions"].split(",")] if "questions" in self.input_type else []
            cseq = [int(x) for x in row["concepts"].split(",")] if "concepts" in self.input_type else []
            rseq = [int(x) for x in row["responses"].split(",")]
            smask = [int(x) for x in row["selectmasks"].split(",")]

            dori["qseqs"].append(qseq)
            dori["cseqs"].append(cseq)
            dori["rseqs"].append(rseq)
            dori["smasks"].append(smask)

            if "timestamps" in row:
                dori["tseqs"].append([int(x) for x in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(x) for x in row["usetimes"].split(",")])

            interaction_num += smask.count(1)

          
            max_len = len(qseq)
            valid_len = sum([1 for x in qseq if x != pad_val])
            for l in range(2, valid_len + 1): 
                q_sub = qseq[:l] + [pad_val] * (max_len - l)
                r_sub = rseq[:l] + [pad_val] * (max_len - l)
                c_sub = cseq[:l] + [pad_val] * (max_len - l) if cseq else []

                smask_sub = smask[:l] + [pad_val] * (max_len - l)

                dori["new_qseqs"].append(q_sub)
                dori["new_rseqs"].append(r_sub)
                if c_sub:
                    dori["new_cseqs"].append(c_sub)
                dori["new_smasks"].append(smask_sub)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            if key not in ["rseqs","new_rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        if dori["cseqs"].nelement() > 0:
            dori["masks"] = (dori["cseqs"][:, :-1] != pad_val) & (dori["cseqs"][:, 1:] != pad_val)
        elif dori["qseqs"].nelement() > 0:
            dori["masks"] = (dori["qseqs"][:, :-1] != pad_val) & (dori["qseqs"][:, 1:] != pad_val)

        if dori["new_qseqs"].nelement() > 0:
            dori["new_masks"] = (dori["new_qseqs"][:, :-1] != pad_val) & (dori["new_qseqs"][:, 1:] != pad_val)


        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)    
        dori["new_smasks"] = (dori["new_smasks"][:, 1:] != pad_val)  
        print(f"interaction_num: {interaction_num}")
      
        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori


    def process_user(self,u):
        item_seq = raw_seq_dict[u][:-2]  # leave one out   
        tmp_seq = item_seq[:1]  
        for idx in range(1, len(item_seq)):
            tmp_seq.append(item_seq[idx])  
            sequences.append((u, tmp_seq[- max_len - 1:]))  
        return sequences


    def _sequence_augment_parallel(self,num):
        all_sequences = []
        num_processes = 1  

        with Pool(processes=num_processes) as pool:
            for sequences in tqdm(pool.imap(self.process_user, range(1, num + 1)), total=num, ncols=100, leave=False,
                                  unit="seq", desc=">>> Sequence Augment"):
                all_sequences.extend(sequences)

        return all_sequences
