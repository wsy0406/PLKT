import sys

import torch

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones
from .ptkt_basemodel import *
import torch.nn as nn
import torch.nn.functional as F
import math


class EnhancePatternWeightNet(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out, hidden_dim=256):
        super(EnhancePatternWeightNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )

        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, seq, seq_alpha, seq_beta, pattern_mask):
        """
            seq: (B, S)
            seq_alpha: (B, S, E)
            seq_beta: (B, S, E)
            pattern_mask: (B, X)
        """

        B, S, E = seq_alpha.shape

        mask = seq != 0  # (B, S)
        seq_emb = seq_alpha / (seq_alpha + seq_beta)  # (B, S, E)
        seq_emb = seq_emb * mask.unsqueeze(-1)  # (B, S, E)

        features = seq_emb.reshape(B, -1)
        features = self.feature_extractor(features)     # (B, E//2)

        features = features.unsqueeze(1)    #(B, 1, E//2)
        attended_features, _ = self.attention(features, features, features)

        outputs = self.pattern_classifier(attended_features.squeeze(1))     # (B, E)

        weight_mask = torch.where(pattern_mask, 0., -10000.)  # (B, X)
        outputs = outputs + weight_mask
        outputs = F.softmax(outputs, dim=-1)  # (B, X)

       
        return outputs



class HNetBoundaryDetector(nn.Module):
    def __init__(self, d_input: int, d_proj: int):
        """
        Args:
            d_input: 输入向量维度 (d)
            d_proj: 投影向量维度 (d')
        """
        super().__init__()
        self.W_q = nn.Linear(d_input, d_proj, bias=False)  
        self.W_k = nn.Linear(d_input, d_proj, bias=False) 

    def forward(self, x, mask=None):
        B, S, D = x.shape   # (B, seq_len, E)

 
        if S == 1:
            return torch.ones(B, 1, dtype=torch.long, device=x.device)

        q = self.W_q(x)
        k = self.W_k(x)

  
        k_shift = torch.cat([
            torch.zeros(B, 1, k.size(-1), device=x.device),  
            k[:, :-1, :] 
        ], dim=1)


        cos_sim = F.cosine_similarity(q, k_shift, dim=-1)


        p = 0.5 * (1 + cos_sim)

        p[:, 0] = 1.0

        b = (p >= 0.5).long()

        if mask is not None:
            b = b * mask

            first_valid_indices = mask.argmax(dim=1)  # (B,)
            for i in range(B):
                idx = first_valid_indices[i]
                b[i, idx] = 1  

        return b

class GatedAttention(nn.Module):
    def __init__(self, hidden_dim, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        )
        self.device = device

    def forward(self, e, e_f):
        """
        e: (B, S, 2*E) 
        e_f: (B, 1, 2*E) 
        return:
            new_e: (B, S, 2*E)
        """
        B, L, D = e.size()
        e = e.to(self.device)   # (B, S, 2*E)
        e_f = e_f.to(self.device)     # (B, 1, 2*E)
        e_f_expanded = e_f.expand(-1, L, -1)  # (B, S, 2*E)

        concat_input = torch.cat([e, e_f_expanded], dim=-1).to(self.device)  # (B, S, 2E*2)

        gate = torch.sigmoid(self.mlp(concat_input)).to(self.device)  # (B, L, 2 * E)

        new_e = e * gate  # (B, L, 2*E)
        return new_e

class PTKT(BaseModel):
    def __init__(self, num_q, num_c, args, writer, device, emb_type="gamma", emb_path="",
                 pretrain_dim=768):
        super(PTKT, self).__init__(args, emb_type, num_q, num_c, device)
        self.model_name = "ptkt"
        self.emb_type = emb_type
        self.writer = writer
        self.num_c = num_c
        self.num_q = num_q
        self.seq_len = args.seq_len
        self.emb_size = args.emb_size
        self.num_attn_heads = args.num_attn_heads
        self.dropout = args.dropout
        self.num_en = args.num_en

        self.bias_weight = args.bias_weight  
        self.temp = args.temp  

        self.get_pattern_num()

        self.device = device

        self.score_linear = nn.Linear(3, 1)


        self.diff_weight = nn.Parameter(torch.tensor(0.1))

      


        for i in range(1, self.args.pattern_level + 1): 
            cur_pattern_num = (self.args.seq_len-1) - i + 1  
            cur_input_dim = self.args.emb_size * (self.args.seq_len-1)  

            cur_weight_net = EnhancePatternWeightNet(cur_input_dim, cur_pattern_num, self.dropout, self.args.emb_size)
            setattr(self, f'weight_net_level{i}', cur_weight_net)


        self.blocks = get_clones(Blocks(self.emb_size, self.num_attn_heads, self.dropout), self.num_en)

        self.dropout_layer = Dropout(self.dropout)
        self.pred = Linear(self.emb_size, 1)

        self.reset_parameters()

        self.l2 = 1e-5

        self.isolation_gate_net = nn.Sequential(
            nn.Linear(self.args.emb_size * 2, self.args.emb_size),
            nn.ReLU(),
            nn.Linear(self.args.emb_size, 1)
        )

    def base_emb(self, q, r, qry):
        x = q + self.num_c * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def enhanced_attention(self, seq_alpha, seq_beta, seq_q, r):
        """
        Args:
            seq_alpha: (B, S, E) 
            seq_beta: (B, S, E) 
            seq_q: (B, S) 
            r: (B, S)
        Returns:
            attended_alpha: (B, S, E)
            attended_beta: (B, S, E) 
            attention_weights: (B, S, 1) 
        """
        B, S, E = seq_alpha.shape


        seq_combined = torch.cat([seq_alpha, seq_beta], dim=-1)  # (B, S, 2E)
        seq_importance = self.importance_net(seq_combined)  # (B, S, 1)


        r_weight = r.unsqueeze(-1).float()  # (B, S, 1)
        dynamic_weight = seq_importance * (1 + r_weight * self.attention_weight)

        attended_alpha = seq_alpha * dynamic_weight
        attended_beta = seq_beta * dynamic_weight
        weight_sum = dynamic_weight.sum(dim=1, keepdim=True) + 1e-8
        normalized_weights = dynamic_weight / weight_sum

        return attended_alpha, attended_beta, normalized_weights

   
    def distance_to_weight(self, pattern_target_sim, mask):
        """
        pattern_target_sim: (B, Y, X) -- similarity
        mask: (B, X) boolean (True: valid)
        """
        # attention mask: valid True -> 0; invalid False -> -inf 
        attention_mask = torch.where(mask, 0.0, torch.tensor(-1e9, device=mask.device)).unsqueeze(1)  # (B,1,X)
        sim_with_mask = pattern_target_sim + attention_mask  # (B, Y, X)
        weight = F.softmax(sim_with_mask, dim=-1)  # (B, Y, X)

        all_false_mask = ~mask.any(dim=-1, keepdim=True)  # (B,1)
        all_false_mask = all_false_mask.unsqueeze(1)  # (B,1,1)
        weight = torch.where(all_false_mask, torch.zeros_like(weight), weight)
        return weight

  

    def get_pattern_num(self, level=None):
        seq_len = self.args.seq_len
        lev_num = self.args.pattern_level

        if level is None:
            # 所有层总 pattern 数
            pattern_num = int(lev_num * (2 * seq_len - lev_num + 1) * 0.5)
            print(f"Pattern type = [sliding], pattern_num = [{pattern_num}]")
            logging.info(f"Pattern type = [sliding], pattern_num = [{pattern_num}]")
            return pattern_num
        else:
            pattern_num_level = seq_len - level + 1
            print(f"Pattern type = [sliding], level = {level}, pattern_num = [{pattern_num_level}]")
            logging.info(f"Pattern type = [sliding], level = {level}, pattern_num = [{pattern_num_level}]")
            return pattern_num_level


    def hnet_bias(self, seq, r, pos):
        pos_alpha, pos_beta = self.get_embedding(pos)  # (B, 1, E)

        seq_alpha, seq_beta = self.get_embedding(seq)  # (B, S, E)
        neg_alpha, neg_beta = self.negative(pos_alpha), self.negative(pos_beta)

        r = r.unsqueeze(-1).float()  # (B, S, 1)
        neg_mask = 1.0 - r
        seq_alpha = seq_alpha * r + self.negative(seq_alpha) * neg_mask
        seq_beta = seq_beta * r + self.negative(seq_beta) * neg_mask

        pattern, pattern_r, mask = self.adaptive_boundary_detection(seq, r.squeeze(-1), self.boundary_detector)  # (B, X, W)

        pattern_r = pattern_r.float().unsqueeze(-1)  # (B, X, W, 1)
        neg_mask = 1.0 - pattern_r

        pattern_alpha, pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
        pattern_alpha = pattern_alpha * pattern_r + self.negative(pattern_alpha) * neg_mask
        pattern_beta = pattern_beta * pattern_r + self.negative(pattern_beta) * neg_mask

        pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta, mask)  # (B, X, E)

        _, pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha, pos_beta)  # (B, 1, X)
        _, pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha, neg_beta)  # (B, Y, X)

        pattern_mask = torch.min(mask, dim=-1)[0]  # (B, X)

        cur_pos_weight = self.distance_to_weight(pattern_pos_dis, pattern_mask)  # (B, 1, X)
        cur_neg_weight = self.distance_to_weight(pattern_neg_dis, pattern_mask)  # (B, Y, X)

        both_weight = self.weight_net_boundary(seq, seq_alpha, seq_beta, pattern_mask)  # (B, X)

        pos_weight = cur_pos_weight + both_weight.unsqueeze(1) * self.bias_weight  # (B, 1, X)
        neg_weight = cur_neg_weight + both_weight.unsqueeze(1) * self.bias_weight  # (B, Y, X)

        pos_score = torch.sum(pos_weight * pattern_pos_dis, dim=-1)  # (B, 1)
        neg_score = torch.sum(neg_weight * pattern_neg_dis, dim=-1)  # (B, Y)

        return pos_score, neg_score, (pos_weight, pattern_pos_dis)

    def multi_level_bias(self, seq_q, seq, r, pos_q, pos, all_mask):
        """
        :param seq_q: (B, S)
        :param seq: (B, S)
        :param r: (B, S)
        :param pos_q: (B, 1)
        :param pos: (B, 1)
        """
        history_diff_emb = self.get_difficulty_emb(seq_q)  # (B, S, E)

        new_seq = seq + self.num_c * r
        new_seq_q = seq_q + self.num_q * r

        skill_seq_alpha, skill_seq_beta = self.get_embedding(new_seq)  # (B, S, E)
        que_seq_alpha, que_seq_beta = self.get_ques_embedding(new_seq_q)     # (B, S, E)

        que_seq_alpha = (que_seq_alpha + history_diff_emb).clamp(min=1e-5)  # (B, S, E)
        que_seq_beta = (que_seq_beta + history_diff_emb).clamp(min=1e-5)  # (B, S, E)
    

        seq_alpha = torch.cat((skill_seq_alpha.unsqueeze(2), que_seq_alpha.unsqueeze(2)), dim=-2)   # (B, S, 2, E)
        seq_beta = torch.cat((skill_seq_beta.unsqueeze(2), que_seq_beta.unsqueeze(2)), dim=-2)      # (B, S, 2, E)
        seq_alpha, seq_beta = self.intersection(seq_alpha, seq_beta, all_mask.unsqueeze(-1))  # ( B, S, E)
     


        pos_pos = pos + self.num_c
        pos_neg = pos
        skill_pos_alpha, skill_pos_beta = self.get_target_embedding(pos_pos)  # (B, 1, E)
        skill_neg_alpha, skill_neg_beta = self.get_target_embedding(pos_neg)  # (B, 1, E)

        pos_pos_q = pos_q + self.num_q
        pos_neg_q = pos_q
        ques_pos_alpha, ques_pos_beta = self.get_target_ques_embedding(pos_pos_q)
        ques_neg_alpha, ques_neg_beta = self.get_target_ques_embedding(pos_neg_q)


        pos_diff_emb = self.get_difficulty_emb(pos_q)  # (B, 1, E)
        ques_pos_alpha = (ques_pos_alpha + pos_diff_emb).clamp(min=1e-5)
        ques_pos_beta = (ques_pos_beta + pos_diff_emb).clamp(min=1e-5)
        ques_neg_alpha = (ques_neg_alpha + pos_diff_emb).clamp(min=1e-5)
        ques_neg_beta = (ques_neg_beta + pos_diff_emb).clamp(min=1e-5)



        pos_alpha = torch.cat((skill_pos_alpha.unsqueeze(2), ques_pos_alpha.unsqueeze(2)), dim=-2)
        pos_beta = torch.cat((skill_pos_beta.unsqueeze(2), ques_pos_beta.unsqueeze(2)), dim=-2)
        neg_alpha = torch.cat((skill_neg_alpha.unsqueeze(2), ques_neg_alpha.unsqueeze(2)), dim=-2)
        neg_beta = torch.cat((skill_neg_beta.unsqueeze(2), ques_neg_beta.unsqueeze(2)), dim=-2)

        pos_neg_mask = torch.ones_like(pos).bool()     # (B, 1)

        pos_alpha, pos_beta = self.intersection(pos_alpha, pos_beta, pos_neg_mask.unsqueeze(-1))  # (B, 1, E)
        neg_alpha, neg_beta = self.intersection(neg_alpha, neg_beta, pos_neg_mask.unsqueeze(-1))



        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        pos_weight_lst = []
        neg_weight_lst = []
        both_weight_lst = []


        for i in range(1, self.args.pattern_level + 1):
            pattern, pattern_r, mask = self.get_pattern_index(new_seq, r, window_size=i)  # (B, X, W)
            q_pattern, q_pattern_r, q_mask_q = self.get_pattern_index(new_seq_q, r, window_size=i)  # (B, X, W)
            old_q_pattern, _, _ = self.get_pattern_index(seq_q, r, window_size=i)
            pattern_diff_emb = self.get_difficulty_emb(old_q_pattern)

            pattern_mask_lst.append(torch.min(mask, dim=-1)[0])  # (B, X)
            skill_pattern_alpha_raw, skill_pattern_beta_raw = self.get_embedding(pattern)  # (B, X, W, E)
            q_pattern_alpha_raw, q_pattern_beta_raw = self.get_ques_embedding(q_pattern)    # (B, X, W, E)

            q_pattern_alpha_raw = (q_pattern_alpha_raw + pattern_diff_emb).clamp(min=1e-5)
            q_pattern_beta_raw = (q_pattern_beta_raw + pattern_diff_emb).clamp(min=1e-5)


            pattern_alpha_raw = torch.cat((skill_pattern_alpha_raw, q_pattern_alpha_raw), dim=-2)
            pattern_beta_raw = torch.cat((skill_pattern_beta_raw, q_pattern_beta_raw), dim=-2)
            mask = torch.cat((mask, q_mask_q), dim=-1)
            pattern_alpha, pattern_beta = self.intersection(pattern_alpha_raw, pattern_beta_raw, mask)  # (B, X, E)
    
            _, pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha,
                                                                  pos_beta)  # (B, 1, X)
            _, pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha,
                                                                  neg_beta)  # (B, 1, X)

            cur_pos_weight = self.distance_to_weight(pattern_pos_dis, pattern_mask_lst[-1])  # (B, 1, X)
            cur_neg_weight = self.distance_to_weight(pattern_neg_dis, pattern_mask_lst[-1])  # (B, 1, X)


            cur_both_weight = getattr(self, f'weight_net_level{i}')(new_seq, seq_alpha, seq_beta,
                                                                    pattern_mask_lst[-1])  # (B, X)

           

            pos_weight_lst.append(cur_pos_weight)
            neg_weight_lst.append(cur_neg_weight)
            both_weight_lst.append(cur_both_weight)
            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)
    
        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, 1, N)
        pos_weight = torch.cat(pos_weight_lst, dim=-1)  # (B, 1, N)
        neg_weight = torch.cat(neg_weight_lst, dim=-1)  # (B, 1, N)
        both_weight = torch.cat(both_weight_lst, dim=-1)  # (B, N)


        pos_weight = pos_weight + both_weight.unsqueeze(1) * self.bias_weight
        neg_weight = neg_weight + both_weight.unsqueeze(1) * self.bias_weight  # (B, 1, N)

        pos_score = torch.sum(pos_weight * pos_dis, dim=-1)  # (B, 1)
        neg_score = torch.sum(neg_weight * neg_dis, dim=-1)  # (B, 1)

     
        return pos_score, neg_score, (pos_weight, pos_dis)

   


    def topkdist(self, dist: torch.Tensor, k = 20):
        values, indices = torch.topk(dist, dim=-1, largest=False, k=k)
        masked_dist = torch.ones_like(dist) * 1e8
        masked_dist = torch.scatter(masked_dist, dim=-1, index=indices, src=values)
        return masked_dist

   
    def single_level_bias(self, seq_q, seq, r, pos_q, pos, all_mask):
        history_diff_emb = self.get_difficulty_emb(seq_q)  # (B, S, E)

        new_seq = seq + self.num_c * r
        new_seq_q = seq_q + self.num_q * r


        skill_seq_alpha, skill_seq_beta = self.get_embedding(new_seq)  # (B, S, E)
        que_seq_alpha, que_seq_beta = self.get_ques_embedding(new_seq_q)
       
        que_seq_alpha = (que_seq_alpha + history_diff_emb).clamp(min=1e-9)  # (B, S, E)
        que_seq_beta = (que_seq_beta + history_diff_emb).clamp(min=1e-9)  # (B, S, E)
        seq_alpha = torch.cat((skill_seq_alpha.unsqueeze(2), que_seq_alpha.unsqueeze(2)), dim=-2)   # (B, S, 2, E)
        seq_beta = torch.cat((skill_seq_beta.unsqueeze(2), que_seq_beta.unsqueeze(2)), dim=-2)      # (B, S ,2, E)

       
        seq_alpha, seq_beta = self.intersection(seq_alpha, seq_beta, all_mask.unsqueeze(-1))    # (B, S, E)
       
        pos_pos = pos + self.num_c
        pos_neg = pos
        skill_pos_alpha, skill_pos_beta = self.get_target_embedding(pos_pos)   # (B, 1, E)
        skill_neg_alpha, skill_neg_beta = self.get_target_embedding(pos_neg)   # (B, 1, E)



        pos_pos_q = pos_q + self.num_q
        pos_neg_q = pos_q

        ques_pos_alpha, ques_pos_beta = self.get_target_ques_embedding(pos_pos_q)
        ques_neg_alpha, ques_neg_beta = self.get_target_ques_embedding(pos_neg_q)

      
        pos_diff_emb = self.get_difficulty_emb(pos_q)  # (B, 1, E)
        ques_pos_alpha = (ques_pos_alpha + pos_diff_emb).clamp(min=1e-9)
        ques_pos_beta = (ques_pos_beta + pos_diff_emb).clamp(min=1e-9)
        ques_neg_alpha = (ques_neg_alpha + pos_diff_emb).clamp(min=1e-9)
        ques_neg_beta = (ques_neg_beta + pos_diff_emb).clamp(min=1e-9)

        pos_alpha = torch.cat((skill_pos_alpha.unsqueeze(2), ques_pos_alpha.unsqueeze(2)), dim=-2)
        pos_beta = torch.cat((skill_pos_beta.unsqueeze(2), ques_pos_beta.unsqueeze(2)), dim=-2)
        neg_alpha = torch.cat((skill_neg_alpha.unsqueeze(2), ques_neg_alpha.unsqueeze(2)), dim=-2)
        neg_beta = torch.cat((skill_neg_beta.unsqueeze(2), ques_neg_beta.unsqueeze(2)), dim=-2)

        pos_neg_mask = torch.ones_like(pos).bool()

        pos_alpha, pos_beta = self.intersection(pos_alpha, pos_beta, pos_neg_mask.unsqueeze(-1))
        neg_alpha, neg_beta = self.intersection(neg_alpha, neg_beta, pos_neg_mask.unsqueeze(-1))


        pos_pattern_dis_lst = []
        neg_pattern_dis_lst = []
        pattern_mask_lst = []
        pos_weight_lst = []
        neg_weight_lst = []
        both_weight_lst = []


        for i in range(self.args.pattern_level, self.args.pattern_level + 1):
            pattern, pattern_r, mask = self.get_pattern_index(new_seq, r, window_size=i)    # (B, x, w)
            q_pattern, q_pattern_r, q_mask_q = self.get_pattern_index(new_seq_q, r, window_size=i)
            old_q_pattern, _, _ = self.get_pattern_index(seq_q, r, window_size=i)
            pattern_diff_emb = self.get_difficulty_emb(old_q_pattern)

            pattern_mask_lst.append(torch.min(mask, dim=-1)[0])  # (B, X)
            skill_pattern_alpha, skill_pattern_beta = self.get_embedding(pattern)  # (B, X, W, E)
            q_pattern_alpha, q_pattern_beta = self.get_ques_embedding(q_pattern) # (B, X, W, E)
            q_pattern_alpha = (q_pattern_alpha + pattern_diff_emb).clamp(min=1e-9)
            q_pattern_beta = (q_pattern_beta + pattern_diff_emb).clamp(min=1e-9)
            pattern_alpha = torch.cat((skill_pattern_alpha, q_pattern_alpha), dim=-2)
            pattern_beta = torch.cat((skill_pattern_beta, q_pattern_beta), dim=-2)
            mask = torch.cat((mask, q_mask_q), dim=-1)

            pattern_alpha, pattern_beta = self.intersection(pattern_alpha, pattern_beta,
                                                            mask)  # (B, X, E)  AND

            _, pattern_pos_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, pos_alpha,
                                                                  pos_beta)  # (B, 1, X)

            _, pattern_neg_dis = self.cal_pattern_target_distance(pattern_alpha, pattern_beta, neg_alpha,
                                                                  neg_beta)  # (B, 1, X)



            cur_pos_weight = self.distance_to_weight(pattern_pos_dis, pattern_mask_lst[-1])  # (B, 1, X)
            cur_neg_weight = self.distance_to_weight(pattern_neg_dis, pattern_mask_lst[-1])  # (B, 1, X)

            cur_both_weight = getattr(self, f'weight_net_level{i}')(new_seq, seq_alpha, seq_beta,
                                                                    pattern_mask_lst[-1])  # (B, X)



            pos_weight_lst.append(cur_pos_weight)
            neg_weight_lst.append(cur_neg_weight)
            both_weight_lst.append(cur_both_weight)
            pos_pattern_dis_lst.append(pattern_pos_dis)
            neg_pattern_dis_lst.append(pattern_neg_dis)

        pos_dis = torch.cat(pos_pattern_dis_lst, dim=-1)  # (B, 1, N)
        neg_dis = torch.cat(neg_pattern_dis_lst, dim=-1)  # (B, 1, N)
        pos_weight = torch.cat(pos_weight_lst, dim=-1)  # (B, 1, N)
        neg_weight = torch.cat(neg_weight_lst, dim=-1)  # (B, 1, N)
        both_weight = torch.cat(both_weight_lst, dim=-1)  # (B, N)


        # # TODO:
        pos_weight = pos_weight + both_weight.unsqueeze(1) * self.bias_weight
        # # pos_weight = pos_weight / pos_weight.sum(dim=-1, keepdim=True)
        neg_weight = neg_weight + both_weight.unsqueeze(1) * self.bias_weight  # (B, 1, N)
        pos_score = torch.sum(pos_weight * pos_dis, dim=-1)  # (B, 1)

        neg_score = torch.sum(neg_weight * neg_dis, dim=-1)  # (B, 1)

        return pos_score, neg_score



    def augment_sequence(self, seq, r, mask_prob=0.2):
        """
        简单的序列增强：将部分题目 mask 掉，或将答题对错flip
        :param seq: (B, S)
        :param r: (B, S)
        :return: 增强后的 (seq_aug, r_aug)
        """
        seq_aug = seq.clone()
        r_aug = r.clone()

        mask = torch.rand_like(seq.float()) < mask_prob
        seq_aug[mask] = 0  

        flip = torch.rand_like(r.float()) < (mask_prob / 2)
        r_aug[flip] = 1 - r_aug[flip]

        return seq_aug, r_aug

    def nt_xent_loss(self, z1, z2, temperature=0.5):
        """
        z1, z2: (B, D) 
        return: contrastive loss
        """
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)  # (2B, D)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )  # (2B, 2B)

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        sim_i_j = torch.exp(similarity_matrix / temperature)
        positives = torch.exp(
            F.cosine_similarity(z1, z2, dim=-1) / temperature
        )
        positives = torch.cat([positives, positives], dim=0)

        loss = -torch.log(positives / sim_i_j.sum(dim=-1))
        return loss.mean()

    def compute_attention_weights(self, ptn_combined, t_combined):
        """
        Args:
            ptn_combined : B, N, 2*E
            t_combined : B, N, 2*E
        Returns:
            attn_weights : B, N, Y
        """
        similarity = torch.einsum('bne,bye->bny', [ptn_combined, t_combined])  # B, N, Y
        attn_weights = F.softmax(similarity, dim=-1) 
        return attn_weights

    def compute_weighted_emb(self, attn_weights, t_combined):
        """
        Args:
            attn_weights : B, N, Y
            t_combined : B, Y, 2*E
        Returns:
            weighted_emb : B, N, 2*E
        """
        weighted_emb = torch.einsum('bny,bye->bne', [attn_weights, t_combined])  # B, N, E
        return weighted_emb

    def knowledge_regularization(self, seq_alpha, seq_beta, r):
        alpha_diff = seq_alpha[:, 1:] - seq_alpha[:, :-1]       # (B, S-1, E)
        beta_diff = seq_beta[:, 1:] - seq_beta[:, :-1]      #(B, S-1, E)

        growth_factor = 0.1
        decay_factor = -0.05
        r_diff = r[:,1:].unsqueeze(-1).float()      # (B, S-1, 1)
        # expected_change = r_diff * 0.1
        expected_change = r_diff * growth_factor + (1 - r_diff) * decay_factor  # (B, S-1, 1)

        smoothness_loss = torch.mean((alpha_diff - expected_change)**2)+\
                            torch.mean((beta_diff - expected_change)**2)
        return smoothness_loss

    def forward(self, q, c, r, qry, cry, mask=None, qtest=False):

        last_qry = qry[:,-1].unsqueeze(-1)
        last_cry = cry[:,-1].unsqueeze(-1)

        if self.args.pattern_type == 'multi_level_bias':
            pos_score, neg_score, pos_weight = self.multi_level_bias(q, c, r, last_qry, last_cry, mask)
            y_pred = torch.sigmoid(pos_score - neg_score)
        elif self.args.pattern_type == 'single_level_bias':
            pos_score, neg_score = self.single_level_bias(q, c, r, last_qry, last_cry, mask)  # (B, 1) (B, 1)
            y_pred = torch.sigmoid(pos_score - neg_score)
        elif self.args.pattern_type == 'hnet_bias':
            pos_score, neg_score, pos_weight = self.hnet_bias(q, r, last_qry)
            y_pred = torch.sigmoid(pos_score - neg_score)   # (B,1)
        else:
            raise ValueError("Invalid pattern type: {}".format(self.args.pattern_type))


        return y_pred

class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb




