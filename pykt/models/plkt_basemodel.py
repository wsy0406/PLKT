import os
import logging
import sys
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_logger
import json

class Regularizer:
    def __init__(self, base_add, min_val, max_val) -> None:
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, item_embedding):
        return torch.clamp(item_embedding + self.base_add, self.min_val,
                           self.max_val) 


# A ∧ B
class BetaIntersection(nn.Module):
    def __init__(self, args) -> None:
        super(BetaIntersection, self).__init__()
        self.args = args
        self.emb_dim = args.emb_size
        self.layer1 = nn.Linear(2 * self.emb_dim, 2 * self.emb_dim)
        self.layer2 = nn.Linear(2 * self.emb_dim, self.emb_dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)


    def forward(self, alpha, beta, mask):
        """
        Args:
            alpha : ..., W, E   (W is window size, E is embedding dim)
            beta : ..., W, E
            mask : ..., W
        """
        all_embeddings = torch.cat([alpha, beta], dim=-1)  # ..., W, 2E
        mask = torch.where(mask, 0., -10000.)

        layer1_act = F.relu(self.layer1(all_embeddings))  # ..., W, 2E
        attention_input = self.layer2(layer1_act)
        attention_input = attention_input + mask.unsqueeze(-1)
        attention = F.softmax(attention_input, dim=-2)  # ..., W, E
        alpha = torch.sum(alpha * attention, dim=-2)  # ..., E
        beta = torch.sum(beta * attention, dim=-2)  # ..., E

        return alpha, beta



class AttnGRUIntersection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_dim = args.emb_size
        self.softplus = nn.Softplus()
        self.eps = 1e-4  

        # Attention Layer
        self.attn_layer1 = nn.Linear(2 * self.emb_dim, 2 * self.emb_dim)
        self.attn_layer2 = nn.Linear(2 * self.emb_dim, 1)

        # GRUs for alpha and beta separately
        self.alpha_gru = nn.GRU(input_size=self.emb_dim, hidden_size=self.emb_dim, batch_first=True)
        self.beta_gru = nn.GRU(input_size=self.emb_dim, hidden_size=self.emb_dim, batch_first=True)

        nn.init.xavier_uniform_(self.attn_layer1.weight)
        nn.init.xavier_uniform_(self.attn_layer2.weight)

    def forward(self, alpha, beta, mask):
        """
        alpha, beta: (B, W, E)
        mask: (B, W)
        """

        B, X, W, E = alpha.shape
        alpha = alpha.reshape(B * X, W, E)
        beta = beta.reshape(B * X, W, E)
        mask = mask.reshape(B * X, W)

        x = torch.cat([alpha, beta], dim=-1)  # (B*X, W, 2E)

        # Attention weights
        attn_score = F.relu(self.attn_layer1(x))  # (B, W, 2E)
        attn_score = self.attn_layer2(attn_score).squeeze(-1)  # (B*X, W)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1).unsqueeze(-1)  # (B*X, W, 1)

        # Weighted input
        # weighted_x = x * attn_weight  # (B*X, W, 2E)
        alpha_weighted = alpha * attn_weight    # (B*X, W, 2E)
        beta_weighted = beta * attn_weight      # (B*X, W, 2E)

        # GRU
        _, alpha_h_n = self.alpha_gru(alpha_weighted)  # (1, B*X, E)
        _, beta_h_n = self.beta_gru(beta_weighted)      # (1, B*X, E)

        alpha_out = self.softplus(alpha_h_n.squeeze(0)).reshape(B, X, E) + self.eps
        beta_out = self.softplus(beta_h_n.squeeze(0)).reshape(B, X, E) + self.eps

        return alpha_out, beta_out

class BetaProjection(nn.Module):
    def __init__(self, args, projection_regularizer) -> None:
        super(BetaProjection, self).__init__()
        self.args = args
        dim = args.dim
        self.layer1 = nn.Linear(args.max_len * dim * 2, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)
        self.layer0 = nn.Linear(dim, dim)

        for nl in range(args.num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f'layer{nl}').weight)

        self.projection_regularizer = projection_regularizer

    def forward(self, seq_alpha, seq_beta, mask):
        """
        Args:
            seq_alpha : (B, S, E)
            seq_beta : (B, S, E)
            mask : (B, S)
        """
        x = torch.cat((seq_alpha, seq_beta), dim=-1)  # (B, S, 2 * E)
        x = x * mask.unsqueeze(-1)  # (B, S, 2 * E)
        for nl in range(1, self.args.num_layers + 1):
            x = F.relu(getattr(self, f'layer{nl}')(x))
        x = self.projection_regularizer(self.layer0(x))
        return x


class BetaNegation(nn.Module):
    def __init__(self) -> None:
        super(BetaNegation, self).__init__()

    def forward(self, embedding):
        embedding = 1. / embedding
        return embedding


# A ∧ B
class GammaIntersection(nn.Module):
    def __init__(self, args) -> None:
        super(GammaIntersection, self).__init__()
        dim = args.emb_size
        self.layer_alpha1 = nn.Linear(dim * 2, dim)
        self.layer_alpha2 = nn.Linear(dim, dim)
        self.layer_beta1 = nn.Linear(dim * 2, dim)
        self.layer_beta2 = nn.Linear(dim, dim)

    def forward(self, alpha_emb, beta_emb, mask):
        """
            alpha_emb : [B, N, S, E]
            mask : [B, N, S]
        """
        all_emb = torch.cat((alpha_emb, beta_emb), dim=-1)  # [B, N, S, 2E]

        mask = torch.where(mask > 0., 1., -10000.)
        layer1_alpha = F.relu(self.layer_alpha1(all_emb))
        # attention1 = self.layer_alpha2(layer1_alpha) * mask.unsqueeze(-1)  # [B, N, S, E]
        attention1 = self.layer_alpha2(layer1_alpha)  # [B, N, S, E]
        attention1 = attention1 * mask.unsqueeze(-1)  # [B, N, S, E]
        attention1 = F.softmax(attention1, dim=-2)  # [B, N, S, E]

        layer1_beta = F.relu(self.layer_beta1(all_emb))
        # attention2 = self.layer_beta2(layer1_beta) * mask.unsqueeze(-1)  # [B, N, S, E]
        attention2 = self.layer_beta2(layer1_beta)
        attention2 = attention2 * mask.unsqueeze(-1)  # [B, N, S, E]
        attention2 = F.softmax(attention2, dim=-2)  # [B, N, S, E]

        alpha = torch.sum(alpha_emb * attention1, dim=-2)  # [B, N, E]
        beta = torch.sum(beta_emb * attention2, dim=-2)  # [B, N, E]

        return alpha, beta


class GammaUnion(nn.Module):
    def __init__(self, args) -> None:
        super(GammaUnion, self).__init__()
        # dim = args.emb_dim
        dim = args.emb_size
        self.layer_alpha1 = nn.Linear(dim * 2, dim)
        self.layer_alpha2 = nn.Linear(dim, dim // 2)
        self.layer_alpha3 = nn.Linear(dim // 2, dim)

        self.layer_beta1 = nn.Linear(dim * 2, dim)
        self.layer_beta2 = nn.Linear(dim, dim // 2)
        self.layer_beta3 = nn.Linear(dim // 2, dim)

        self.dropout = nn.Dropout(p=0.5)
        self.regularizer = Regularizer(1, 0.15, 1e9)

    def forward(self, ptn_alpha, ptn_beta, mask):
        """
            ptn_alpha: [B, N, E]
            ptn_beta: [B, N, E]
            mask: [B, N]
        """
        padding_mask = torch.where(mask, 0., -10000.)  # [B, N]
        all_emb = torch.cat((ptn_alpha, ptn_beta), dim=-1)  # [B, N, 2E]

        l1_alpha = F.relu(self.layer_alpha1(all_emb))  # [B, N, E]
        l2_alpha = F.relu(self.layer_alpha2(l1_alpha))  # [B, N, E // 2]
        l3_alpha = F.relu(self.layer_alpha3(l2_alpha))  # [B, N, E]
        attn_alpha = F.softmax(l3_alpha + padding_mask.unsqueeze(-1), dim=-2)  # [B, N, E]

        l1_beta = F.relu(self.layer_beta1(all_emb))  # [B, N, E]
        l2_beta = F.relu(self.layer_beta2(l1_beta))  # [B, N, E // 2]
        l3_beta = F.relu(self.layer_beta3(l2_beta))  # [B, N, E]
        attn_beta = F.softmax(l3_beta + padding_mask.unsqueeze(-1), dim=-2)  # [B, N, E]

        epsilon = 1e-8
        k = ptn_alpha * attn_alpha  # [B, N, E]
        o = 1 / (ptn_beta * attn_beta + epsilon)  # [B, N, E]
        k_sum = torch.pow(torch.sum(k * o, dim=-2), 2) / (torch.sum(torch.pow(o, 2) * k, dim=-2) + epsilon)  # [B, E]
        o_sum = torch.sum(k * o, dim=-2) / (k_sum * o.shape[-2] + epsilon)  # [B, E]
        # Welch–Satterthwaite equation

        alpha_emb = k_sum  # [B, E]
        beta_emb = o_sum  # [B, E]
        alpha_emb = self.regularizer(alpha_emb)
        beta_emb = self.regularizer(beta_emb)
        return alpha_emb, beta_emb


class GammaProjection(nn.Module):
    def __init__(self, args) -> None:
        super(GammaProjection, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.emb_dim

        self.layer_alpha0 = nn.Linear(self.emb_dim, self.hidden_dim)
        self.layer_alpha1 = nn.Linear(self.hidden_dim, self.emb_dim)

        self.layer_beta0 = nn.Linear(self.emb_dim, self.hidden_dim)
        self.layer_beta1 = nn.Linear(self.hidden_dim, self.emb_dim)

        self.projection_regularizer = Regularizer(1, 0.15, 1e9)

    def forward(self, alpha, beta):
        """
            alpha: [B, S, E]
            beta: [B, S, E]
        """
        all_alpha = alpha  # [B, S, E]
        all_beta = beta  # [B, S, E]

        all_alpha = F.tanh(self.layer_alpha0(all_alpha))
        all_alpha = self.layer_alpha1(all_alpha)
        all_alpha = self.projection_regularizer(all_alpha)  # [B, S, E]

        all_beta = F.tanh(self.layer_beta0(all_beta))
        all_beta = self.layer_beta1(all_beta)
        all_beta = self.projection_regularizer(all_beta)  # [B, S, E]

        return all_alpha, all_beta

class GammaNegation(nn.Module):
    def __init__(self) -> None:
        super(GammaNegation, self).__init__()

    def forward(self, alpha):
        neg_alha = 1. / alpha
        return neg_alha


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

class BaseModel(nn.Module):
    # def __init__(self, args, emb_type, item_num, n_pid) -> None:
    def __init__(self, args, emb_type, q_num, item_num, device) -> None:
        super(BaseModel, self).__init__()
    # def __init__(self, args, emb_type, item_num, device) -> None:
    #     super(BaseModel, self).__init__()

        self.args = args
        # self.logger = get_logger(self.args.log_dir)
        # self.log_args(args)
        self.item_num = item_num
        # self.n_pid = n_pid
        print("---------------------------------item_num is :", item_num)

        self.device = device

        self.recorded_both_weights = [] 

        with open(f"/home/user/workspace/Knowledge_tracing/data/{args.dataset_name}/item_difficulty_new_mapped.json", "r") as f:
            difficulty_dict = json.load(f)  

        max_qid = max(map(int, difficulty_dict.keys()))
        difficulty_list = [0.0] * (max_qid + 1)
        for qid, diff in difficulty_dict.items():
            difficulty_list[int(qid)] = float(diff)

        self.difficulty_table = nn.Embedding.from_pretrained(
            torch.tensor(difficulty_list).float().unsqueeze(1).to(device),  # (num_q, 1)
            freeze=True 
        )

        self.difficulty_linear = nn.Linear(1, self.args.emb_size)
        self.pos_difficulty_linear = nn.Linear(1, self.args.emb_size)



        emb_type = emb_type.lower()
        if emb_type in ['beta', 'gamma']:

            self.item_embedding = nn.Parameter(torch.zeros(2*item_num+1, self.args.emb_size * 2))  # (alpha, beta)
            self.ques_embedding = nn.Parameter(torch.zeros(2*q_num+1, self.args.emb_size * 2))
            self.target_item_embedding = nn.Parameter(torch.zeros(2 * item_num + 1, self.args.emb_size * 2))  # (alpha, beta)
            self.target_ques_embedding = nn.Parameter(torch.zeros(2 * q_num + 1, self.args.emb_size * 2))


            self.position_emb = nn.Parameter(torch.zeros(args.seq_len, args.emb_size * 2))
            self.gamma = nn.Parameter(torch.tensor([args.gamma]), requires_grad=False)
            self.response_embedding = nn.Parameter(torch.zeros(2, self.args.emb_size * 2))

            self.gated_attention = GatedAttention(self.args.emb_size, self.device).to(self.device)

            if emb_type == 'beta':
                self.regularizer = Regularizer(1, 0.05, 1e9)
                self.embedding_range = nn.Parameter(torch.tensor([self.gamma.item() / args.emb_size]), requires_grad=False)
                self.intersection = BetaIntersection(args)
                # self.qcinteg = Beta_qcInteg(args)
                # self.intersection = AttnGRUIntersection(args)
                self.negative = BetaNegation()
                nn.init.uniform_(tensor=self.item_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
                nn.init.uniform_(tensor=self.position_emb, a=-self.embedding_range.item(), b=self.embedding_range.item())
                nn.init.uniform_(tensor=self.ques_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())



            elif emb_type == 'gamma':
                self.epsilon = 2.0  # gamma embedding
                self.regularizer = Regularizer(1, 0.15, 1e9)
                self.embedding_range = nn.Parameter(
                    torch.tensor([(self.gamma.item() + self.epsilon) / self.args.emb_size]), requires_grad=False)
                self.intersection = GammaIntersection(self.args)
                self.negative = GammaNegation()
                self.union = GammaUnion(args)
                # self.union = GammaUnion(self.args)
                nn.init.uniform_(tensor=self.item_embedding, a=-3. * self.embedding_range.item(),
                                 b=3. * self.embedding_range.item())
                nn.init.uniform_(tensor=self.position_emb, a=-3. * self.embedding_range.item(),
                                 b=3. * self.embedding_range.item())
                nn.init.uniform_(tensor=self.ques_embedding, a=-3. * self.embedding_range.item(),
                                 b=3. * self.embedding_range.item())
    def load_pro2skill_matrix(self, pro2skill_path, num_q, num_c):
        pro2skill = torch.zeros(num_q, num_c)

        with open(pro2skill_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if len(row) >= 2:
                    ques_id = int(row[0])
                    skill_id = int(row[1])
                    if ques_id < num_q and skill_id < num_c:
                        pro2skill[ques_id, skill_id] = 1

        return pro2skill

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_hh_l0)
                nn.init.xavier_uniform_(m.weight_ih_l0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

   
    def log_args(self, args):
        self.logger.info("Model Arguments:")
        for k, v in vars(args).items():
            self.logger.info(f"{k}: {v}")

  
    def vec_to_distribution(self, alpha, beta):
        assert torch.min(alpha) >= 1e-5 and torch.min(
            beta) >= 1e-5, "Error: alpha or beta is too small, min value is [{}]".format(torch.min(alpha))
        if self.args.emb_type == 'beta':
            return torch.distributions.beta.Beta(alpha, beta)
        elif self.args.emb_type == 'gamma':
            return torch.distributions.gamma.Gamma(alpha, beta)
        else:
            raise ValueError("Error embedding type => {}".format(self.args.emb_type.lower()))

    def cal_distance(self, dist1, dist2):
        return self.gamma - torch.norm(torch.distributions.kl.kl_divergence(dist1, dist2), p=1, dim=-1)

    def cal_raw_distance(self, dist1, dist2):
        return torch.norm(torch.distributions.kl.kl_divergence(dist1, dist2), p=1, dim=-1)


    def get_embedding(self, indices):
        """beta or gamma"""
        emb = self.item_embedding[indices.long()]  # (..., 2E)
        # emb = self.item_embedding(indices)  # (..., 2E)
        emb = self.regularizer(emb)
        alpha, beta = torch.chunk(emb, 2, dim=-1)
        return alpha, beta

    def get_target_embedding(self, indices):
        """beta or gamma"""
        emb = self.target_item_embedding[indices.long()]  # (..., 2E)
        # emb = self.item_embedding(indices)  # (..., 2E)
        emb = self.regularizer(emb)
        alpha, beta = torch.chunk(emb, 2, dim=-1)
        return alpha, beta

    def get_ques_embedding(self, indices):
        emb = self.ques_embedding[indices.long()]
        emb = self.regularizer(emb)
        alpha, beta = torch.chunk(emb, 2, dim=-1)
        return alpha, beta

    def get_target_ques_embedding(self, indices):
        emb = self.target_ques_embedding[indices.long()]
        emb = self.regularizer(emb)
        alpha, beta = torch.chunk(emb, 2, dim=-1)
        return alpha, beta


    def get_difficulty_emb(self, qid_tensor):
        """
        qid_tensor: (B, S) 或 (B, 1)
        difficulty_emb: (B, S, E)
        """
        diff_scalar = self.difficulty_table(qid_tensor)  # (B, S, 1)
        diff_emb = self.difficulty_linear(diff_scalar)  # (B, S, E)
        return diff_emb

   
    def get_position_embedding(self, position):
        """
            position: [1, S]
        """
        emb = self.position_emb[position.long()]  # [1, S, 2E]
        emb = self.regularizer(emb)
        alpha, beta = torch.chunk(emb, 2, dim=-1)
        return alpha, beta


    def get_pattern_index(self, seq, r, window_size):

        B, S = seq.shape

        def extract_window(tensor):
            t = tensor.unsqueeze(1).unsqueeze(1).float()  # [1, 1, B, S]
            unfold = nn.Unfold(kernel_size=(1, window_size), stride=(1, 1))
            t_unfolded = unfold(t).transpose(-2, -1).reshape(B, -1, window_size)  # [B, X, W]
            return t_unfolded

        sub_seq = extract_window(seq)
        sub_r = extract_window(r.squeeze(-1))
        mask = sub_seq != 0

        return sub_seq.long(), sub_r.long(), mask

   

    def adaptive_boundary_detection(self, seq, r, detector):

        B, S = seq.shape

        seq_alpha, seq_beta = self.get_embedding(seq)  # (B, S, E)
        seq_rep = (seq_alpha + seq_beta) / 2.0

        mask = (seq != 0).long()  # (B, S)
        boundaries = detector(seq_rep, mask)  


        all_sub_seq = []
        all_sub_r = []
        all_mask = []

        for b in range(B):
            cur_seq = seq[b]  # [S]
            cur_r = r[b].squeeze() if r.dim() == 3 else r[b]  # [S]
            cur_boundaries = boundaries[b]  # [S]


            non_zero_indices = torch.nonzero(cur_seq != 0, as_tuple=False)
            if len(non_zero_indices) == 0:

                start_index = 0
                end_index = S - 1
            else:
                start_index = non_zero_indices[0].item()
                end_index = non_zero_indices[-1].item()

      
            valid_boundaries = cur_boundaries[start_index:end_index + 1]

           
            boundary_indices = torch.where(valid_boundaries == 1)[0].cpu().numpy()

            boundary_indices = [idx + start_index for idx in boundary_indices]


            if not boundary_indices or boundary_indices[-1] != end_index:
                boundary_indices.append(end_index)

            if not boundary_indices or boundary_indices[0] != start_index:
                boundary_indices.insert(0, start_index)

            segments_seq = []
            segments_r = []
            segments_mask = []


            for i in range(len(boundary_indices) - 1):
                start = boundary_indices[i]
                end = boundary_indices[i + 1]
                segment_length = end - start

     
                if segment_length == 0:
                    continue

                seg_seq = cur_seq[start:end]
                seg_r = cur_r[start:end]

 
                if segment_length > self.args.max_segment_length:
                    seg_seq = seg_seq[:self.args.max_segment_length]
                    seg_r = seg_r[:self.args.max_segment_length]
                    seg_mask = torch.ones(self.args.max_segment_length, dtype=torch.bool, device=seq.device)

                else:
                    pad_len = self.args.max_segment_length - segment_length
                    seg_seq = F.pad(seg_seq, (pad_len, 0), value=0)
                    seg_r = F.pad(seg_r, (pad_len, 0), value=0)
                    seg_mask = F.pad(
                        torch.ones(segment_length, dtype=torch.bool, device=seq.device),
                        (pad_len, 0), value=0
                    )

                segments_seq.append(seg_seq)
                segments_r.append(seg_r)
                segments_mask.append(seg_mask)


            num_segments = len(segments_seq)
            if num_segments > self.args.max_num_segments:
                segments_seq = segments_seq[:self.args.max_num_segments]
                segments_r = segments_r[:self.args.max_num_segments]
                segments_mask = segments_mask[:self.args.max_num_segments]
            elif num_segments < self.args.max_num_segments:
                pad_num = self.args.max_num_segments - num_segments
                for _ in range(pad_num):
                    segments_seq.append(torch.zeros(self.args.max_segment_length, dtype=torch.long, device=seq.device))
                    segments_r.append(torch.zeros(self.args.max_segment_length, dtype=torch.float, device=seq.device))
                    segments_mask.append(torch.zeros(self.args.max_segment_length, dtype=torch.bool, device=seq.device))


            all_sub_seq.append(torch.stack(segments_seq))
            all_sub_r.append(torch.stack(segments_r))
            all_mask.append(torch.stack(segments_mask))

 
        sub_seq = torch.stack(all_sub_seq)
        sub_r = torch.stack(all_sub_r)
        mask = torch.stack(all_mask)

        return sub_seq, sub_r, mask


    def cal_pattern_target_distance(self, ptn_alpha, ptn_beta, t_alpha, t_beta):

        """
            Args:
                ptn_alpha : B, N, E
                ptn_beta : B, N, E
                t_alpha : B, Y, E
                t_beta : B, Y, E
        """
        ptn_alpha = ptn_alpha.unsqueeze(1).repeat(1, t_alpha.shape[1], 1, 1)  # (B, Y, N, E)
        ptn_beta = ptn_beta.unsqueeze(1).repeat(1, t_alpha.shape[1], 1, 1)  # (B, Y, N, E)
        t_alpha = t_alpha.unsqueeze(2)  # (B, Y, 1, E)
        t_beta = t_beta.unsqueeze(2)  # (B, Y, 1, E)


        ptn_dist = self.vec_to_distribution(ptn_alpha, ptn_beta)
        t_dist = self.vec_to_distribution(t_alpha, t_beta)
        raw_dis = self.cal_raw_distance(t_dist, ptn_dist)  

        gamma_dis = self.gamma - raw_dis
        return raw_dis, gamma_dis

    
    # def forward(self, qseq, seq, rseq, qrseq, crseq, mask):
    #     raise NotImplementedError()
    def forward(self, qseq, seq, rseq, qrseq, crseq):
        raise NotImplementedError()
    # def forward(self, cq, cc, cr, qrseq, mask):
    #     raise NotImplementedError()
    # def forward(self, seq, rseq, crseq, mask):
    #     raise NotImplementedError()

    def predict(self, seq, pos, neg):
        raise NotImplementedError()

    def calculate_loss(self, seq, pos, neg):
        raise NotImplementedError()

    def temperature_schedule(self, epoch, max_epoch, start_temp=1.0, end_temp=0.01):
        """
        Args:
            epoch: current epoch
            max_epoch: total epoch
            start_temp: initial temperature
            end_temp: final temperature
        """
        if epoch >= max_epoch:
            return end_temp
        temp = start_temp - (start_temp - end_temp) * (epoch / max_epoch)
        return temp


