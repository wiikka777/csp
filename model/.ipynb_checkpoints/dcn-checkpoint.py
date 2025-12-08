import torch
import torch.nn as nn
from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron

class My_DeepCrossNetworkModel(nn.Module):
    """
    原始 DCN 模型
    """
    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        return self.linear(torch.cat([x_l1, h_l2], dim=1)).squeeze(1)


class My_DeepCrossNetworkModel_withCommentsRanking(nn.Module):
    """
    (修正版)
    支持 CPU embedding 字典传入, 但在初始化时一次性将 Tensors 移至 GPU。
    在 forward 中执行快速、安全的 GPU-native 索引。
    
    text_embeddings = {
        "video_emb_tensor_cpu": Tensor(Nv, D),
        "video_id2idx": dict,
        "comment_emb_tensor_cpu": Tensor(Nc, D),
        "comment_id2idx": dict
    }
    """
    def __init__(self, field_dims, comments_dims, embed_dim, num_layers,
                 mlp_dims, dropout, text_embeddings,
                 attention_dim=64, nhead=5):
        super().__init__()

        # 基础 embedding
        self.individual_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.shared_embedding = FeaturesEmbedding([comments_dims], embed_dim)
        self.embed_dim = embed_dim

        # (关键修正!)
        # 在模型初始化时, *一次性* 将嵌入表从 CPU 移到 GPU
        # 你的 48GB VRAM 足够容纳它们
        print("Moving embedding tables to GPU...")
        self.video_emb_gpu = text_embeddings["video_emb_tensor_cpu"].cuda()
        self.comment_emb_gpu = text_embeddings["comment_emb_tensor_cpu"].cuda()
        print("Embedding tables moved to GPU.")
        
        # 映射表 (id -> index) 仍然保留在 CPU, 它们很小
        self.video_id2idx = text_embeddings["video_id2idx"]
        self.comment_id2idx = text_embeddings["comment_id2idx"]

        # 获取嵌入维度 (从 GPU 张量)
        self.text_embed_dim = self.video_emb_gpu.size(1)
        self.text_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)
        self.comment_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)

        # DCN 输入维度
        self.embed_output_dim = len(field_dims) * embed_dim + 6 * embed_dim + embed_dim
        self.seq_len = len(field_dims) + 7

        # DCN 主体
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=nhead, dropout=dropout, batch_first=True)

        # 评论打分模块
        self.comment_score_linear = nn.Sequential(
            nn.Linear(self.embed_output_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 6),
        )
        self.comment_score_linear_ = nn.Sequential(
            nn.Linear(self.embed_output_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 6),
        )
        #self.softmax = nn.Softmax(dim=1)

        #self.comment_probs = None
        #self.comment_probs_ = None
        self.comment_logits = None
        self.comment_logits_ = None

    # (移除!) 不再需要慢速的 get_... 函数
    # def get_video_embeddings(self, video_ids): ...
    # def get_comment_embeddings(self, comment_ids): ...

    # ---------------------- forward (已修正为快速版) ----------------------
    def forward(self, x):
        # x 是在 GPU 上的 batch
        current_device = x.device
        
        # 1. 基础特征 (已在 GPU)
        individual_embed_x = self.individual_embedding(x[:, :-6])
        # shared_embed_x = self.shared_embedding(x[:, -6:]) # 这行似乎没用

        # 2. (修正!) 快速、安全地在 GPU 上获取 Video Embeddings
        video_ids_batch = x[:, -8] # (B)
        # 在 CPU 上构建索引列表 (使用 .get() 避免 KeyError, 找不到则用 0)
        idx_video = [self.video_id2idx.get(str(int(v.item())), 0) for v in video_ids_batch]
        # 将 *索引* 移到 GPU
        idx_video_tensor = torch.tensor(idx_video, dtype=torch.long, device=current_device)
        # (快速!) 直接在 GPU 上索引 GPU 嵌入表
        text_embeds = self.video_emb_gpu[idx_video_tensor]
        text_embeds = self.text_dim_reducer(text_embeds) # (在 GPU 上降维)

        # 3. (修正!) 快速、安全地在 GPU 上获取 Comment Embeddings
        comment_ids_batch = x[:, -6:] # (B, 6)
        B, K = comment_ids_batch.size()
        # 在 CPU 上构建 2D 索引列表 (B x K)
        idx_list_comment = [
            [self.comment_id2idx.get(str(int(cid.item())), 0) for cid in row]
            for row in comment_ids_batch
        ]
        # 将 *索引* 移到 GPU
        idx_comment_tensor = torch.tensor(idx_list_comment, dtype=torch.long, device=current_device)
        # (快速!) 直接在 GPU 上索引 GPU 嵌入表
        comment_embeds = self.comment_emb_gpu[idx_comment_tensor]
        comment_embeds = self.comment_dim_reducer(comment_embeds) # (在 GPU 上降维)

        # --- 4. 组合与 DCN (和原来一样) ---
        text_embeds = text_embeds.unsqueeze(1)
        embed_x = torch.cat([individual_embed_x, text_embeds, comment_embeds], dim=1)
        embed_x = embed_x.view(B, -1)

        # Multihead attention
        embed_x_attn, _ = self.multihead_attn(embed_x.view(B, self.seq_len, self.embed_dim),
                                              embed_x.view(B, self.seq_len, self.embed_dim),
                                              embed_x.view(B, self.seq_len, self.embed_dim))
        embed_x = embed_x_attn.reshape(B, -1)

        # DCN
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        p = self.linear(torch.cat([x_l1, h_l2], dim=1)).squeeze(1)

        # 评论打分
        #comment_scores = self.comment_score_linear(embed_x)
        #comment_scores_ = self.comment_score_linear_(embed_x)
        #self.comment_probs = self.softmax(comment_scores)
        #self.comment_probs_ = self.softmax(comment_scores_)
        comment_logits = self.comment_score_linear(embed_x)
        comment_logits_ = self.comment_score_linear_(embed_x)
        
        # 存储 Logits
        self.comment_logits = comment_logits
        self.comment_logits_ = comment_logits_

        return p

    def get_comment_probs(self):
        #return self.comment_probs
        return self.comment_logits

    def get_comment_probs_(self):
        #return self.comment_probs_
        return self.comment_logits_