# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   model_cross_transformer.py
# @Time   :   2021/9/3 15:09
import numpy as np
import torch.nn as nn

from .model_transformer import PositionWiseFeedForward, ScaledDotProductAttention


class CrossMultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention module
    '''

    def __init__(self, n_head, d_model_q, d_model_kv, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_q, n_head * d_k)
        self.w_ks = nn.Linear(d_model_kv, n_head * d_k)
        self.w_vs = nn.Linear(d_model_kv, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model_q + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model_kv + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model_kv + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model_q)

        self.fc = nn.Linear(n_head * d_v, d_model_q)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q (len_q, dim_q)
            k (len_k, dim_k)
            v (len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (len_q, d_model)
            attn (len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        len_q, _ = q.size()
        len_k, _ = k.size()
        len_v, _ = v.size()  # len_k==len_v

        residual = q

        q = self.w_qs(q).view(len_q, n_head, d_k)
        k = self.w_ks(k).view(len_k, n_head, d_k)
        v = self.w_vs(v).view(len_v, n_head, d_v)

        q = q.permute(1, 0, 2).contiguous().view(-1, len_q, d_k)  # nh x lq x dk
        k = k.permute(1, 0, 2).contiguous().view(-1, len_k, d_k)  # nh x lk x dk
        v = v.permute(1, 0, 2).contiguous().view(-1, len_v, d_v)  # nh x lv x dv
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n) x .. x ..
        output, attn = self.attention(q, k, v, mask=attn_mask)

        output = output.view(n_head, len_q, d_v)
        output = output.permute(1, 0, 2).contiguous().view(len_q, -1)  # lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class CrossEncoderLayer(nn.Module):
    """
    Compose with two layers: cross_attn and ffn
    """

    def __init__(self, d_model_q, d_model_kv, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(CrossEncoderLayer, self).__init__()
        self.cross_attn = CrossMultiHeadAttention(n_head, d_model_q, d_model_kv, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model_q, d_inner, dropout=dropout)

    def forward(self, q, k, v, non_pad_mask=None, attn_mask=None):
        """

        Args:
            q:
            k:
            v:
        Returns:

        """
        enc_output, enc_slf_attn = self.cross_attn(
            q, k, v, attn_mask=attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output.unsqueeze(0))
        if non_pad_mask is not None:
            enc_output *= non_pad_mask.float()
        enc_output = enc_output.squeeze()
        return enc_output, enc_slf_attn


class CrossTransformerEncoder(nn.Module):
    """
    A encoder model with cross-attention mechanism.
    """

    def __init__(self, n_layer, num_head, k_dim, v_dim, d_model_q, d_model_kv, d_inner, dropout_rate=0.1,
                 graph_matrix=None):
        super().__init__()
        self.graph_matrix = graph_matrix
        self.layer_stack = nn.ModuleList([
            CrossEncoderLayer(d_model_q, d_model_kv, d_inner, num_head, k_dim, v_dim, dropout=dropout_rate)
            for _ in range(n_layer)])

    def forward(self, edge_repr, obj_repr_k, obj_repr_v):
        """
        Args:
            edge_repr:
            obj_repr_k:
            obj_repr_v:
        Returns:
            enc_output [Tensor] (#total_box, d_model)
        """
        slf_attn_mask = None
        non_pad_mask = None

        enc_output = edge_repr
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, obj_repr_k, obj_repr_v,
                non_pad_mask=non_pad_mask,
                attn_mask=slf_attn_mask
            )

        return enc_output
