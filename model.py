from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling, GATConv,GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from config import TrainingConfig
import pickle
from prop_transformer import PropBert
import math


class SelfAttention_new(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention_new, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        # Expand the batch_size, embedding_size input to batch_size, 1, embedding_size
        # to simulate a sequence of length 1
        # print('in', x)
        x = x.unsqueeze(1)

        N = x.shape[0]
        value_len, key_len, query_len = 1, 1, 1

        # Split the embedding into self.heads pieces
        values = x.reshape(N, value_len, self.heads, self.head_dim)
        keys = x.reshape(N, key_len, self.heads, self.head_dim)
        queries = x.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Apply the final linear transformation and remove the sequence length dimension
        out = self.fc_out(out).squeeze(1)
        # print('out', out)
        return out

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * num_attention_heads, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class multimodal_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(multimodal_Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, num_attention_heads, hidden_dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size
        self.all_head_size = hidden_size * num_attention_heads

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores_1(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_2(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, int(self.attention_head_size / self.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores_1(mixed_query_layer)
        key_layer = self.transpose_for_scores_1(mixed_key_layer)
        value_layer = self.transpose_for_scores_1(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_head_size * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


# class GNN(torch.nn.Module):
#     def __init__(self, feature_size: int, edge_size: int, config: TrainingConfig):
#         super().__init__()
#         self.config = config
#
#         self.conv_layers = ModuleList([])
#         self.transform_layers = ModuleList([])
#         self.pooling_layers = ModuleList([])
#         self.bn_layers = ModuleList([])
#
#         # Transformation layer
#         self.conv1 = TransformerConv(
#             in_channels=feature_size,
#             out_channels=self.config.embedding_size,
#             heads=self.config.n_heads,
#             dropout=self.config.dropout_rate,
#             edge_dim=edge_size,
#             beta=True,
#         )
#         self.transform1 = Linear(
#             in_features=self.config.embedding_size * self.config.n_heads,
#             out_features=self.config.embedding_size,
#         )
#         self.bn1 = BatchNorm1d(num_features=self.config.embedding_size)
#
#         # Other layers
#         for layer in range(self.config.n_layers):
#             conv = TransformerConv(
#                 in_channels=self.config.embedding_size,
#                 out_channels=self.config.embedding_size,
#                 heads=self.config.n_heads,
#                 dropout=self.config.dropout_rate,
#                 edge_dim=edge_size,
#                 beta=True,
#             )
#             self.conv_layers.append(conv)
#
#             transform = Linear(
#                 in_features=self.config.embedding_size * self.config.n_heads,
#                 out_features=self.config.embedding_size,
#             )
#             self.transform_layers.append(transform)
#
#             bn = BatchNorm1d(num_features=self.config.embedding_size)
#             self.bn_layers.append(bn)
#
#             if layer % self.config.top_k_every_n == 0:
#                 pooling = TopKPooling(
#                     in_channels=self.config.embedding_size,
#                     ratio=self.config.top_k_ratio,
#                 )
#                 self.pooling_layers.append(pooling)
#
#         # Linear output layers
#         self.linear1 = Linear(self.config.embedding_size * 2, self.config.dense_neurons)
#         self.linear2 = Linear(self.config.dense_neurons, int(self.config.dense_neurons / 2))
#         self.linear3 = Linear(int(self.config.dense_neurons / 2), 1)
#
#     def forward(
#             self,
#             x: torch.Tensor,
#             edge_index: torch.Tensor,
#             edge_attr: torch.Tensor,
#             batch_index: Optional[torch.Tensor] = None,
#
#     ):
#         # Initial transformation
#         x = self.conv1(x, edge_index, edge_attr)
#         x = torch.relu(self.transform1(x))
#         x = self.bn1(x)
#
#         # Hold the intermediate graph representations
#         global_representation = []
#
#         # Intermediate blocks
#         layer_num = 0
#         for layer in range(self.config.n_layers):
#             x = self.conv_layers[layer](x, edge_index, edge_attr)
#             x = torch.relu(self.transform_layers[layer](x))
#             x = self.bn_layers[layer](x)
#             # layer_num = layer_num + 1
#             # print('layer_num ', layer_num)
#
#             # Always aggregate last layer
#             if layer % self.config.top_k_every_n == 0 or layer == self.config.n_layers - 1:
#                 pooling = self.pooling_layers[int(layer / self.config.top_k_every_n)]
#                 x, edge_index, edge_attr, batch_index, _, _ = pooling(
#                     x, edge_index, edge_attr, batch_index
#                 )
#                 # Add current representation
#                 # aaa = gmp(x, batch_index)
#                 # bbb = gap(x, batch_index)
#                 global_representation.append(
#                     torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
#                 )
#             rep = sum(global_representation)
#             x = torch.relu(self.linear1(rep))
#             x = F.dropout(x, p=0.8, training=self.training)
#             x = torch.relu(self.linear2(x))
#             x = F.dropout(x, p=0.8, training=self.training)
#             x = self.linear3(x)
#
#             return x, rep


class GNN(torch.nn.Module):
    def __init__(self, feature_size: int, edge_size: int, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.conv_layers = ModuleList([])
        self.transform_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer using GATConv
        self.conv1 = TransformerConv(
            in_channels=feature_size,
            out_channels=self.config.embedding_size,
            heads=self.config.n_heads,
            dropout=self.config.dropout_rate,
            edge_dim=edge_size
        )
        self.transform1 = Linear(
            in_features=self.config.embedding_size * self.config.n_heads,
            out_features=self.config.embedding_size,
        )
        # self.transform1 = Linear(
        #     in_features=self.config.embedding_size,
        #     out_features=self.config.embedding_size,
        # )
        self.bn1 = BatchNorm1d(num_features=self.config.embedding_size)

        # Other layers
        for layer in range(self.config.n_layers):
            conv = TransformerConv(
                in_channels=self.config.embedding_size,
                out_channels=self.config.embedding_size,
                heads=self.config.n_heads,
                dropout=self.config.dropout_rate,
                edge_dim=edge_size
            )
            self.conv_layers.append(conv)

            transform = Linear(
                in_features=self.config.embedding_size * self.config.n_heads,
                out_features=self.config.embedding_size,
            )
            # transform = Linear(
            #     in_features=self.config.embedding_size,
            #     out_features=self.config.embedding_size,
            # )
            self.transform_layers.append(transform)

            bn = BatchNorm1d(num_features=self.config.embedding_size)
            self.bn_layers.append(bn)

            if layer % self.config.top_k_every_n == 0:
                pooling = TopKPooling(
                    in_channels=self.config.embedding_size,
                    ratio=self.config.top_k_ratio,
                )
                self.pooling_layers.append(pooling)

        # Linear output layers
        self.linear1 = Linear(self.config.embedding_size * 2, self.config.dense_neurons)
        self.linear2 = Linear(self.config.dense_neurons, int(self.config.dense_neurons / 2))
        self.linear3 = Linear(int(self.config.dense_neurons / 2), 1)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch_index: Optional[torch.Tensor] = None,
    ):
        # Initial transformation

        # x = self.conv1(x, edge_index, edge_attr)
        x = self.conv1(x, edge_index, edge_attr)

        x = torch.relu(self.transform1(x))
        x = self.bn1(x)

        # Hold the intermediate graph representations
        global_representation = []

        # Intermediate blocks
        for layer in range(self.config.n_layers):
            # x = self.conv_layers[layer](x, edge_index, edge_attr)
            x = self.conv_layers[layer](x, edge_index, edge_attr)
            x = torch.relu(self.transform_layers[layer](x))
            x = self.bn_layers[layer](x)

            # Pooling logic remains unchanged
            if layer % self.config.top_k_every_n == 0 or layer == self.config.n_layers - 1:
                pooling = self.pooling_layers[int(layer / self.config.top_k_every_n)]
                x, edge_index, edge_attr, batch_index, _, _ = pooling(
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(
                    torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
                )

        rep = sum(global_representation)
        x = torch.relu(self.linear1(rep))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x, rep


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_classes):
        super(GAT, self).__init__()

        # GAT layers
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)

        # Output layer
        self.fc = torch.nn.Linear(hidden_dim * num_heads * 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GAT layers
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # Global pooling to obtain graph-level representation
        rep = torch.cat([gmp(x, data.batch), gap(x, data.batch)], dim=1)

        # Apply the final classification layer
        x = self.fc(rep)
        x = torch.sigmoid(x)

        return x, rep


class CAM(torch.nn.Module):
    def __init__(self, feature_size: int, edge_size: int, config: TrainingConfig):
        super(CAM, self).__init__()
        # 最后的mlp分类的输入维度
        mlp_in_dim = 256
        # mlp_in_dim = 896
        # mlp_in_dim = 100
        mlp_hidden_dim = 512
        mlp_out_dim = 128
        # GNN model
        self.gnn = GNN(feature_size=feature_size, edge_size=edge_size, config=config)
        # self.gnn = GAT()

        # self.bert = BERT()
        self.prop_bert = PropBert('config_bert_property.json')
        self.feature_extractor = MyNet(input_size=512, output_size=128)

        self.cross_attention = CrossAttention(128, 32)
        self.gate_fusion = GatedMultimodalLayer(128, 128, 256)
        self.cross_attention1 = CrossAttention(128, 32)
        # self.cross_attention_1 = CrossAttention1(128, 768, 32)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
        self.classifier = nn.Linear(mlp_out_dim, 1)
        # self.self_attention = SelfAttention_new(256, 8)
        # self.attenion = multimodal_Attention(128, 4, 0.2, 0.2)

    def forward(self, data, morgan, ids, prop, feat_kg):
        # 计算分子图特征
        # print(data.edge_attr.float().shape)
        # print(data.edge_attr.float())
        _, rep_graph = self.gnn(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        # 提取分子指纹特征
        rep_morgan = self.feature_extractor(morgan)
        prop_feature = self.prop_bert(prop)
        # 交叉注意力获取分子图和指纹的联合特征
        rep_chem = self.cross_attention(rep_graph, rep_morgan)
        #
        # rep_all = self.cross_attention1(prop_feature, rep_chem)
        # 计算dti-bert的特征
        # _, rep_dti = self.bert(ids)
        # 交叉注意力获取结构和靶标的联合特征
        # rep_cross = self.cross_attention(rep_chem, rep_dti)
        combined_rep = torch.cat((prop_feature, rep_chem), dim=1)

        # combined_rep = self.gate_fusion(prop_feature, rep_chem)
        # score = self.mlp_classifier(rep_all)
        # total_feature = torch.stack((rep_chem, prop_feature), 1)
        # attention_output = self.attenion(total_feature)
        # combined_rep = attention_output.view(attention_output.shape[0], -1)
        # self_attention_rep = self.self_attention(combined_rep)
        score = self.mlp_classifier(combined_rep)
        # score = self.mlp_classifier(feat_kg)
        logits_clsf = self.classifier(score)
        return logits_clsf, score


class MyNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 256)
        self.linear5 = torch.nn.Linear(256, output_size)
        self.norm = torch.nn.LayerNorm(256)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.norm(self.linear3(x) + x)
        x = self.norm(self.linear3(x) + x)
        x = self.norm(self.linear3(x) + x)
        x = F.relu(self.linear5(x))
        return x


# class CrossAttention1(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim):
#         super(CrossAttention1, self).__init__()
#         self.query = nn.Linear(input_dim1, output_dim)
#         self.key = nn.Linear(input_dim2, output_dim)
#         self.value = nn.Linear(input_dim2, output_dim)
#         self.norm = nn.LayerNorm(output_dim)
#         self.linear = nn.Linear(output_dim, 128)
#
#     def forward(self, input1, input2):
#         q = self.query(input1)
#         k = self.key(input2)
#         v = self.value(input2)
#         attention_scores = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力分数
#         attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化得到注意力权重
#         fused_feature = torch.matmul(attention_weights, v)  # 加权融合特征
#         output = self.linear(fused_feature)
#         return output


class CrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.linear = nn.Linear(output_dim, 128)

    def forward(self, input1, input2):
        q = self.query(input1)
        k = self.key(input2)
        v = self.value(input2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力分数
        attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化得到注意力权重
        fused_feature = self.norm(torch.matmul(attention_weights, v))  # 加权融合特征
        output = self.linear(fused_feature)
        return output


class GatedMultimodalLayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo et al.' (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1 = size_in1
        self.size_in2 = size_in2
        self.size_out = size_out

        # Define layers
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        # Process inputs through hidden layers with Tanh activation
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))

        # Concatenate the hidden layers' outputs
        x = torch.cat((h1, h2), dim=1)

        # Process concatenated outputs through sigmoid layer
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        # Apply gating mechanism
        gated_h1 = z.view(z.size(0), -1) * h1
        gated_h2 = (1 - z.view(z.size(0), -1)) * h2

        # Return the gated combination of the two modalities
        return gated_h1 + gated_h2


# class CrossAttention(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout_rate=0.1):
#         super(CrossAttention, self).__init__()
#         self.query = nn.Linear(input_dim, output_dim)
#         self.key = nn.Linear(input_dim, output_dim)
#         self.value = nn.Linear(input_dim, output_dim)
#         self.norm = nn.LayerNorm(output_dim)
#         self.dropout = nn.Dropout(dropout_rate)  # 初始化 Dropout 层
#         self.linear = nn.Linear(output_dim, 128)
#
#     def forward(self, input1, input2):
#         q = self.query(input1)
#         k = self.key(input2)
#         v = self.value(input2)
#
#         attention_scores = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力分数
#         attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化得到注意力权重
#         attention_weights = self.dropout(attention_weights)  # 在注意力权重上应用 Dropout
#
#         fused_feature = torch.matmul(attention_weights, v)  # 加权融合特征
#         fused_feature = self.norm(fused_feature)  # 应用层归一化
#         fused_feature = self.dropout(fused_feature)  # 在融合特征上应用 Dropout
#
#         output = self.linear(fused_feature)
#         return output


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return x


# class MLPDecoder(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.5):
#         super(MLPDecoder, self).__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#
#         self.dropout = nn.Dropout(dropout_rate)  # Only one dropout layer
#
#         self.fc3 = nn.Linear(hidden_dim, out_dim)
#         self.bn3 = nn.BatchNorm1d(out_dim)
#
#     def forward(self, x):
#         x = self.bn1(F.relu(self.fc1(x)))
#         x = self.bn2(F.relu(self.fc2(x)))
#         x = self.dropout(x)  # Apply dropout here, before the last BN layer
#         x = self.bn3(F.relu(self.fc3(x)))
#         return x


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        embedding = self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        global n_head
        n_head = 6
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        global batch_size, max_pred, n_layers, n_heads, d_model, d_ff, d_k, d_v
        batch_size = 16
        max_pred = 5  # max tokens of prediction
        n_layers = 3
        n_heads = 6  # number of heads in Multi-Head Attention
        d_model = 1024  # Embedding Size
        d_ff = 128 * 4  # 4*d_model, FeedForward dimension
        d_k = d_v = 64  # dimension of K(=Q), V
        with open('ben_targets2index.pickle', 'rb') as file:
            token2index = pickle.load(file)
        vocab_size = len(token2index)

        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.classifier = nn.Linear(2, 2)

    def forward(self, input_ids):
        output = self.embedding(input_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            # output: [batch_size, max_len, d_model]

        # classification
        # only use [CLS]
        representation = output[:, 0, :]
        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)
        # representation = reduction_feature
        return logits_clsf, representation
