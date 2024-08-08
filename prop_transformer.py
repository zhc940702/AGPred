import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


# 假设 calculate_property 是一个预先定义的函数来获取化合物的特征表示
def calculate_property(chemical_structure):
    # 这个函数的具体实现应根据化学结构计算出一个特征张量
    pass


class PropBert(nn.Module):
    def __init__(self, bert_config_file, output_dim=128):
        super(PropBert, self).__init__()
        self.bert_config = BertConfig.from_json_file(bert_config_file)
        self.bert_model = BertModel(config=self.bert_config)
        self.property_embed = nn.Linear(1, 768)
        self.property_cls = nn.Parameter(torch.zeros(1, 1, 768))
        self.property_proj = nn.Linear(768, output_dim)

    def forward(self, prop_input):
        # Calculate chemical property features
        prop = prop_input.unsqueeze(2)

        # Embed the property
        prop_embedding = self.property_embed(prop)

        # Concatenate the [CLS] token embedding
        properties = torch.cat([self.property_cls.expand(prop.size(0), -1, -1), prop_embedding], dim=1)

        # Pass through the BERT model
        outputs = self.bert_model(inputs_embeds=properties)
        encoded_properties = outputs.last_hidden_state

        # Project the encoding of the [CLS] token to the new feature space and normalize it
        prop_feat = F.normalize(self.property_proj(encoded_properties[:, 0, :]), dim=-1)

        return prop_feat


