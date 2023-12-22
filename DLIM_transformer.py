import torch
import torch.nn as nn
import math



def process_feature_map(feature_map, transformer):
    #device = feature_map.device
    #print("device", device)
    #feature_map = feature_map.to(transformer.device)
    #print("transformer.device", transformer.device)
    feature_map = feature_map.to(transformer.device)
    batch_size, num_features, x_cor, y_cor = feature_map.size()
    flattened = feature_map.view(batch_size, num_features, x_cor * y_cor).permute(2, 0, 1)
    transformed = transformer(flattened)
    restored = transformed.permute(1, 2, 0).view(batch_size, num_features, x_cor, y_cor)
    return restored


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomTransformer(nn.Module):
    def __init__(self, feature_size, nhead, nhid, nlayers, dropout=0.5, device=None):
        super(CustomTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_size, dropout, device=device)
        encoder_layers = nn.TransformerEncoderLayer(feature_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.device = device

    def forward(self, src):
        src = src.to(self.pos_encoder.device)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output
