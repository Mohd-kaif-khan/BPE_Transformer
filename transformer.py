import json
import math
import torch
import tokens
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)

        position = torch.arange(self.max_seq_len).reshape(self.max_seq_len,1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE.to(get_device())
    

class SentenceEmbedding(nn.Module):
    def __init__(self, lang_to_index, max_seq_len, d_model, type,start_token,end_token,padding_token) -> None:
        super().__init__()
        self.lang_to_index = lang_to_index
        self.max_seq_len = max_seq_len
        self.typee = type
        self.embedding = nn.Embedding(len(self.lang_to_index), d_model)
        self.start = start_token
        self.end = end_token
        self.padding = padding_token
        self.position_encoder = PositionalEncoding(max_seq_len, d_model)
    def batchTokenize(self, batch, start, end):

        def tokenize(sentence, start, end):
            ids = tokens.encode(sentence, self.typee)
            if start:
                ids.insert(0, self.lang_to_index[self.start])
            if end:
                ids.append(self.lang_to_index[self.end])

            for _ in range(len(ids), self.max_seq_len):
                ids.append(self.lang_to_index[self.padding])
           
            return torch.tensor(ids)
        
        tokenized = []

        for i in batch:
            tokenized.append(tokenize(i, start, end))
        tokenized = torch.stack(tokenized)

        return tokenized.to(get_device())
    
    def forward(self, batch, start, end):
        x = self.batchTokenize(batch, start, end)

        x = self.embedding(x)
        pos = self.position_encoder()

        return (x+pos)
    

class MultiheadAttention(nn.Module):
    def __init__(self, d_model,num_head,head_dim) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = head_dim
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size , seq_len, input_dim = x.size()

        qkv = self.qkv_layer(x).reshape(batch_size, seq_len, self.num_head, 3*self.head_dim)

        qkv = qkv.permute(0,2,1,3)

        q,k,v = qkv.chunk(3, dim=-1)

        scaled = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scaled = scaled.permute(1,0,2,3) + mask
            scaled = scaled.permute(1,0,2,3)
        
        attention = F.softmax(scaled, dim=-1)
        #for making padding zero
        # attention = torch.where(torch.isnan(attention), torch.tensor(0.0), attention)
        Values = torch.matmul(attention, v)

        return self.linear(Values.permute(0,2,1,3).reshape(batch_size, seq_len, self.d_model))


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EncoderLayers(nn.Module):
    def __init__(self, d_model, num_head,head_dim,drop_prob,ffn) -> None:
        super().__init__()
        self.attention = MultiheadAttention(d_model,num_head,head_dim)
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.feedforward = PositionwiseFeedForward(d_model, ffn, drop_prob)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        _x = x.clone()
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = self.norm(_x+x)

        _x = x.clone()
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.norm(_x + x)

        return x
    

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, mask = inputs
        for module in self._modules.values():
            x = module(x, mask)
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model,max_seq_len, num_head, head_dim, drop_prob, english_to_index, ffn, encoder_type, n_layers, start_token,end_token,padding_token) -> None:
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(english_to_index, max_seq_len, d_model, encoder_type,start_token,end_token,padding_token)
        self.layers = SequentialEncoder( *[EncoderLayers(d_model, num_head,head_dim,drop_prob,ffn) for _ in range(n_layers)] )
    def forward(self, x, mask, start = False, end=False):
        x = self.sentence_embedding(x, start, end)
        x = self.layers(x, mask)
        return x
    

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, head_dim) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = head_dim
        self.kv_layer =nn.Linear(d_model, d_model*2)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask):
        batch_size, max_seq_len, input_dim = x.size()
        kv = self.kv_layer(x).reshape(batch_size, max_seq_len, self.num_head, self.head_dim*2).permute(0,2,1,3)
        q = self.q_layer(y).reshape(batch_size, max_seq_len, self.num_head, self.head_dim).permute(0,2,1,3)

        k, v = kv.chunk(2, dim=-1)

        scaled = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scaled = scaled.permute(1,0,2,3)+mask
            scaled = scaled.permute(1,0,2,3)
        
        attention = F.softmax(scaled, dim=-1)
        #for making padding zero
        # attention = torch.where(torch.isnan(attention), torch.tensor(0.0), attention)
        Values = torch.matmul(attention, v).permute(0,2,1,3).reshape(batch_size,max_seq_len, self.d_model)

        return self.linear(Values)
    

class DecoderLayers(nn.Module):
    def __init__(self, d_model, num_head,head_dim,drop_prob,ffn) -> None:
        super().__init__()
        self.attention = MultiheadAttention(d_model,num_head,head_dim)
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.dropout = nn.Dropout(p=drop_prob)
        self.feedforward = PositionwiseFeedForward( d_model, ffn, drop_prob)
        self.crossattention = CrossMultiHeadAttention(d_model, num_head, head_dim)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.attention(y,self_attention_mask)
        y = self.dropout(y)
        y = self.norm(_y + y)

        _y = y.clone()
        y = self.crossattention(x, y, cross_attention_mask)
        y = self.dropout(y)
        y = self.norm(_y + y)

        _y = y.clone()
        y = self.feedforward(y)
        y = self.dropout(y)

        y = self.norm(_y + y)

        return y
    

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs

        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y
    

class Decoder(nn.Module):
    def __init__(self, d_model,max_seq_len, num_head, head_dim, drop_prob, hindi_to_index, ffn, decoder_type, n_layers, start_token, end_token, padding_token) -> None:
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(hindi_to_index, max_seq_len, d_model, decoder_type,start_token,end_token,padding_token)
        self.layers = SequentialDecoder(*[ DecoderLayers(d_model, num_head,head_dim,drop_prob,ffn) for _ in range(n_layers) ])
    
    def forward(self, x, y, self_attention_mask, cross_attention_mask, start=True, end=True):
        y = self.sentence_embedding(y, start, end)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y
    
    
class Transformer(nn.Module):
    def __init__(self, d_model, max_seq_len, num_head, head_dim, drop_prob, english_to_index, ffn, hindi_to_index, encoder_type, decoder_type, 
                 n_layers, start_token, end_token, padding_token) -> None:
        super().__init__()
        self.encoder = Encoder(d_model,max_seq_len, num_head, head_dim, drop_prob, english_to_index, ffn, encoder_type, n_layers, start_token, end_token, padding_token)
        self.decoder = Decoder(d_model,max_seq_len, num_head, head_dim, drop_prob, hindi_to_index, ffn, decoder_type, n_layers, start_token, end_token, padding_token)
        self.linear = nn.Linear(d_model, len(hindi_to_index))
        self.checker = None
    def forward(self, x, y, encoder_mask, decoder_attention_mask, decoder_cross_attention_mask):
        x = self.encoder(x, encoder_mask)
        y = self.decoder(x, y, decoder_attention_mask, decoder_cross_attention_mask)

        out = self.linear(y)
        if self.checker is None:
            print("****************************************************")
            self.checker = 100
        return out
        # return F.softmax(out, dim=-1)