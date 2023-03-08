from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):###?
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):##??
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):###??
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):#MASK=tensor w7ayed
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model
#x = lambda a : a + 10
#print(x(5))
    def forward(self, x, mask):
        # self.sublayer[0]==>SublayerConnection((norm): LayerNorm()(dropout): Dropout(p=0.1, inplace=False))
        # self.sublayer[1]==>SublayerConnection((norm): LayerNorm()(dropout): Dropout(p=0.1, inplace=False))

       # self.feed_forward==>
        # PositionwiseFeedForward(
        #   (w_1): Linear(in_features=512, out_features=512, bias=True)
        #   (w_2): Linear(in_features=512, out_features=512, bias=True)
        #   (dropout): Dropout(p=0.1, inplace=False))

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):###??
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #sublayer(self.norm(x))==>matrix
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):#scale,shift# 34an my7sl4 explosion
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Decoder(
           #     DecoderLayer( self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
            #    self.num_layers
            #),#end decoder call
class Decoder(nn.Module):
    def __init__(self, layer, N):
        #layer=>decoder_layer
        # N=>number of layer
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        hid=hidden_states #tensor
        smask=src_mask#matrix w7yed
        tmask=tgt_mask# bool matrix feha true w false
        mem=memory #tensor
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)


#     DecoderLayer( self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
#to transfer its states over generation steps, where
#the states record important pattern information with
#each row (namely, memory slot) representing some
#pattern information.2
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):##sub layer bs bta5od el memory
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)##?
                nn.init.constant_(m.bias, 0.1)##?

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat

##. Then, at time step
##t, the matrix from the previous step, Mt−1, is functionalized as the query and its concatenations with
##the previous output serve as the key and value to
##feed the multi-head attention module. Given H
##heads used in Transformer, there are H sets of
##queries, keys and values via three linear transformations,
#to enrich each token (embedding vector) with contextual information from the whole sentence.
# the self-attention mechanism employs multiple heads (eight parallel attention calculations)
# so that the model can tap into different embedding subspaces
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):#(numofheads,model)
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h#؟؟
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:#؟
            mask = mask.unsqueeze(1)#؟
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)#؟
        return self.linears[-1](x)

#The position-wise feed-forward network (FFN) has a linear layer, ReLU, and another linear layer,
# which process each embedding vector independently with identical weights.

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):#d_ff??
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)#first_linear_layer
        self.w_2 = nn.Linear(d_ff, d_model)##second_linear_layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

#We tokenize an input sentence into distinct elements (tokens)
# as we often do for any neural translation model.
# A tokenized sentence is a fixed-length sequence.
# For instance, if the maximum length is 200, every sentence will have 200 tokens with trailing paddings
#To feed those tokens into the neural network, we convert each token into an embedding vector,
#hese tokens are typically integer indices in a vocabulary dataset. So, it may be a sequence of numbers like the below:
#(8667, 1362, 106, 0, 0, …, 0)
#The number 8667 corresponds to the token Hello


######################################################
#bt7dd el postion bt3 el klma f 3la asaha b3rf m3na el kelma fi el gomla de
#A position  is an integer from 0 to a pre-defined maximum number of tokens in a sentence
##################
class PositionalEncoding(nn.Module):#max_len==>maximum number of tokens in a sentence
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



# #In doing so, the relational memory uses a matrix
# to transfer its states over generation steps, where
# the states record important pattern information with
# each row (namely, memory slot)
# representing some pattern information
# .2 During the generation, the
# matrix is updated step-by-step with incorporating
# the output from previous steps. Then, at time step
# t, the matrix from the previous step, Mt−1, is functionalized as the query and its concatenations with
# the previous output serve as the key and value to
# feed the multi-head attention module. Given H
# heads used in Transformer, there are H sets of
# queries, keys and values via three linear transformations, respectively. For each head, we obtain
# the query, key and value in the relational memory
# through

class RelationalMemory(nn.Module):#extra mem

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots#mem slots
        self.num_heads = num_heads#heads#there are H sets of queries, keys and values
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model) ##to feed multihead
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),#via three linear transformations
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)
#the relational memory uses a matrix
#to transfer its states over generation steps, where
#the states record important pattern information with
#each row (namely, memory slot)
    def init_memory(self, batch_size):##???
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):#query key value # gate mecanizim
        memory = memory.reshape(-1, self.num_slots, self.d_model)#reshape
        q = memory#query
        k = torch.cat([memory, input.unsqueeze(1)], 1)#key
        v = torch.cat([memory, input.unsqueeze(1)], 1)#value
        #During the generation, the matrix is updated step-by-step with incorporating the output from previous steps
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)##??
#gate mechism
 # Consider that the relational memory is performed in a recurrent manner along with the decoding process

#We therefore introduce residual connections and a gate mechanism. The former is formulated
        # as M˜ t = fmlp(Z + Mt−1) + Z + Mt−1 (7)
        # W = nn.Linear(self.d_model, self.d_model * 2)
        #U = nn.Linear(self.d_model, self.d_model * 2)
        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))###??
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy##??
        attn = MultiHeadedAttention(self.num_heads, self.d_model)##extract true meaing of word mn elsyak

        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)##??+d_ff?
        position = PositionalEncoding(self.d_model, self.dropout)##btt7dd el postion
        rm = RelationalMemory(num_slots=self.rm_num_slots,  d_model=self.rm_d_model,num_heads=self.rm_num_heads)
        model = Transformer(# self.encoder = encoder, self.decoder = decoder, self.src_embed = src_embed,self.tgt_embed = tgt_embed, self.rm = rm
            #The encoder block uses the self-attention mechanism  to enrich each token (embedding vector) with contextual information from the whole sentence.
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),##?
            Decoder(
                DecoderLayer( self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers
            ),#end decoder call
            lambda x: x,#src_embd???
            nn.Sequential(  Embeddings(self.d_model, tgt_vocab), c(position)),#trgt_embd??
            rm#relatiom mem
        )
        for p in model.parameters():#??
            if p.dim() > 1:#??
                nn.init.xavier_uniform_(p)#??
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):###???
        # Clip the length of att_masks and att_feats to the maximum length

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs
    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):###??
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
