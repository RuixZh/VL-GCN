import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.sequence_modules import TemporalTransformer, TAggregate, LayerNorm
from modules.optimization import KLLoss, CrossEn
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, isBias=True):
        super(GraphConvolution, self).__init__()
        self.fc_1 = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()
        self.isBias = isBias

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, input, adj):
        seq = self.fc_1(input)
        seq = torch.spmm(adj, seq)
        if self.isBias:
            seq += self.bias_1
        return self.act(seq)

class SemanticAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.W = nn.Parameter(torch.zeros(size=(in_ft, out_ft)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_ft)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_ft)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        N = x.size()[1]
        h = torch.mm(x.view(-1, self.in_ft), self.W)
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0],1))
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(2, -1)
        semantic_attentions = semantic_attentions.mean(dim=1,keepdim=True)
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        semantic_attentions = semantic_attentions.view(2, 1,1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_ft)

        input_embedding = x.view(2, N, self.in_ft)

        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding

class KGLearner(nn.Module):
    def __init__(self, args, embed_dim, transformer_heads):
        super(KGLearner, self).__init__()

        self.args = args
        self.c2d_gc = GraphConvolution(embed_dim, embed_dim) # 1 layer
        self.v2d_gc = GraphConvolution(embed_dim, embed_dim) # 1 layer
        self.d2v_gc = GraphConvolution(embed_dim, embed_dim) # 1 layer
        self.d2v_gc2 = GraphConvolution(embed_dim, embed_dim) # 2 layer
        self.semantic_att = SemanticAttention(embed_dim, embed_dim//4)
        self.fc = nn.Linear(3 * embed_dim, embed_dim)
        # self.classifier = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     torch.nn.ReLU(),
        #     nn.Linear(embed_dim // 2, 24)
        # )
        self.classifier = nn.Linear(embed_dim, 24)

        assert self.args.sim_header in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        num_frames = args.max_frames

        if self.args.sim_header == "Transf" :
            self.frame_position_embeddings = nn.Embedding(self.args.max_frames, embed_dim)
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)

        if self.args.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)


        self.apply(self.init_weights)

        if self.args.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.max_frames, embed_dim=embed_dim, n_layers=6)

        if self.args.sim_header == 'Conv_1D' :
            self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
            weight = torch.zeros(embed_dim, 1, 3)
            weight[:embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4:, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

        self.loss_fct = nn.CrossEntropyLoss()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_logits(self, x1, x2, logit_scale):
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_x1 = logit_scale * x1 @ x2.t()
        logits_per_x2 = logits_per_x1.t()

        return logits_per_x1, logits_per_x2

    def seq_encoder(self, x):
        b, t, c = x.size()
        x = x.contiguous()

        if self.args.sim_header == "meanP":
            pass
        elif self.args.sim_header == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.args.sim_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.args.sim_header == "LSTM":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.args.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError('Unknown optimizer: {}'.format(self.args.sim_header))
        return x.mean(dim=1, keepdim=False)

    def forward(self, frame_emb, cd_adj, dc_adj, vd_adj, dv_adj, subevent, event, logit_scale, ground_truth=None):
        '''
        frame_emb: visual emb (max_frame, dim)
        cd_adj: description-category adjacency matrix  (365, 24)
        vd_adj: video-description adjacency matrix  (bs, 365)
        subevent: description embedding (365, dim)
        event: category embedding (24, dim)
        '''
        video_emb = self.seq_encoder(frame_emb)

        num_sevt = subevent.shape[0]
        num_evt = event.shape[0]
        bs = frame_emb.shape[0]
        # 1 layer
        c2d_emb = self.c2d_gc(event.float(), dc_adj) # (subevent, dim)
        v2d_emb = self.v2d_gc(video_emb.float(), dv_adj.t()) # (subevent, dim)
        d2v_emb = self.d2v_gc(subevent.float(), vd_adj) # (video, dim)
        # att or mean
        # v2d_att_emb = (c2d_emb + v2d_emb) / 2
        att_emb = torch.stack([c2d_emb, v2d_emb], 0)
        v2d_att_emb = self.semantic_att(att_emb)

        # 2 layer
        d2v_emb2 = self.d2v_gc2(v2d_att_emb, vd_adj) # (video, dim)

        video_cat = self.fc(torch.cat([d2v_emb2, d2v_emb, video_emb], -1))

        preds = self.classifier(video_cat)

        loss = self.loss_fct(preds, ground_truth)

        values_1, indices_1 = preds.view(bs, -1).softmax(dim=-1).topk(1, dim=-1)
        return loss, indices_1
