import math
import torch
from torch import nn
import torch.nn.functional as F

class GRUPosterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_gru_layers, drop_prob=0.2):
        super(GRUPosterior, self).__init__()
        # the drop_prob won't work if the n_gru_layers is set as 1
        # the input data should be of (seq_len, bs, input_dim)
        self.hidden_dim = hidden_dim
        self.n_gru_layers = n_gru_layers
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_gru_layers, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (seq_len, bs, input_dim)
        # out: (seq_len, bs, hidden_dim), h: (n_gru_layers, bs, hidden_dim) which is the final hidden state
        out, h = self.gru(x) # if we don't provide an initial hidden tensor as input, it will be a zero tensor by default
        assert out.shape[1] == 1
        out = out.view(-1, self.hidden_dim)
        # out: (seq_len, output_dim)
        out = self.fc(self.relu(out)) # log_prob

        return out

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_gru_layers, batch_size, self.hidden_dim).zero_()

        return hidden

    def forward_step(self, x, h):
        # out: (1, bs, hidden_dim), h: (n_gru_layers, bs, hidden_dim)
        out, h = self.gru(x, h)
        # out: (bs, out_dim)
        out = self.fc(self.relu(out[-1]))

        return out, h


class ContextPosterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_limit: float):
        super(ContextPosterior, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        nn.init.zeros_(self.linear.bias)
        self.context_limit = context_limit
        self.output_dim = output_dim

    def forward(self, seq): # (bs, epi_len, input_dim)
        self.inter_states, _ = self.lstm(seq) # (bs, epi_len, 2 * hidden_dim) # bi-lstm
        logit_seq = self.linear(self.inter_states) # (bs, epi_len, context_dim) # linear
        self.logits = torch.mean(logit_seq, dim=1) # (bs, context_dim) # average pooling since the bi-LSTM is location-invariant
        mean = self.logits

        return mean

    def log_prob_context(self, seq, cnt):
        mean = self.forward(seq) # (bs, context_dim)
        log_probs = F.log_softmax(mean, dim=-1) # (bs, context_dim)
        log_prob = (log_probs * cnt).sum(dim=-1, keepdim=True) # (bs, 1)

        return log_prob

    def sample_context(self, seq, fixed=False):
        mean = self.forward(seq) # (bs, context_dim)
        if fixed:
            sample = mean.argmax(dim=-1, keepdim=True) # (bs, 1)
            return F.one_hot(sample.squeeze(-1), num_classes=self.output_dim).float() # (bs, context_dim)
        else:
            sample = F.gumbel_softmax(mean, hard=False, tau=1.0).multinomial(1).long()
            return F.one_hot(sample.squeeze(-1), num_classes=self.output_dim).float() # (bs, context_dim)


    def sample_contexts(self, seq_list, fixed=False): # seq_list: [(seq_num, seq_len, input_dim), (seq_num, seq_len, input_dim), ...]
        # get the minimal trajectory length, the same idea is used in the implementation of smile
        min_traj_len = min([seq.shape[1] for seq in seq_list])

        input_list = []
        for seq in seq_list:
            assert seq.shape[0] == 1
            input_list.append(seq[:, :min_traj_len, :])
        input_tensor = torch.cat(input_list, dim=0) # (bs, seq_len, input_dim)
        cnt = self.sample_context(input_tensor, fixed=fixed)

        return cnt


class MLPContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, inter_dim, output_dim, context_limit: float):
        super(MLPContextEncoder, self).__init__()

        self.trans_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, inter_dim))

        self.aggregator = nn.Sequential(
            nn.Linear(inter_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_fc = nn.Linear(hidden_dim, output_dim)
        self.log_sig_fc = nn.Linear(hidden_dim, output_dim)

        self.input_dim = input_dim
        self.context_limit = context_limit

    def forward(self, seq):
        # seq_num * (seq_len, input_dim), note that the length of each sequence may not be the same
        # we don't cut them to make them have the same length like in smile implementation, since the goal info may be essential
        # instaed we take the whole length into consideration, which will ne a strict implementation of their paper
        seq_num = len(seq)
        embeddings = []
        for i in range(seq_num):
            input_list = seq[i] # (seq_len, input_dim)
            embedding_list = self.trans_enc(input_list) # (seq_len, inter_dim)
            embedding = torch.mean(embedding_list, dim=0, keepdim=True) # (1, inter_dim)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0) # (seq_num, inter_dim)
        embedding = torch.sum(embeddings, dim=0, keepdim=True) # (1, inter_dim)
        hid_info = self.aggregator(embedding) # (1, hidden_dim)
        mean = self.mean_fc(hid_info)
        log_std = self.log_sig_fc(hid_info)
        return mean, log_std

    def sample_context(self, seq, fixed=False): # used only for predicting the context for the expert trajectories
        # this is truncated since the true context is also truncated
        # different from the smile implementation because we have different context distribution definitions
        mean, log_std = self.forward(seq)
        if fixed:
            context = mean
        else:
            eps = torch.empty_like(mean).normal_()
            context = mean + log_std.exp() * eps

        # return self.context_limit * torch.tanh(context/self.context_limit)
        # TODO
        return context.clamp(-self.context_limit, self.context_limit) # self.context_limit * sigma

    def sample_contexts(self, seq_list, fixed=False): # seq_list: [(seq_num, seq_len, input_dim), (seq_num, seq_len, input_dim), ...]
        # get the minimal trajectory length, the same idea is used in the implementation of smile
        min_traj_len = min([seq.shape[1] for seq in seq_list])
        input_list = []
        for seq in seq_list:
            input_list.append((seq[:, :min_traj_len, :]).unsqueeze(0))
        input_tensor = torch.cat(input_list, dim=0) # (bs, seq_num, seq_len, input_dim)
        bs, seq_num, seq_len = input_tensor.shape[:3]

        embedding_list = self.trans_enc(input_tensor.view(-1, self.input_dim))
        embedding_list = embedding_list.reshape(bs, seq_num, seq_len, -1)
        embeddings = torch.mean(embedding_list, dim=-2)
        embedding = torch.sum(embeddings, dim=1)

        hid_info = self.aggregator(embedding)  # (bs, hidden_dim)
        mean = self.mean_fc(hid_info)
        log_std = self.log_sig_fc(hid_info)
        if fixed:
            context = mean
        else:
            eps = torch.empty_like(mean).normal_()
            context = mean + log_std.exp() * eps

        # return self.context_limit * torch.tanh(context/self.context_limit)
        # TODO
        return context.clamp(-self.context_limit, self.context_limit) # self.context_limit * sigma







