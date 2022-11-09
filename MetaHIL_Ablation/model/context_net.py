import math
import torch
from torch import nn

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
        self.a_log_std = torch.nn.Parameter(torch.empty(1, output_dim, dtype=torch.float32).fill_(0.))
        self.context_limit = context_limit

    def forward(self, seq): # (bs, epi_len, input_dim)
        self.inter_states, _ = self.lstm(seq) # (bs, epi_len, 2 * hidden_dim) # bi-lstm
        #TODO: add relu before sending it to self.linear, while relu is not used in Tianshou's RNN policy network
        logit_seq = self.linear(self.inter_states) # (bs, epi_len, context_dim) # linear
        self.logits = torch.mean(logit_seq, dim=1) # (bs, context_dim) # average pooling since the bi-LSTM is location-invariant
        #TODO: mean = self.context_limit * torch.tanh(self.logits)
        #TODO: a seperate std network conditioned on state like in Tianshou
        mean, logstd = self.logits, self.a_log_std.expand_as(self.logits)
        return mean, logstd # TODO: clamp

    def log_prob_context(self, seq, cnt):
        mean, logstd = self.forward(seq)
        # cnt_ori = torch.arctanh(cnt/self.context_limit) * self.context_limit
        # unbounded_log_prob = (-((cnt_ori - mean) ** 2) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)
        #TODO: overlook the correction from the tanh clamp
        # check eq. 21 in SAC paper
        # tanh_correction = torch.log(1-(cnt/self.context_limit).pow(2.0) + 1e-6).sum(dim=-1, keepdim=True)
        # log_prob = unbounded_log_prob - tanh_correction
        log_prob = (-((cnt - mean) ** 2) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)
        return log_prob

    def sample_context(self, seq, fixed=False): # used only for predicting the context for the expert trajectories
        # this is truncated since the true context is also truncated
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
            assert seq.shape[0] == 1
            input_list.append(seq[:, :min_traj_len, :])
        input_tensor = torch.cat(input_list, dim=0) # (bs, seq_len, input_dim)
        cnt = self.sample_context(input_tensor, fixed=fixed)

        return cnt








