# ============================================================================
# Sparse attentive backtracking module, adopted for python 3
#
# Original author: Rosemary Ke (https://github.com/nke001)
# Original repository: nke001/sparse_attentive_backtracking_release
#
# Author: Anthony G. Chen
# ============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class RNN_LSTM(nn.Module):
    """
    For model comparison
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = []
        h_t = torch.zeros(x.size(0), self.hidden_size,
                          requires_grad=True).cuda()
        c_t = torch.zeros(x.size(0), self.hidden_size,
                          requires_grad=True).cuda()
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp = (outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] * shp[1], self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out, None

    def print_log(self):
        model_name = '_regular-LSTM_'
        model_log = ' Regular LSTM.......'
        return (model_name, model_log)


class RNN_LSTM_truncated(nn.Module):
    """
    For model comparison
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=1):
        super(RNN_LSTM_truncated, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.truncate_length = truncate_length
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = []
        h_t = torch.zeros(x.size(0), self.hidden_size,
                          requires_grad=True).cuda()
        c_t = torch.zeros(x.size(0), self.hidden_size,
                          requires_grad=True).cuda()

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm(input_t, (h_t.detach(), c_t.detach()))
                # c_t = c_t.detach()
            else:
                h_t, c_t = self.lstm(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp = (outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] * shp[1], self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = '_trun-LSTM_trun_len_' + str(self.truncate_length)
        model_log = ' trun LSTM.....trun_len:' + str(self.truncate_length)
        return (model_name, model_log)


class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        self.top_k = top_k

    def forward(self, attn_raw):
        """
        Sparsify the attention
        :param attn_raw: the original (non-sparse) attention weights,
                         shape (batch, mem_size)
        :return: attn_w: the attention weights where only the top-k are
                         non-zero, normalized to sum to one;
                         shape (batch, mem_size)
        """

        eps = 10e-8
        cur_mem_size = attn_raw.size()[1]

        # Compute the k-th largest attention weight (scalar)
        if cur_mem_size <= self.top_k:
            # Compute the min if mem_size <= k
            delta = torch.min(attn_raw, dim=1,
                              keepdim=True)[0] + eps
        else:
            delta = torch.topk(attn_raw, self.top_k, dim=1,
                               sorted=True)[0][:, -1:] + eps

        # Linearly shift the top-k weights to be non-zero, set rest to zero
        attn_w = attn_raw - delta.repeat(1, cur_mem_size)
        attn_w = torch.clamp(attn_w, min=0)  # (batch, mem_size)

        # Normalize weights to sum to one
        attn_w_sum = torch.sum(attn_w, dim=1, keepdim=True)  # (batch, 1)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, cur_mem_size)

        return attn_w_normalize



class self_LSTM_sparse_attn(nn.Module):
    # TODO: change name of module?

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 truncate_length=100, block_attn_grad_past=False,
                 remem_every_k=1,
                 top_k=5,
                 print_attention_step=1):
        """
        Sparse attentive backtracking
        :param input_dim: input dimension
        :param hidden_dim: hidden state dimension
        :param num_layers: NOTE: not used?
        :param num_classes: number of classes to predict for output layer
        :param truncate_length: length of truncated BPTT
        :param block_attn_grad_past: ?? TODO
        :param remem_every_k: add hidden state to memory every k steps
        :param top_k: number of memories to extract using sparse attention
        :param print_attention_step: ?? TODO
        """
        super(self_LSTM_sparse_attn, self).__init__()

        # Attributes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.truncate_length = truncate_length
        self.block_attn_grad_past = block_attn_grad_past
        self.remem_every_k = remem_every_k
        self.top_k = top_k

        self.print_attention_step = print_attention_step

        # Initialize network components
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.tanh = torch.nn.Tanh()

        self.w_t = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        nn.init.normal_(self.w_t, mean=0.0, std=0.01)
        self.sparse_attn = Sparse_attention(top_k=self.top_k)

    def forward(self, x):
        # device
        device = self.fc.weight.device

        # Size
        batch_size = x.size(0)
        seq_length = x.size(1)  # TODO seq should be dim 0
        input_size = x.size(2)

        # initialize hidden states
        h_t = torch.zeros((batch_size, self.hidden_dim),
                          device=device, requires_grad=True)
        c_t = torch.zeros((batch_size, self.hidden_dim),
                          device=device, requires_grad=True)

        # Will eventually grow to (batch_size, seq_length/k, hidden_size)
        # with more and more concatenations.
        h_mem = h_t.view(batch_size, 1, self.hidden_dim)

        h_t_seq_list = []
        m_t_seq_list = []
        attn_w_viz = []

        for i, input_t in enumerate(x.chunk(seq_length, dim=1)):  # TODO seq is dim 0
            cur_mem_size = h_mem.size(1)  # TODO seq should be dim 0

            # ==
            # Input and one cycle of LSTM
            input_t = input_t.contiguous().view(batch_size, input_size)

            # Feed LSTM, generate hidden and cell states,
            # h_t and c_t both have size (batch, hid_dim)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm(input_t, (h_t.detach(), c_t.detach()))
            else:
                h_t, c_t = self.lstm(input_t, (h_t, c_t))

            # ==
            # Compute raw (non-sparse) attention

            # Repeat h_t over all memory slots: (batch, mem_size, hid_dim)
            h_repeated = (h_t.clone().unsqueeze(1)
                          .repeat(1, cur_mem_size, 1))

            # Concat to [h_t, memory], shape (batch, mem_size, hid_dim*2)
            mlp_h_attn = torch.cat((h_repeated, h_mem), 2)

            # (Optimal) block past attention gradient
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            # Compute attention via tanh then matmul with weight matrix
            # Weight matrix has shape (hid_dim * 2 , 1)
            # Output (attn_w) has shape (batch, mem_size, 1)
            mlp_h_attn = self.tanh(mlp_h_attn)
            attn_w = torch.matmul(mlp_h_attn, self.w_t)

            # ==
            # Sparsify attention: set top-k to non-zero and normalize

            attn_w = attn_w.view(batch_size, cur_mem_size)
            attn_w = self.sparse_attn(attn_w)
            attn_w = attn_w.view(batch_size, cur_mem_size, 1)

            # ==
            # Extract from memory using attention

            attn_w_rep = attn_w.repeat(1, 1, self.hidden_dim)
            h_mem_w = attn_w_rep * h_mem  # (batch, mem_size, hid_dim)

            # Attention-extracted memory information
            m_t = torch.sum(h_mem_w, dim=1)  # (batch, hid_dim)

            # Feed attn_c to hidden state h_t
            h_t += m_t  # (batch, hid_dim)

            # ==
            # At regular intervals, remember a hidden state
            if (i + 1) % self.remem_every_k == 0:
                h_mem = torch.cat((h_mem, h_t.view(batch_size,
                                                   1, self.hidden_dim)), dim=1)

            # Record outputs
            h_t_seq_list += [h_t]
            m_t_seq_list += [m_t]

            # TODO: not sure what this is doing.
            if self.print_attention_step >= (seq_length - i - 1):
                attn_w_viz.append(attn_w.mean(dim=0).view(cur_mem_size))

        # ==
        # Compute per-timestep output
        #   y_t = V_1 * h_t + V_2 * m_t + b

        # Tensors: full sequence of hidden states and per-time memory
        # TODO seq is dim 0
        h_t_seq = torch.stack(h_t_seq_list,
                              dim=1)  # (batch, seq_len, hid_dim)
        m_t_seq = torch.stack(m_t_seq_list,
                              dim=1)  # (batch, seq_len, hid_dim)
        out_seq = torch.cat((h_t_seq, m_t_seq),
                            dim=2)  # (batch, seq_len, 2 * hid_dim)

        # Compute per-timestep output
        # TODO seq is dim 0
        out = out_seq.contiguous().view(-1, out_seq.size(2))  # (batch * seq_len, 2 * hid_dim)
        out = self.fc(out)
        out = out.view(batch_size, seq_length, self.num_classes)

        # Return values
        #   out: output values (batch, seq_len, num_classes)
        #   attn_w_viz: attention weights for visualization
        return out, attn_w_viz

    def print_log(self):
        """TODO potentially refactor"""
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_' + str(
            self.truncate_length) + 'attn_everyk_' + str(
            self.attn_every_k)  # + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k) + '....attn_everyk_' + str(
            self.attn_every_k) + '.....truncate_length:' + str(self.truncate_length)
        return (model_name, model_log)





def attention_visualize(attention_timestep, filename):
    # visualize attention
    plt.matshow(attention_timestep)
    filename += '_attention.png'
    plt.savefig(filename)

