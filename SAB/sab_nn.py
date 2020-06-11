# ============================================================================
# Sparse attentive backtracking module, adopted for python 3 and made to
# work like another pytorch RNN layer.
#
# Original author: Rosemary Ke (https://github.com/nke001)
# Original repository: nke001/sparse_attentive_backtracking_release
#
# Note: wanted to implement SAB with both LSTM and GRU, but a direct GRU
#       implementation results in gradient explosion. So will keep to just
#       LSTM for now.
#
# Author: Anthony G. Chen
# ============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AttentionMemoryCell(nn.Module):
    def __init__(self, hidden_size, top_k=5,
                 block_attn_grad_past=False):
        super(AttentionMemoryCell, self).__init__()

        self.hidden_size = hidden_size
        self.top_k = top_k
        self.block_attn_grad_past = block_attn_grad_past  # TODO where to initialize this?

        self.tanh = torch.nn.Tanh()
        self.w_t = nn.Parameter(torch.Tensor(self.hidden_size * 2, 1))
        nn.init.normal_(self.w_t, mean=0.0, std=0.01)

    def sparse_attention(self, attn_raw):
        """
        Sparsify the attention
        :param attn_raw: the original (non-sparse) attention weights,
                         shape (mem_size, batch)
        :return: attn_w: the attention weights where only the top-k are
                         non-zero, normalized to sum to one;
                         shape (mem_size, batch)
        """
        eps = 10e-8
        cur_mem_size = attn_raw.size(0)

        # Compute the k-th largest attention weight
        # delta is of size (1, batch)
        if cur_mem_size <= self.top_k:
            # Compute the min if mem_size <= k
            delta = torch.min(attn_raw, dim=0,
                              keepdim=True)[0] + eps
        else:
            delta = torch.topk(attn_raw, self.top_k, dim=0,
                               sorted=True)[0][-1:, :] + eps

        # Linearly shift the top-k weights to be non-zero, set rest to zero
        attn_w = attn_raw - delta.repeat(cur_mem_size, 1)
        attn_w = torch.clamp(attn_w, min=0)  # (mem_size, batch)

        # Normalize weights to sum to one
        attn_w_sum = torch.sum(attn_w, dim=0, keepdim=True)  # (1, batch)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(cur_mem_size, 1)

        return attn_w_normalize

    def forward(self, h_t, mem):
        """
        Attention operation between a hidden state and memory buffer to
        produce a memory tensor

        :param h_t: hidden tensor of shape (batch, hidden_dim)
        :param mem: memory buffer of shape (mem_size, batch, hidden_dim)
        :return: m_t: memory tensor of shape (batch, hidden_dim)
        """

        # ==
        # Compute non-sparse attention
        cur_mem_size = mem.size(0)
        batch_size = mem.size(1)

        # Repeat h_t from (batch, hid_dim) to (mem_size, batch, hid_dim)
        #   one for each memory slot
        h_repeated = (h_t.clone().unsqueeze(0)
                      .repeat(cur_mem_size, 1, 1))

        # Concat to [h_t, memory], shape (mem_size, batch, hid_dim*2)
        mlp_h_attn = torch.cat((h_repeated, mem), dim=2)

        # (Optional) block past attention gradient
        if self.block_attn_grad_past:
            mlp_h_attn = mlp_h_attn.detach()

        # Compute attention via tanh then matmul with weight matrix
        # Weight matrix has shape (hid_dim * 2 , 1)
        # Output (attn_w) has shape (mem_size, batch, 1)
        mlp_h_attn = self.tanh(mlp_h_attn)
        attn_w = torch.matmul(mlp_h_attn, self.w_t)

        # ==
        # Sparsify attention: set top-k to non-zero and normalize
        attn_w = attn_w.view(cur_mem_size, batch_size)
        attn_w = self.sparse_attention(attn_w)
        attn_w = attn_w.view(cur_mem_size, batch_size, 1)

        # ==
        # Extract from memory using attention
        attn_w_rep = attn_w.repeat(1, 1, self.hidden_size)
        h_mem_w = attn_w_rep * mem  # (mem_size, batch, hid_dim)

        # Attention-extracted memory information
        m_t = torch.sum(h_mem_w, dim=0)  # (batch, hid_dim)

        return m_t


class SAB_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers,
                 truncate_length=100,
                 remem_every_k=1,
                 k_top_attn=5,
                 block_attn_grad_past=False,
                 batch_first=False):
        super(SAB_LSTM, self).__init__()

        # Network attribute
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Sparse backprop attribute
        self.truncate_length = truncate_length
        self.remem_every_k = remem_every_k
        self.k_top_attn = k_top_attn
        self.block_attn_grad_past = block_attn_grad_past

        if batch_first:
            raise NotImplementedError

        # ==
        # Initialization
        self.rnnCell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Sparse attention
        self.attnMemCell = AttentionMemoryCell(hidden_size=self.hidden_size,
                                               top_k=self.k_top_attn)

    def forward(self, input, hx=None, mem=None):
        """
        Run LSTM with sparse attentive backtracking
        :param input: Tensor, shape (seq_len, batch, input_size)
        :param hx: Tuple of (h_0, c_t)
                       h_0: shape (1, batch, hidden_size), init hidden state
                       c_0: shape (1, batch, hidden_size), init cell state
        :param mem: Tensor, shape (mem_size_0, batch, hidden_size),
                    init memory buffer tensor
        :return: out:   shape (seq_len, batch, hidden_size)
                        output tensor
                 Tuple of (h_final, c_final):
                        both of size (1, batch, hidden_size)
                        final hidden and cell states
                 mem:   final memory buffer,
                        shape (mem_size_final, batch, hidden_size)

        NOTE: h_t by standard have (layer*direction, batch, hidden_size),
              but this is not implemented here. only the dimension
              convention is kept
        """

        # Sizes
        seq_length = input.size(0)
        batch_size = input.size(1)
        input_size = input.size(2)

        # Unpack recurrent and memory states
        if hx is None:
            # TODO add initialization, with the right device
            raise NotImplementedError
        else:
            h_t, c_t = hx
            h_t = h_t.view(batch_size, self.hidden_size)
            c_t = c_t.view(batch_size, self.hidden_size)

        if mem is None:
            raise NotImplementedError

        # Output lists
        h_t_seq_list = []
        m_t_seq_list = []

        # TODO add attention visualization?

        # Iterate over sequence
        for i, x_t in enumerate(input.chunk(seq_length, dim=0)):
            # ==
            # Input and one cycle of GRU
            x_t = x_t.contiguous().view(batch_size, input_size)

            # Feed GRU, generate hidden and cell states,
            # h_t and c_t both have size (batch, hid_dim)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.rnnCell(x_t, (h_t.detach(), c_t.detach()))
            else:
                h_t, c_t = self.rnnCell(x_t, (h_t, c_t))

            # ==
            # Get memory
            m_t = self.attnMemCell.forward(h_t, mem)  # (batch, hid_dim)

            # Feed attn_c to hidden state h_t
            h_t += m_t  # (batch, hid_dim)

            # ==
            # At regular intervals, remember a hidden state
            if (i + 1) % self.remem_every_k == 0:
                mem = torch.cat((mem, h_t.view(1, batch_size,
                                               self.hidden_size)), dim=0)

            # Record outputs
            h_t_seq_list += [h_t]
            m_t_seq_list += [m_t]

        # ==
        # Compute output values
        h_t_seq = torch.stack(h_t_seq_list,
                              dim=0)  # (seq_len, batch, hid_dim)
        m_t_seq = torch.stack(m_t_seq_list,
                              dim=0)  # (seq_len, batch, hid_dim)

        out_seq = torch.cat((h_t_seq, m_t_seq),
                            dim=2)  # (batch, seq_len, 2 * hid_dim)

        out = out_seq.contiguous(). \
            view(-1, out_seq.size(2))  # (seq_len * batch, 2 * hid_dim)
        out = self.fc(out)
        out = out.view(seq_length, batch_size, self.output_size)

        # Format and out
        h_final = h_t.view(1, batch_size, self.hidden_size)
        c_final = c_t.view(1, batch_size, self.hidden_size)

        return out, (h_final, c_final), mem
