# ==
# Test training script for SAB
#
# Anthony G. Chen
# ==

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from sab_nn import *


def get_data_copying_task(mem_seq_len: int = 4,
                          emp_seq_len: int = 8,
                          n_data: int = 512,
                          batch_size: int = 16):
    """
    Generate data for the sequence memorization task

    :param mem_seq_len: the length of sequence to be memorized
    :param emp_seq_len: the length of delay before recall
    :param n_data: total size of dataset to generate
    :param batch_size: minibatch size, n_data should be divisible in this
    :return: training data X and Y both with the following shape:
             (num batches, sequence length, minibatch size)
                num batches = n_data / batch_size
                sequence length = (2 * mem_seq_len) + emp_seq_len
                minibatch size = batch_size
    """

    assert n_data % batch_size == 0

    # ==
    # Set range of value to be generated
    mem_range = (1, 9)
    marker_value = 9

    # ==
    # Generate sequences
    # Generate the sequence to be memorized
    mem_seq = np.random.randint(low=mem_range[0], high=mem_range[1],
                                size=(n_data, mem_seq_len))
    # Generate empty sequences in the middle
    zeros1 = np.zeros((n_data, emp_seq_len - 1))
    zeros2 = np.zeros((n_data, emp_seq_len))
    marker = marker_value * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, mem_seq_len))

    # Concatenate the partial sequences to get the full data sequence
    # NOTE: Rosemary had a set type to int32 and int64 (respectively) here,
    #       might also have to include (potential TODO)
    # Shape: (sample_size, seq_len)
    X = np.concatenate((mem_seq, zeros1, marker, zeros3), axis=1)
    Y = np.concatenate((zeros3, zeros2, mem_seq), axis=1)

    # ==
    # Reshape based on batch size and return
    X = X.reshape((X.shape[0] // batch_size), batch_size, X.shape[1])
    Y = Y.reshape((Y.shape[0] // batch_size), batch_size, Y.shape[1])

    # Reshape to (full batch, seq_len, batch size)
    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)

    return X, Y


def train(args: argparse.ArgumentParser, device='cpu', logger=None):
    # ===
    # Get data
    # TODO change this to a pytorch dataloader in the future
    mem_seq_len = 8
    emp_seq_len = 64
    total_num_data = 100
    minibatch_size = 20

    train_X, train_Y = get_data_copying_task(mem_seq_len=mem_seq_len,
                                             emp_seq_len=emp_seq_len,
                                             n_data=total_num_data,
                                             batch_size=minibatch_size)
    print('data shape', np.shape(train_X))

    # ===
    # Initialize model

    # TODO NOTE input size can only be 1 b.c. of data generated is
    #      only scalar. potentially change for future
    nn_input_size = 1
    num_classes = 10

    if args.model == "sparseattn":
        rnn = self_LSTM_sparse_attn(nn_input_size, args.rnn_dim, args.rnn_layers,
                                    num_classes,
                                    truncate_length=args.trunc,
                                    top_k=args.topk,
                                    remem_every_k=args.attk,
                                    print_attention_step=100)  # , block_attn_grad_past=True)
    elif args.model == "lstm":
        # TODO modify because of changes in the batch configuration
        # will throw error since I have not changed the RNN setting yet
        rnn = RNN_LSTM(nn_input_size, args.rnn_dim, args.rnn_layers, num_classes)

    else:
        raise NotImplementedError

    rnn.to(device)

    # ==
    criterion = nn.CrossEntropyLoss()
    l2_crit = nn.MSELoss()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)

    # ==
    # Train
    for epoch in range(args.num_epochs):

        epoch_loss = 0.0
        epoch_acc_sum = 0.0
        epoch_step = 0

        # TODO: randomize sampling?
        for batch_idx in range(len(train_X)):
            # ==
            # Get and format training data
            x = train_X[batch_idx]
            y = train_Y[batch_idx]

            # Cast X to torch tensor
            x = np.asarray(x, dtype=np.float32)
            t_x = torch.from_numpy(x)
            t_x = t_x.view(t_x.size()[0], t_x.size()[1], nn_input_size)
            t_x = t_x.to(device)

            # Cast Y to tensor
            y = np.asarray(y, dtype=np.float32)
            t_y = torch.from_numpy(y)
            t_y = t_y.view(t_y.size()[0], t_y.size()[1], nn_input_size)
            t_y = t_y.long().to(device)  # long for class index

            # NOTE: t_x and t_y are of shape (seq_len, minibatch, input_size)

            optimizer.zero_grad()

            outputs, attn_w_viz = rnn(t_x)

            outputs_reshp = outputs.view(-1, num_classes)
            labels_reshp = t_y.reshape(-1)

            loss = criterion(outputs_reshp, labels_reshp)

            # optimize stuff
            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), args.clipnorm)
            optimizer.step()

            # ==
            # Logs

            # Compute accuracy
            # TODO seq len should be the first dim
            acc = (outputs.max(dim=2, keepdim=True)[1][-mem_seq_len:, :, :]
                   == t_y[-mem_seq_len:, :, :]).sum()
            acc = acc * 1.0 / (minibatch_size * mem_seq_len)

            # Log stuff
            epoch_loss += loss.item()
            epoch_acc_sum += acc
            epoch_step += 1

        # Logging
        if logger is not None:
            # Add reward
            logger.add_scalar('Loss/Training', epoch_loss / epoch_step,
                              global_step=epoch)
            logger.add_scalar('Acc/Training', epoch_acc_sum / epoch_step,
                              global_step=epoch)
            # TODO should probably also log time

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Total loss: {epoch_loss}, '
                      f'Avg loss: {epoch_loss / epoch_step}, '
                      f'Acc:{epoch_acc_sum / epoch_step}')

        else:
            print(f'Epoch: {epoch}, Total loss: {epoch_loss}, '
                  f'Avg loss: {epoch_loss / epoch_step}, '
                  f'Acc:{epoch_acc_sum / epoch_step}')


if __name__ == "__main__":
    # ===
    # Argparse
    parser = argparse.ArgumentParser(description='SAB')

    #
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')

    # Training data set-up

    # TODO write these?

    # Model and config
    parser.add_argument("--model", default="sparseattn", type=str,
                        choices=["sparseattn", "sparseattn_predict", "trunc", "baseline", "lstm"],
                        help="Model Selection.")

    parser.add_argument("--rnn_dim", default=128, type=int,
                        help="RNN hidden state size")

    parser.add_argument("--rnn_layers", default=2, type=int,
                        help="Number of RNN layers")

    parser.add_argument("--attk", default=2, type=int,
                        help="Attend every K timesteps")

    parser.add_argument("--topk", default=10, type=int,
                        help="Attend only to the top K most important timesteps.")

    # Optimization parameters
    parser.add_argument("--trunc", default=10, type=int,
                        help="Truncation length")

    parser.add_argument("-T", default=200, type=int,
                        help="Copy Distance")

    parser.add_argument("--clipnorm", "--cn", default=1.0, type=float,
                        help="The norm of the gradient will be clipped at this magnitude.")

    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate")

    # TODO NOTE not used. Change to augment network size in the future?
    parser.add_argument('--input_size', type=int, default=1,
                        help='Input dimension of RNN')

    parser.add_argument("--predict_m", default=20, type=int,
                        help="predict m steps forward for hidden states")

    # Logging
    parser.add_argument("--log_dir", default=None, type=str,
                        help="Location of tensorboard logs")

    args = parser.parse_args()
    print(args)

    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Logging
    if args.log_dir is not None:
        # Tensorboard logger
        logger = SummaryWriter(log_dir=args.log_dir)

    else:
        logger = None

    # ==
    # hmm
    train(args, device=device, logger=logger)
