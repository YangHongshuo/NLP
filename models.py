"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File

Hongshuo Yang
hy2712
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

torch.manual_seed(91)

class DenseNetwork(nn.Module):
    def __init__(self, embeddings, embedding_dim, num_labels):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.layer1 = nn.Linear(embedding_dim, 16)
        self.layer2 = nn.Linear(16, num_labels)

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        ##
        embeds = self.embeddings(x)
        embeds_sentence = torch.sum(embeds, dim=1)
        out1 = F.relu(self.layer1(embeds_sentence))
        out2 = self.layer2(out1)
        return out2

##
class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, embedding_dim, num_labels):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=32, num_layers=2, batch_first=True, dropout=0.1)
        self.outLayer = nn.Linear(32, num_labels)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class

        #remove padding from x
        lengths = torch.count_nonzero(x, dim=1)
        embeds = self.embeddings(x)
        packed_embeds = rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        hidden_states = self.lstm(packed_embeds)[1][0]
        hidden_2 = hidden_states[1]
        out2 = self.outLayer(hidden_2)
        # print(out2.shape)
        return out2

# extension-grading
# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings, embedding_dim, num_labels):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.embeddings = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(3, embedding_dim))
        # self.maxpool = nn.MaxPool1d(kernel_size=89, stride=1)
        self.linear = nn.Linear(30, num_labels)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        embeds = self.embeddings(x)
        embeds = torch.unsqueeze(embeds, dim=1)
        conv = self.cnn(embeds)
        conv = torch.squeeze(conv)
        out1 = F.relu(conv)
        kernel_size = out1.shape[2]
        pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        out2 = pool(out1)
        out2 = torch.squeeze(out2)
        out3 = self.linear(out2)
        return out2


