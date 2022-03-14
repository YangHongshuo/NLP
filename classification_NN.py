"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Main File

Hongshuo Yang
hy2712
"""

## Imports
import sys
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                    (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.
MODEL_PATH = 'model.pth'
##
def train_model(model, loss_fn, optimizer, train_generator, dev_generator, early_stopping = False):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """

    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    # TODO: updating the model parameters with each batch (we suggest you use torch.optim.Adam to start);
    # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
    # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
    # TODO: Make sure to print the dev set loss each epoch to stdout.

    # extension-grading
    # third class stopping criterion from: https://link.springer.com/chapter/10.1007/3-540-49430-8_3
    if early_stopping == True:
        losses = []
        STRIP_SIZE = 5
        SUCCESSION = 3
        loss_increase_count = 0
    else:
        last_loss = 0
    for epoch in range(500):
        for X_b, y_b in train_generator:
            # zero out gradients from old batch
            model.zero_grad()

            # compute predicted values
            y_pred = model(X_b)

            # compute loss
            loss = loss_fn(y_pred.double(), y_b.long())

            # do backward pass and update gradient
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        for X_b, y_b in dev_generator:
            y_pred = model(X_b)
            # update total loss for the epoch
            total_loss += loss_fn(y_pred.double(), y_b.long()).item()
        # print(epoch, total_loss)
        print(total_loss, file=sys.stdout)

        # extension-grading
        if early_stopping:
            losses.append(total_loss)
            if epoch == STRIP_SIZE - 1:
                torch.save(model.state_dict(), MODEL_PATH)
            if epoch >= STRIP_SIZE and (epoch+1)%STRIP_SIZE==0:
                if losses[epoch] < losses[epoch-STRIP_SIZE]:
                    torch.save(model.state_dict(), MODEL_PATH)
                    loss_increase_count = 0
                else:
                    loss_increase_count += 1
                    # print(loss_increase_count)
                    if loss_increase_count == SUCCESSION:
                        break
        else:
            if epoch == 0:
                last_loss = total_loss
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                if total_loss > last_loss:
                    break
                else:
                    last_loss = total_loss
                    torch.save(model.state_dict(), MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))
    return model

##
def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))

##
def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    ## Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    ## Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result

    num_labels = len(LABEL_NAMES)
    if args.model == 'dense':
        ## train and test the dense network model
        dense_model = models.DenseNetwork(embeddings, EMBEDDING_DIM, num_labels)
        dense_optimizer = optim.Adam(dense_model.parameters(), lr=0.001)
        dense_model = train_model(dense_model,loss_fn,dense_optimizer,train_generator,dev_generator)
        ##
        test_model(dense_model,loss_fn,test_generator)
    elif args.model == 'RNN':
        ## train and test the rnn model
        rnn_model = models.RecurrentNetwork(embeddings, EMBEDDING_DIM, num_labels)
        rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.002)
        rnn_model = train_model(rnn_model, loss_fn, rnn_optimizer, train_generator, dev_generator)
        ##
        test_model(rnn_model, loss_fn, test_generator)
    ##
    # extension-grading
    elif args.model == 'extension1':
        ## train and test the model with early stopping
        model = models.DenseNetwork(embeddings, EMBEDDING_DIM, num_labels)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
        model = train_model(model, loss_fn, optimizer, train_generator, dev_generator, True)
        ##
        test_model(model, loss_fn, test_generator)
    ##
    elif args.model == 'extension2':
        ## train and test the rnn model
        model = models.ExperimentalNetwork(embeddings, EMBEDDING_DIM, num_labels)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
        ##
        test_model(model, loss_fn, test_generator)

##
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=False, default='extension1',
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
