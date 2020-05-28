from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_loader):
                labels = batch[1]
                output = model.forward(batch[0])
                loss = self.loss_func(output, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()

                if (((i+1) * len(labels)) % log_nth == 0):
                    current_iteration = i + 1
                    total_iterations = len(train_loader)
                    batch_size = batch[0].shape[0]
                    print("[Iteration ", batch_size * current_iteration, "/",
                          batch_size * total_iterations, "] TRAIN loss: ", loss.data.item())

            epoch_train_batch = next(iter(train_loader))
            epoch_val_batch = next(iter(val_loader))

            train_labels = epoch_train_batch[1]
            val_labels = epoch_val_batch[1]

            train_output = model.forward(epoch_train_batch[0])
            val_output = model.forward(epoch_val_batch[0])

            train_loss = self.loss_func(train_output, train_labels)
            val_loss = self.loss_func(val_output, val_labels)

            _, train_predicted = torch.max(train_output, 1)
            train_total = len(train_labels)
            train_correct = (train_predicted == train_labels).sum()
            train_accuracy = train_correct.float() / train_total

            _, val_predicted = torch.max(val_output, 1)
            val_total = len(val_labels)
            val_correct = (val_predicted == val_labels).sum()
            val_accuracy = val_correct.float() / val_total

            print("[Epoch ", epoch + 1, "/", num_epochs, "] TRAIN acc/loss: ",
                  train_accuracy.data.item(), "/", train_loss.data.item())
            print("[Epoch ", epoch + 1, "/", num_epochs, "] VAL acc/loss: ",
                  val_accuracy.data.item(), "/", val_loss.data.item())

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_accuracy)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_accuracy)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
