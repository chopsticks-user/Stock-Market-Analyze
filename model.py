import numpy as np
import pandas as pd
import torch as T
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
import time
from indicators import Indicators
from tools import change, lin_reg_model

class LSTMModel(T.nn.Module):
    def __init__(self, in_features_dim, hidden_dim, out_features_dim, file_path = None):
        super(LSTMModel, self).__init__()
        self.file_path = file_path
        self.lstm = T.nn.LSTM(input_size = in_features_dim, hidden_size = hidden_dim, num_layers = 3)
        self.out_features = T.nn.Linear(in_features = hidden_dim, out_features = out_features_dim)
        self.optimizer = T.optim.SGD(self.parameters(), lr = 0.1)
        self.loss = T.nn.HuberLoss()

        self.last_out_features = None

        self.device = T.device("cuda")
        self.to(self.device)

    def forward(self, in_features):
        in_features.to(self.device)
        lstm_out, _ = self.lstm(in_features)
        out_features = self.out_features(lstm_out)
        self.last_out_features = out_features[-1]
        return self.last_out_features

    def backward(self, target):
        self.zero_grad()
        loss = self.loss(self.last_out_features, target)
        loss.backward()
        self.optimizer.step()
        return loss

