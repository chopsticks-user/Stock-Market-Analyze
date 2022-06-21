from functools import total_ordering
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
import time

class fcnn(nn.Module):
    def __init__(self, in_features_dim = 1, lstm_layers_dim =  1, out_features_dim = 1):
        super().__init__()
        # self.lstm = nn.LSTM(in_features_dim, lstm_layers_dim, 2)
        self.fc1 = nn.Linear(in_features_dim, lstm_layers_dim)
        self.fc2 = nn.Linear(lstm_layers_dim, lstm_layers_dim)
        self.fc = nn.Linear(lstm_layers_dim, out_features_dim)
        self.loss = nn.L1Loss(reduction = "sum")
        self.optimizer = O.Adam(self.parameters())
        self.device = T.device("cuda")

        self.last = None

        self.to(self.device)

    def forward(self, x):
        # lstm_out, _ = self.lstm(x)
        # fc_out = self.fc(lstm_out)
        # soft_max = F.softmax(fc_out)
        # self.last = soft_max
        # return soft_max

        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = self.fc(x)
        self.last = x
        return x

    def backward(self, target):
        self.zero_grad()
        loss = self.loss(self.last, target)
        loss.backward()
        self.optimizer.step()
        return loss


n = fcnn(1, 16, 1)
cnt = 0
total_loss = 0.0
while True:
    i = np.random.rand(1) * 10
    input = T.tensor([i], dtype = T.float).to(n.device)
    target = T.tensor([i ** 3 + 2 * i **2 + 4], dtype = T.float).to(n.device)
    val = n.forward(input)
    loss = n.backward(target)
    total_loss += loss
    if cnt % 1000 == 0:
        print(f"Step {cnt}, Predicted: {val}, Actual: {target}, Average Loss: {total_loss / 1000}")
        total_loss = 0.0
    cnt += 1
    # time.sleep(0.5)


# input = T.tensor(np.array([-.05, -.07, .02, .01, .03, .04]), dtype = T.float).to(n.device)
# target = T.tensor(np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]]), dtype = T.float).to(n.device)

# print(*n.parameters(), end = "\n\n\n\n\n\n\n")

# out = n.forward(input.unsqueeze(1))
# loss = n.backward(target)


# print(n.forward(input.unsqueeze(1)))
# # print(*n.parameters())

# # n.forward(input[1].unsqueeze(0))
# # n.backward(target[1])

# # print(n.forward(input[1].unsqueeze(0)))