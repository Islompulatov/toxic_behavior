from torch import nn
import torch.nn.functional as F
from torch import optim
import torch


class Classifier(nn.Module):
    def __init__(self, max_seq_len, emb_dim, hidden1=16, hidden2=16):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(max_seq_len*emb_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 6)
    
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# MAX_SEQ_LEN = 32
# model = Classifier(MAX_SEQ_LEN, 300, 16, 16)

# test = torch.rand((32,32*300))


# out = model.forward(test)
# print(out)