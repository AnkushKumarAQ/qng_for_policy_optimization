import torch.nn as nn
import torch.nn.functional as F


class ClassicalPolicyNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=64):
        super(ClassicalPolicyNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        action_probs = F.softmax(self.layer2(x), dim=-1)
        return action_probs
        