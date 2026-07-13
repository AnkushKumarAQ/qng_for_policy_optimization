import torch.nn as nn
import torch.nn.functional as F

class ClassicalPolicyNetwork(nn.Module):
    """
    Classical Multi-Layer Perceptron (MLP) Policy Network.
    Architecture: Linear -> Tanh -> Linear -> Softmax
    """
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(ClassicalPolicyNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.tanh = nn.Tanh() # Tanh is used over ReLU to bound activations and prevent extreme gradients
        self.layer2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        # Dim=-1 ensures probabilities sum to 1 across the action dimension
        action_probs = F.softmax(self.layer2(x), dim=-1)
        return action_probs
        