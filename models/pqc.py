import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from config import BOUNDS_TORCH


class PQCPolicyNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device('lightning.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
        def qnode(inputs, params):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            for layer in params:
                for i in range(n_qubits):
                    qml.RY(layer[i, 0], wires=i)
                    qml.RZ(layer[i, 1], wires=i)

                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        weight_shapes = {"params": (n_layers, n_qubits, 2)}

        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer('bounds', BOUNDS_TORCH)

    def forward(self, x):
        x_scaled = torch.clamp(x / self.bounds, -1.0, 1.0) * np.pi
        
        return self.softmax(self.qlayer(x_scaled))


def create_qng_qnode(noise_type='analytical'):
    if noise_type == 'noisy':
        dev = qml.device('lightning.qubit', wires=4 + 1, shots=1024)
    else:
        dev = qml.device('lightning.qubit', wires=4 + 1)

    @qml.qnode(dev, interface='autograd', diff_method='parameter-shift')
    def qnode(params, x):
        for i in range(4):
            qml.RY(x[i], wires=i)

        for layer in params:
            for i in range(4):
                qml.RY(layer[i, 0], wires=i)
                qml.RZ(layer[i, 1], wires=i)

            for i in range(4):
                qml.CNOT(wires=[i, (i + 1) % 4])
                
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    return qnode
