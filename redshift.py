# create equivariant network using e3nn
import e3nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import GatedBlock, GatedBlockParity, NormActivation, Sequential, Kernel, Convolution
from e3nn.non_linearities.rescaled_act import rescaled_act
from e3nn.non_linearities.gated_block import GatedBlock

class RedshiftNet(nn.Module):
    def __init__(self, num_classes, num_channels, num_layers, num_nodes, num_features, num_radial, num_spherical, num_blocks, act, dropout):
        super(RedshiftNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.num_blocks = num_blocks
        self.act = act
        self.dropout = dropout
    
        # input layer
        self.input = nn.Sequential(
            nn.Linear(self.num_features, self.num_nodes),
            nn.BatchNorm1d(self.num_nodes),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(self.num_layers):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(self.num_nodes, self.num_nodes),
                    nn.BatchNorm1d(self.num_nodes),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
            )
    
        # output layer
        self.output = nn.Sequential(
            nn.Linear(self.num_nodes, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.input(x)
        for i in range(self.num_layers):
            x = self.hidden[i](x)
        x = self.output(x)
        return x
    
# create the network and train it on random data
net = RedshiftNet(num_classes=3, num_channels=1, num_layers=3, num_nodes=64, num_features=10, num_radial=3, num_spherical=3, num_blocks=3, act='relu', dropout=0.5)
x = torch.randn(100, 10)
y = net(x)
print(y.shape)

# train net with gradient descent
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for i in range(100):
    optimizer.zero_grad()
    y_pred = net(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(loss)

# save the network  
torch.save(net.state_dict(), 'redshift.pt')
