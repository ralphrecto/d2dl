import torch.nn as nn

def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
        nn.init.normal_(m.bias)

class LinearRegression(nn.Module):
    def __init__(self, num_inputs: int = 1):
        super().__init__()
        self.net = nn.LazyLinear(num_inputs)

        # initialize weights
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, x):
        return self.net(x)

class SoftmaxRegression(nn.Sequential):
    def __init__(self, input_d: int, output_d: int):
        super().__init__(
            nn.Linear(input_d, output_d, bias=True)
            # nn.CrossEntropyLoss takes as input unnormalized logits
        )

        self.apply(init_weights)