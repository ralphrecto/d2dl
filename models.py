import torch.nn as nn

def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)
        nn.init.normal_(m.bias, std=0.1)

class LinearRegression(nn.Sequential):
    def __init__(self, num_outputs: int = 1):
        super().__init__(
            nn.LazyLinear(num_outputs),
            nn.ReLU()
        )

class SoftmaxRegression(nn.Sequential):
    def __init__(self, input_d: int, output_d: int):
        super().__init__(
            nn.Linear(input_d, output_d)
            # nn.CrossEntropyLoss takes as input unnormalized logits
        )

class SoftmaxRELURegression(nn.Sequential):
    def __init__(self, input_d: int, output_d: int):
        super().__init__(
            nn.Linear(input_d, output_d),
            nn.ReLU()
        )