import torch.nn as nn


class MiniLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)

    

    def forward(self, x):
        logit = self.conv1(x)
        return logit


def get_model(config):
    model = eval(config.model.name)(config)
    return model


if __name__ == "__main__":
    model = MiniLLM()
    