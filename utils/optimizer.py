import torch.optim as optim


def get_optimizer(config, model):
    optimizer = eval(config.optimizer.name)(model.parameters())
    return optimizer