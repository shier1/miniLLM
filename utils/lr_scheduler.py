from torch.optim.lr_scheduler import CosineAnnealingLR



def get_lr_scheduler(config):
    scheduler = eval(config.lr_scheduler.name)()
    return scheduler