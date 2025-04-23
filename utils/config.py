from yacs.config import CfgNode as CN


def get_config(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        config = CN.load_cfg(yaml_file)
    config.freeze()
    return config

if __name__ == "__main__":
    config = get_config("./configs/miniLLM.yaml")
    print(config)