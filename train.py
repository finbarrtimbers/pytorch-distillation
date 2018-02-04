import argparse
import utils
import networks

def get_flags():
    arg_parser = argparse.ArgumentParser(
        description="Parser for distillation experiment.")
    arg_parser.add_argument("--dataset", type=str, default="Cifar10Normal")
    arg_parser.add_argument("--network", type=str, default="deep")
    args = arg_parser.parse_args()
    return args

def main():
    flags = get_flags()
    if flags.network == "deep":
        net = networks.deep_resnet
    elif flags.network == "shallow":
        net = networks.shallow_resnet
    train, val, test = get_datasets(flags.dataset)



if __name__ == "__main__":
    main()
