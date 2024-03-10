import torch
import numpy as np

from utils.utils import Trainer
import argparse
import yaml


def seed_init(init_seed):
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed) # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/transformer.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()
    
    print(f'Starting experiment with configurations in {args.config_filename}...')
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_init(args.seed)

    trainer = Trainer(args, device)
    if args.running_mode == "train":
        trainer.train()
        trainer.test()
    elif args.running_mode == "test":
        trainer.test()







