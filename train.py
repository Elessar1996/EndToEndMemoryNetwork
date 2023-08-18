import argparse
import trainer
import msa_trainer

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--dataset_dir", type=str, default="data/data/en-10k")
    parser.add_argument("--task", type=int, default=20)  #default 1
    parser.add_argument("--max_hops", type=int, default=1)#default 3
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay_interval", type=int, default=25)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--max_clip", type=float, default=40.0)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    model = t.fit()

    return model

def main_msa(config):
    t = trainer.Trainer(config)
    model = t.fit()

    return model




if __name__ == "__main__":
    config = parse_config()
    main(config)