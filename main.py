from train import main, parse_config, main_msa


if __name__ == '__main__':
    config = parse_config()
    model = main(config)