from train import main, parse_config


if __name__ == '__main__':
    config = parse_config()
    main(config)