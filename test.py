from Dataset import bAbIDataset
import random
import torch

if __name__ == "__main__":


    # dataset_dir = 'data/en'
    #
    # dataset = bAbIDataset(dataset_dir=dataset_dir, task_id=20)
    #
    #
    #
    #
    #
    # random_indexes = random.sample([i for i in range(len(dataset))], k=5)
    #
    #
    # for r in random_indexes:
    #
    #     print(f'for random index {r}, data={dataset[r]}')
    #
    # print(dataset.vocab)
    # print(f'length of vocab: {len(dataset.vocab)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)