from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch

def get_train_loader(train_dataset, batch_size, num_workers, seed, drop_last):
    num_train_active = len(torch.nonzero(torch.tensor([data.y for data in train_dataset])))
    num_train_inactive = len(train_dataset) - num_train_active
    print(f'training # of molecules: {len(train_dataset)}, actives: {num_train_active}')

    train_sampler_weight = torch.tensor([(1. / num_train_inactive)
                                         if data.y == 0
                                         else (1. / num_train_active)
                                         for data in
                                         train_dataset])

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = WeightedRandomSampler(weights=train_sampler_weight,
                                          num_samples=len(
                                              train_sampler_weight),
                                          generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return train_loader

def get_test_loader(test_dataset, batch_size, num_workers, seed):
    num_test_active = len(torch.nonzero(torch.tensor([data.y for data in test_dataset])))
    print(f'test # of molecules: {len(test_dataset)}, actives: {num_test_active}')
    generator = torch.Generator()
    generator.manual_seed(seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader



