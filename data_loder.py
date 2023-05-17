from data_set import CUB
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import config
arg = config.Config.config()
batch_size = arg['batch_size']
time1 = time.time()
path = arg['path']
num_workers = arg['num_workers']

def data_lode():
    transform_train = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize((550, 550)),
            transforms.RandomCrop(448, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transform_test = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize((550, 550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = CUB(
            path,
            train=True,
            transform=transform_train,
            target_transform=None
        )
    test_dataset = CUB(
            path,
            train=False,
            transform=transform_test,
            target_transform=None
        )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers,
                                                   drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers,
                                                  drop_last=False)
    time2 = time.time()
    print(time2-time1)
    return train_dataloader, test_dataloader

