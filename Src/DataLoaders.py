                                                  

import torch
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import warnings
import logging
from pathlib import Path

class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test, dim_image=128, train=False, data_root=None):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        project_root = Path(__file__).resolve().parent.parent
        default_data_root = project_root / 'Data'
        self.data_root = Path(data_root or os.environ.get('EGA_FLOW_DATA_ROOT', default_data_root)).expanduser().resolve()
        self.dim_image = dim_image
        self.train = train

    def load_data(self):
        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
                   
            img_dir = self.data_root / 'celeba' / 'img_align_celeba' / 'img_align_celeba'
            partition_csv = self.data_root / 'celeba' / 'list_eval_partition.csv'

                      
            train_dataset = CelebADataset(img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(img_dir, partition_csv, partition=2, transform=transform)

            shuffle_train = len(train_dataset) > 0

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=shuffle_train,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'afhq_cat':
                                                                
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            img_dir_train = self.data_root / 'afhq_cat' / 'train' / 'cat'
            img_dir_val = self.data_root / 'afhq_cat' / 'val' / 'cat'
            img_dir_test = self.data_root / 'afhq_cat' / 'test' / 'cat'

            train_dataset = AFHQDataset(img_dir_train, batchsize=self.batch_size_test, transform=transform)
            val_dataset = AFHQDataset(img_dir_val, batchsize=self.batch_size_test, transform=transform)
            test_dataset = AFHQDataset(img_dir_test, batchsize=self.batch_size_test, transform=transform)

            shuffle_train = len(train_dataset) > 0

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=shuffle_train,
                collate_fn=custom_collate, drop_last=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        else:
            raise ValueError("EGA-Flow package supports only celeba and afhq_cat")

        data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        return data_loaders


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

                                           
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not Path(img_path).exists():
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class AFHQDataset(Dataset):
                           

    def __init__(self, img_dir, batchsize, category='cat', transform=None):
        all_files = os.listdir(img_dir)
        manifest = Path(img_dir).parent / 'selection_manifest.tsv'
        if manifest.exists():
            ordered_files = []
            available = set(all_files)
            for line in manifest.read_text().splitlines()[1:]:
                cols = line.split('	')
                if len(cols) >= 4 and cols[3] in available:
                    ordered_files.append(cols[3])
            remaining = sorted(available - set(ordered_files))
            self.files = ordered_files + remaining
        else:
            self.files = sorted(all_files)
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not Path(img_path).exists():
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


def custom_collate(batch):
                            

    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)
