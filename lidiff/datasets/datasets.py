import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
#!
from lidiff.datasets.dataloader.SemanticKITTITemporal_av2 import HDF5Dataset
from lidiff.utils.collations import SparseSegmentCollation_av2
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule']

class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    # def train_dataloader(self):
    #     collate = SparseSegmentCollation()

    #     data_set = TemporalKITTISet(
    #         data_dir=self.cfg['data']['data_dir'],
    #         seqs=self.cfg['data']['train'],
    #         split=self.cfg['data']['split'],
    #         resolution=self.cfg['data']['resolution'],
    #         num_points=self.cfg['data']['num_points'],
    #         max_range=self.cfg['data']['max_range'],
    #         dataset_norm=self.cfg['data']['dataset_norm'],
    #         std_axis_norm=self.cfg['data']['std_axis_norm'])
    #     loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
    #                         num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
    #     return loader

    def train_dataloader(self):
        collate = SparseSegmentCollation_av2()

        data_set = HDF5Dataset( #! change for av2
            directory=self.cfg['data']['data_dir'],
            # seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            # resolution=self.cfg['data']['resolution'],
            # num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'])
            # dataset_norm=self.cfg['data']['dataset_norm'],
            # std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, pin_memory=True)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation_av2()

        data_set = HDF5Dataset(
            directory=self.cfg['data']['data_dir'],
            # seqs=self.cfg['data']['validation'],
            split='val',
            # resolution=self.cfg['data']['resolution'],
            # num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'])
            # dataset_norm=self.cfg['data']['dataset_norm'],
            # std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate,pin_memory=True)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation_av2()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

dataloaders = {
    'KITTI': TemporalKittiDataModule,
}

