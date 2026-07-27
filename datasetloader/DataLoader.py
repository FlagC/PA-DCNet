import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io


class RadioMapSeerLoader(Dataset):
    """
    RadioMapSeer Data Loader.
    """

    def __init__(self, maps_inds=None, phase="train",
                 ind1=0, ind2=0,
                 thresh=0.0,
                 dir_dataset="./data/RadioMapSeer",
                 numTx=80,
                 carsSimul="no",
                 carsInput="no",
                 fsplInput="no",
                 reverseInput="no",
                 transform=transforms.ToTensor(),
                 augment_cfg=None):
        """
        Args:
            maps_inds (np.ndarray, optional): Shuffled map indices.
            phase (str): "train", "val", or "test".
            ind1, ind2 (int): Start and end indices for custom data splits.
            dir_dataset (str): Root directory of the RadioMapSeer dataset.
            numTx (int): Number of transmitters per map.
            carsSimul (str): "no" or "yes". Use simulation with or without cars.
            carsInput (str): "no" or "yes". Take inputs with or without cars channel.
            fsplInput (str): "no" or "yes". If "yes", replace Tx map with log-distance FSPL-like map.
            transform (callable, optional): Transform to be applied to the images.
        """

        if maps_inds is None:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom
            self.ind1 = ind1
            self.ind2 = ind2

        self.thresh = thresh
        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        self.fsplInput = fsplInput
        self.reverseInput = reverseInput
        self.phase = phase

        # Setup gain directory based on cars simulation
        if carsSimul == "no":
            self.dir_gain = os.path.join(self.dir_dataset, "gain", "DPM")
        else:
            self.dir_gain = os.path.join(self.dir_dataset, "gain", "carsDPM")

        self.dir_buildings = os.path.join(
            self.dir_dataset, "png", "buildings_complete")
        self.dir_Tx = os.path.join(self.dir_dataset, "png", "antennas")

        # Setup cars directory if needed
        if carsInput != "no":
            self.dir_cars = os.path.join(self.dir_dataset, "png", "cars")

        self.transform = transform
        self.height = 256
        self.width = 256

        augment_cfg = augment_cfg or {}
        self.augment = bool(augment_cfg.get("enabled", False)) and phase == "train"
        self.augment_hflip_p = float(augment_cfg.get("hflip_p", 0.0))
        self.augment_vflip_p = float(augment_cfg.get("vflip_p", 0.0))
        self.augment_rot90 = bool(augment_cfg.get("rot90", False))

        # 预生成坐标网格
        yy, xx = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width, dtype=np.float32),
            indexing="ij"
        )
        self.grid_y = yy
        self.grid_x = xx

    def _apply_train_augmentation(self, inputs_numpy, image_gain):
        if not self.augment:
            return inputs_numpy, image_gain

        if self.augment_hflip_p > 0.0 and np.random.rand() < self.augment_hflip_p:
            inputs_numpy = np.flip(inputs_numpy, axis=1)
            image_gain = np.flip(image_gain, axis=1)

        if self.augment_vflip_p > 0.0 and np.random.rand() < self.augment_vflip_p:
            inputs_numpy = np.flip(inputs_numpy, axis=0)
            image_gain = np.flip(image_gain, axis=0)

        if self.augment_rot90:
            k = int(np.random.randint(0, 4))
            if k:
                inputs_numpy = np.rot90(inputs_numpy, k=k, axes=(0, 1))
                image_gain = np.rot90(image_gain, k=k, axes=(0, 1))

        # np.flip/rot90 may create negative-stride views, which ToTensor cannot handle.
        return np.ascontiguousarray(inputs_numpy), np.ascontiguousarray(image_gain)

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def _generate_fspl_like_map(self, tx_binary_map):
        # 基于tx位置生成log10梯度下降的输入通道

        # tx_binary_map: shape [H, W] or [H, W, 1]
        if tx_binary_map.ndim == 3:
            tx_binary_map = tx_binary_map[:, :, 0]

        tx_positions = np.argwhere(tx_binary_map > 0.5)

        tx_y, tx_x = tx_positions[0]

        tx_y = np.float32(tx_y)
        tx_x = np.float32(tx_x)

        # meter per pixel : 1m
        dist = np.sqrt((self.grid_y - tx_y) ** 2 + (self.grid_x - tx_x) ** 2)

        # 只保留 log 距离梯度；绝对常数项和频率项对归一化后的空间形状没有作用
        dist_safe = np.maximum(dist, 1.0)

        # 负的 log-distance 衰减：距离越近值越大，越远值越小
        fspl_like = -np.log10(dist_safe)

        # 归一化到 [0,1]
        vmin = fspl_like.min()
        vmax = fspl_like.max()
        if vmax > vmin:
            fspl_like = (fspl_like - vmin) / (vmax - vmin)
        else:
            fspl_like = np.ones_like(fspl_like, dtype=np.float32)

        # 强制 Tx 位置为 1
        fspl_like[int(tx_y), int(tx_x)] = 1.0

        # 增加通道维
        fspl_like = np.expand_dims(fspl_like.astype(np.float32), axis=2)
        return fspl_like

    def __getitem__(self, idx):
        map_idx_in_split = idx // self.numTx
        tx_idx_in_map = idx % self.numTx
        dataset_map_ind = self.maps_inds[self.ind1 + map_idx_in_split] + 1

        name1 = str(dataset_map_ind) + ".png"
        name2 = str(dataset_map_ind) + "_" + str(tx_idx_in_map) + ".png"

        def check_file_exists(file_path):
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"文件缺失：{file_path}，索引：{dataset_map_ind}，TX：{tx_idx_in_map}"
                )

        # Load building map
        img_name_buildings = os.path.join(self.dir_buildings, name1)
        check_file_exists(img_name_buildings)
        image_buildings = np.asarray(
            io.imread(img_name_buildings), dtype=np.float32
        ) / 255.0

        # 反转建筑图输入
        if self.reverseInput == "yes":
            image_buildings = 1.0 - image_buildings
            
        # Load transmitter map
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        check_file_exists(img_name_Tx)
        image_Tx = np.asarray(
            io.imread(img_name_Tx), dtype=np.float32
        ) / 255.0

        # 如果启用 fsplInput，则将第二通道替换为 FSPL-like log-distance
        if self.fsplInput == "yes":
            image_Tx = self._generate_fspl_like_map(image_Tx)

        # Load ground truth gain map
        img_name_gain = os.path.join(self.dir_gain, name2)
        check_file_exists(img_name_gain)
        image_gain = np.expand_dims(
            io.imread(img_name_gain).astype(np.float32), axis=2
        ) / 255.0

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        # Ensure channel dimension exists
        if image_buildings.ndim == 2:
            image_buildings = np.expand_dims(image_buildings, axis=2)
        if image_Tx.ndim == 2:
            image_Tx = np.expand_dims(image_Tx, axis=2)

        # Prepare third channel based on cars input setting
        if self.carsInput == "no":
            # Use buildings as third channel
            third_channel = image_buildings
        else:
            # Load cars map for third channel
            img_name_cars = os.path.join(self.dir_cars, name1)
            check_file_exists(img_name_cars)
            image_cars = np.asarray(
                io.imread(img_name_cars), dtype=np.float32
            ) / 255.0
            if image_cars.ndim == 2:
                image_cars = np.expand_dims(image_cars, axis=2)
            third_channel = image_cars

        # Concatenate to form a 3-channel input
        # (Buildings, Tx/FSPL-like, Buildings/Cars)
        inputs_numpy = np.concatenate(
            [image_buildings, image_Tx, third_channel], axis=2
        )

        inputs_numpy, image_gain = self._apply_train_augmentation(inputs_numpy, image_gain)

        if self.transform:
            inputs = self.transform(inputs_numpy).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
        else:
            inputs = torch.from_numpy(
                inputs_numpy.transpose((2, 0, 1))
            ).type(torch.float32)
            image_gain = torch.from_numpy(
                image_gain.transpose((2, 0, 1))
            ).type(torch.float32)

        return inputs, image_gain, name2
