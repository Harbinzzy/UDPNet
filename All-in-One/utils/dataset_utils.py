import os
import random
import copy
from PIL import Image
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img,apply_consistent_augmentation
from utils.degradation_utils import Degradation

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {
            'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, "gopro": 5,
            "lol": 6,
        }

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if "gopro" in self.de_type:
            self._init_gopro_ids()
        if "lol" in self.de_type:
            self._init_lol_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir+"gt/")
        # print(name_list)
        clean_ids += [self.args.denoise_dir + "gt/" + id_ for id_ in name_list if id_.strip() in temp_ids]
        # print(clean_ids)

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
        
    def _init_gopro_ids(self):
        temp_ids = []
        # print(self.opt)
        # print(self.opt["data_file_dir"])
        gopro = self.args.data_file_dir + "gopro/gopro.txt"
        temp_ids += [self.args.gopro_dir + id_.strip()
                     for id_ in open(gopro)]
        self.gopro_ids = [
            {"clean_id": x, "de_type": 5} for x in temp_ids
        ] * 10

        self.gopro_counter = 0

        self.num_gopro = len(self.gopro_ids)

        print("Total GoPro Ids : {}".format(self.num_gopro))

    def _init_lol_ids(self):
        temp_ids = []
        lol = self.args.data_file_dir+"lol/lol.txt"
        temp_ids += [self.args.lol_dir + id_.strip() for id_ in open(lol)]
        self.lol_ids = [
            {"clean_id": x, "de_type": 6} for x in temp_ids
        ] * 50

        self.lol_counter = 0

        self.num_lol = len(self.lol_ids)
        print("Total LOL Ids : {}".format(self.num_lol))


    def _crop_patch(self, img_1, img_2, img_3):
        # 确保图像的维度符合预期
        if img_1 is None or img_2 is None or img_3 is None:
            raise ValueError("Input images for cropping are None.")
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_3 = img_3[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2, patch_3
    def _get_non_gopro_name(self, hazy_name):
        dir_name = hazy_name.split("input")[0] + "target/"
        name = hazy_name.split("/")[-1]
        nonhazy_name = dir_name + name
        return nonhazy_name

    def _get_non_lol_name(self, hazy_name):
        dir_name = hazy_name.split("low")[0] + "high/"
        name = hazy_name.split("/")[-1]
        nonhazy_name = dir_name + name
        return nonhazy_name

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "gopro" in self.de_type:
            self.sample_ids += self.gopro_ids
        if "lol" in self.de_type:
            self.sample_ids += self.lol_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            degrad_img_dir = os.path.dirname(clean_id)
            if de_id == 0:
                depth_dir = os.path.join(os.path.dirname(degrad_img_dir), "noisy15_depth2l")
            if de_id == 1:
                depth_dir = os.path.join(os.path.dirname(degrad_img_dir), "noisy25_depth2l")
            if de_id == 2:
                depth_dir = os.path.join(os.path.dirname(degrad_img_dir), "noisy50_depth2l")
            depth_filename = os.path.basename(clean_id).split('.')[0] + ".png"
            depth_path = os.path.join(depth_dir, depth_filename)
            depth_map = np.array(Image.open(depth_path).convert("L"))
            if depth_map is None:
                    raise ValueError(f"Depth map is None for {depth_path}")
            depth_map = depth_map[:, :, np.newaxis]
            depth_map = crop_img(depth_map, base=16)
            depth_patch = self.crop_transform(depth_map)
            depth_patch = np.array(depth_patch)
            clean_name = clean_id.split("/")[-1].split('.')[0]
            depth_patch = depth_patch[:, :, np.newaxis]
            clean_patch, depth_patch = random_augmentation(clean_patch, depth_patch)

            degrad_patch = self.D.single_degrade(clean_patch, de_id)

        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
                degrad_img_dir = os.path.dirname(sample["clean_id"])
                depth_dir = os.path.join(os.path.dirname(degrad_img_dir), "depth2l")
                depth_filename = os.path.basename(sample["clean_id"]).split('.')[0] + ".png"
                depth_path = os.path.join(depth_dir, depth_filename)
                depth_map = np.array(Image.open(depth_path).convert("L"))
                depth_map = depth_map[:, :, np.newaxis]
                if depth_map is None:
                    raise ValueError(f"Depth map is None for {depth_path}")
                depth_map = crop_img(depth_map, base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
                degrad_img_dir = os.path.dirname(sample["clean_id"])
                depth_dir = degrad_img_dir.replace("/synthetic/", "/depth2l/")
                depth_filename = os.path.splitext(os.path.basename(sample["clean_id"]))[0] + ".png"
                depth_path = os.path.join(depth_dir, depth_filename)
                depth_map = np.array(Image.open(depth_path).convert("L"))
                depth_map = depth_map[:, :, np.newaxis]
                if depth_map is None:
                    raise ValueError(f"Depth map is None for {depth_path}")
                depth_map = crop_img(depth_map, base=16)
            elif de_id == 5:
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_non_gopro_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
                degrad_img_dir = os.path.dirname(sample["clean_id"])
                depth_dir = os.path.join(os.path.dirname(degrad_img_dir), "depth2l")
                depth_filename = os.path.basename(sample["clean_id"]).split('.')[0] + ".png"
                depth_path = os.path.join(depth_dir, depth_filename)
                depth_map = np.array(Image.open(depth_path).convert("L"))
                depth_map = depth_map[:, :, np.newaxis]
                if depth_map is None:
                    raise ValueError(f"Depth map is None for {depth_path}")
                depth_map = crop_img(depth_map, base=16)
            elif de_id == 6:
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_non_lol_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
                degrad_img_dir = os.path.dirname(sample["clean_id"])
                depth_dir = os.path.join(os.path.dirname(degrad_img_dir), "depth2l")
                depth_filename = os.path.basename(sample["clean_id"]).split('.')[0] + ".png"
                depth_path = os.path.join(depth_dir, depth_filename)
                depth_map = np.array(Image.open(depth_path).convert("L"))
                depth_map = depth_map[:, :, np.newaxis]
                if depth_map is None:
                    raise ValueError(f"Depth map is None for {depth_path}")
                depth_map = crop_img(depth_map, base=16)

            degrad_patch, clean_patch, depth_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img, depth_map))
            
        degrad_patch = np.concatenate((degrad_patch, depth_patch), axis=-1)
        
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 25

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)

        depth_dir = os.path.join(os.path.dirname(os.path.dirname(self.clean_ids[clean_id])), "depth2l")
        depth_filename = clean_name + ".png"
        depth_path = os.path.join(depth_dir, depth_filename)
        
        depth_map = np.array(Image.open(depth_path).convert("L"))
        depth_map = depth_map[:, :, np.newaxis]
        depth_map = crop_img(depth_map, base=16)

        noisy_img = np.concatenate((noisy_img, depth_map), axis=-1)

        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)
        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean

class LolDataset(Dataset):
    def __init__(self, args, task="lol",addnoise = False,sigma = 15):
        super(LolDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'lol': 0}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.lol_path + 'low/')
            # print(name_list)
            print(self.args.lol_path)
            self.ids += [self.args.lol_path + 'low/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("low", "high")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("low")[0] + 'high/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        depth_dir = os.path.join(os.path.dirname(os.path.dirname(degraded_path)), "depth2l")
        degraded_name = os.path.basename(degraded_path).split('.')[0]
        depth_filename = degraded_name + ".png"
        depth_path = os.path.join(depth_dir, depth_filename)

        depth_map = np.array(Image.open(depth_path).convert("L"))
        depth_map = depth_map[:, :, np.newaxis]
        depth_map = crop_img(depth_map, base=16)

        degraded_img = np.concatenate((degraded_img, depth_map), axis=-1)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length
    
class GoproDataset(Dataset):
    def __init__(self, args, task="gopro",addnoise = False,sigma = 15):
        super(GoproDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'gopro': 0}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path + 'input/')
            self.ids += [self.args.gopro_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        depth_dir = os.path.join(os.path.dirname(os.path.dirname(degraded_path)), "depth2l")
        degraded_name = os.path.basename(degraded_path).split('.')[0]
        depth_filename = degraded_name + ".png"
        depth_path = os.path.join(depth_dir, depth_filename)

        depth_map = np.array(Image.open(depth_path).convert("L"))
        depth_map = depth_map[:, :, np.newaxis]
        depth_map = crop_img(depth_map, base=16)

        # 将深度图作为新通道添加到 degraded_img 上
        degraded_img = np.concatenate((degraded_img, depth_map), axis=-1)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = 15):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'enhance': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path +'input/')
            self.ids += [self.args.gopro_path + 'input/' + id_ for id_ in name_list]
        #elif self.task_idx == 3:
        #    self.ids = []
        #    name_list = os.listdir(self.args.enhance_path + 'low/')
        #    self.ids += [self.args.enhance_path + 'low/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name

        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")

        elif self.task_idx == 3:
            gt_name = degraded_name.replace("low", "high")
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        depth_dir = os.path.join(os.path.dirname(os.path.dirname(degraded_path)), "depth2l")
        degraded_name = os.path.basename(degraded_path).rsplit('.', 1)[0]
        depth_filename = degraded_name + ".png"
        depth_path = os.path.join(depth_dir, depth_filename)

        depth_map = np.array(Image.open(depth_path).convert("L"))
        depth_map = depth_map[:, :, np.newaxis]
        depth_map = crop_img(depth_map, base=16)

        degraded_img = np.concatenate((degraded_img, depth_map), axis=-1)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

class DehazeDataset(Dataset):
    def __init__(self, args, task="dehaze",addnoise = False,sigma = 15):
        super(DehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        depth_dir = os.path.join(os.path.dirname(os.path.dirname(degraded_path)), "depth2l")
        degraded_name = os.path.splitext(os.path.basename(degraded_path))[0]
        depth_filename = degraded_name + ".png"
        depth_path = os.path.join(depth_dir, depth_filename)

        depth_map = np.array(Image.open(depth_path).convert("L"))
        depth_map = depth_map[:, :, np.newaxis]
        depth_map = crop_img(depth_map, base=16)

        degraded_img = np.concatenate((degraded_img, depth_map), axis=-1)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    

    

