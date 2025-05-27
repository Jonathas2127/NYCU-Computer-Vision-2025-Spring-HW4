import os
import random
import copy
from PIL import Image
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def save_image_tensor(image_tensor, output_path="output/"):
    image_np = torch_to_np(image_tensor)
    # print(image_np.shape)
    p = np_to_pil(image_np)
    p.save(output_path)

#===degrade_utils==============================

class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1
    
#===image_utils=========================
def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

#===============================================================================================================

class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.clean_dir = args.clean_dir
        self.degraded_dir = args.degraded_dir
        self.patch_size = args.patch_size

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(self.patch_size),
        ])
        self.toTensor = ToTensor()

        # Automatically pair rain/snow degraded and clean images
        self.sample_ids = []
        for fname in sorted(os.listdir(self.degraded_dir)):
            degraded_path = os.path.join(self.degraded_dir, fname)

            if fname.startswith("rain-"):
                clean_fname = fname.replace("rain-", "rain_clean-")
            elif fname.startswith("snow-"):
                clean_fname = fname.replace("snow-", "snow_clean-")
            else:
                continue  # skip if filename doesn't match expected pattern

            clean_path = os.path.join(self.clean_dir, clean_fname)

            if os.path.exists(clean_path):
                self.sample_ids.append({
                    "clean_id": clean_path,
                    "degraded_id": degraded_path,
                    "de_type": 3  # you can later customize this if needed
                })

        print(f"✅ Found {len(self.sample_ids)} matched rain/snow image pairs.")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]

        degraded_img = crop_img(np.array(Image.open(sample["degraded_id"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)

        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degraded_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        fname = os.path.basename(sample["degraded_id"])
        return [fname, sample["de_type"]], degrad_patch, clean_patch

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

        return patch_1, patch_2

    def __len__(self):
        return len(self.sample_ids)


#===============================================================================================================================

class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
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
            # print(name_list)
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

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length