import math
import random
from logging import getLogger

from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

logger = getLogger()


class JigsawTransform:
    
    def __init__(
        self,
        num_patches: int = 9, 
        resize: int = 256, 
        crop_size: int = 255,
        grayscale_p: float = 0.2, 
        jitter_value: int = 21, 
        mean_std = [
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        ]
        
    ):

        self.num_patches = num_patches
        assert jitter_value >= 0, "Negative jitter not supported"
        self.jitter = jitter_value
        self.grid_size = int(math.sqrt(self.num_patches))  # usually = 3
        self.gray_p = grayscale_p
        self.__img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size)
            ])
        self.__patch_transform = transforms.Compose([
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std),
        ])

    def __call__(self, x: Image):

        x = self.__img_transform(x)
        if np.random.rand() <= self.gray_p:
            x = x.convert('LA').convert('RGB')

        # patch extraction loop
        grid_size = int(x.size[0] / self.grid_size)
        patch_size = grid_size - self.jitter
        jitter = np.random.randint(
            0, self.jitter + 1, (2, self.grid_size, self.grid_size)
        )
        tensor_patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x_offset = i * grid_size
                y_offset = j * grid_size
                y0 = y_offset + jitter[1, i, j]
                y1 = y0 + patch_size
                x0 = x_offset + jitter[0, i, j]
                x1 = x0 + patch_size
                coords = np.array([x0, y0, x1, y1]).astype(int)
                patch = x.crop(coords.tolist())
                assert patch.size[0] == patch_size, "Image not cropped properly"
                assert patch.size[1] == patch_size, "Image not cropped properly"
                tensor_patch = self.__patch_transform(patch)
                tensor_patches.append(tensor_patch)

        return tensor_patches


def rgb_jittering(x: Image):
            x = np.array(x, 'int32')
            for ch in range(3):
                x[:, :, ch] += np.random.randint(-2, 2)
            x[x > 255] = 255
            x[x < 0] = 0
            return x.astype('uint8')


class MultiCropRotateJigsawDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropRotateJigsawDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index
        self.jigsaw_transform = JigsawTransform()
        self.perm_path = 'permutations/permutations_max_100.npy'
        self.perm_loaded = False
        self.perms = None
        self._load_perms()

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def _load_perms(self):
        self.perms = np.load(self.perm_path)
        if np.min(self.perms) == 1:
            self.perms = self.perms - 1
        self.perm_loaded = True

    def do_jigsaw(self, img):
        jigsaw_tensors = self.jigsaw_transform(img)
        perm_index = np.random.randint(self.perms.shape[0])
        shuffled_patches = [
            torch.FloatTensor(jigsaw_tensors[i]) for i in self.perms[perm_index]
        ]
        jigsaw_tensor = torch.stack(shuffled_patches) # num_patches x C x H x W
        jigsaw_target = torch.Tensor([perm_index]).long()

        return jigsaw_tensor, jigsaw_target

    @staticmethod
    def rotate_img(img, rot):
        if rot == 0: # 0 degrees rotation
            return img
        elif rot == 90: # 90 degrees rotation
            # return torch.flipud(torch.transpose(img, 1, 2))
            return torch.rot90(img, 1,  (1, 2))
        elif rot == 180: # 90 degrees rotation
            return torch.rot90(img, 2,  (1, 2))
            # return torch.fliplr(torch.flipud(img))
        elif rot == 270: # 270 degrees rotation / or -90
            return torch.rot90(img, 3,  (1, 2))
            # return torch.transpose(torch.flipud(img), 1, 2)
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)

        j1, jt1 = self.do_jigsaw(image)

        rt1 = random.randint(0, 3)
        rt2 = random.randint(0, 3)

        rotation_angles = [0, 90, 180, 270]

        multi_crops = list(map(lambda trans: trans(image), self.trans))

        r1 = self.rotate_img(multi_crops[0], rotation_angles[rt1])
        r2 = self.rotate_img(multi_crops[1], rotation_angles[rt2])

        return multi_crops, (r1, r2), (rt1, rt2), (j1, jt1)



class MultiCropRotationDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropRotationDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    @staticmethod
    def rotate_img(img, rot):
        if rot == 0: # 0 degrees rotation
            return img
        elif rot == 90: # 90 degrees rotation
            # return torch.flipud(torch.transpose(img, 1, 2))
            return torch.rot90(img, 1,  (1, 2))
        elif rot == 180: # 90 degrees rotation
            return torch.rot90(img, 2,  (1, 2))
            # return torch.fliplr(torch.flipud(img))
        elif rot == 270: # 270 degrees rotation / or -90
            return torch.rot90(img, 3,  (1, 2))
            # return torch.transpose(torch.flipud(img), 1, 2)
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)

        target_1 = random.randint(0, 3)
        target_2 = random.randint(0, 3)

        rotation_angles = [0, 90, 180, 270]

        multi_crops = list(map(lambda trans: trans(image), self.trans))

        aug_1_rot = self.rotate_img(multi_crops[0], rotation_angles[target_1])
        aug_2_rot = self.rotate_img(multi_crops[1], rotation_angles[target_2])

        if self.return_index:
            return index, multi_crops
        return multi_crops, (aug_1_rot, aug_2_rot), (target_1, target_2)


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
