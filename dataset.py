from torch.utils.data import Dataset
import glob
from PIL import Image
import os

CATEGORY = {'apple_fuji_l': 0, 'apple_fuji_m': 1, 'apple_fuji_s': 2,
            'apple_yanggwang_l': 3, 'apple_yanggwang_m': 4, 'apple_yanggwang_s': 5,
            'cabbage_green_l': 6, 'cabbage_green_m': 7, 'cabbage_green_s': 8,
            'cabbage_red_l': 9, 'cabbage_red_m': 10, 'cabbage_red_s': 11,
            'chinese-cabbage_l': 12, 'chinese-cabbage_m': 13, 'chinese-cabbage_s': 14,
            'garlic_uiseong_l': 15, 'garlic_uiseong_m': 16, 'garlic_uiseong_s': 17,
            'mandarine_hallabong_l': 18, 'mandarine_hallabong_m': 19, 'mandarine_hallabong_s': 20,
            'mandarine_onjumilgam_l': 21, 'mandarine_onjumilgam_m': 22, 'mandarine_onjumilgam_s': 23,
            'onion_red_l': 24, 'onion_red_m': 25, 'onion_red_s': 26,
            'onion_white_l': 27, 'onion_white_m': 28, 'onion_white_s': 29,
            'pear_chuhwang_l': 30, 'pear_chuhwang_m': 31, 'pear_chuhwang_s': 32,
            'pear_singo_l': 33, 'pear_singo_m': 34, 'pear_singo_s': 35,
            'persimmon_bansi_l': 36, 'persimmon_bansi_m': 37, 'persimmon_bansi_s': 38,
            'persimmon_booyu_l': 39, 'persimmon_booyu_m': 40, 'persimmon_booyu_s': 41,
            'persimmon_daebong_l': 42, 'persimmon_daebong_m': 43, 'persimmon_daebong_s': 44,
            'potato_seolbong_l': 45, 'potato_seolbong_m': 46, 'potato_seolbong_s': 47,
            'potato_sumi_l': 48, 'potato_sumi_m': 49, 'potato_sumi_s': 50,
            'radish_winter-radish_l': 51, 'radish_winter-radish_m': 52, 'radish_winter-radish_s': 53}


class QCDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.path = sorted(glob.glob(os.path.join(data_path, "*", "*", "*.png")))
        self.transform = transform

    def __getitem__(self, index):
        image = self.path[index]
        json = self.path[index][:-3]+'json'
        # ./dataset/train\apple_fuji\apple_fuji_L_1-10_5DI90.png
        img = Image.open(image).convert("RGB")
        if self.transform is not None :
            img = self.transform(img)

        # label_temp = onion_red_S_75-19_4DI45.png
        label = image.lower().split('\\')[-1]


        if 'chinese-cabbage' in label:
            # chinese-cabbage_l
            label = label.split('_')
            label = label[0] + '_' + label[1]
        else :
            label = label.split('_')
            label = label[0] + '_' + label[1] + '_' + label[2]

        labeling = CATEGORY[label]

        return img, labeling, label

    def __len__(self):
        return len(self.path)