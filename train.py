import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import numpy as np
import random
import glob
from PIL import Image
from tqdm import tqdm

import data_split, exploration, models

# seed
def set_seed(seed = 7777):
    # Sets the seed of the entire notebook so results are the same every time we run # This is for REPRODUCIBILITY
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset split
# origin_path = "./QC_sample/"
# data_split.split(origin_path)

# dataset
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
    def __init__(self, data_path, mode, transform=None):
        self.path = sorted(glob.glob(os.path.join(data_path, mode, "*", "*.png")))
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        image = self.path[index]
        json = self.path[index][:-3]+'json'
        # ./dataset/train\apple_fuji\apple_fuji_L_1-10_5DI90.png
        img = Image.open(image).convert("RGB")
        if self.transform is not None :
            img = self.transform(img)

        # label = exploration.take_label(json).split('/')[0]
        label_temp = image.split('\\')[-1]

        for cate in CATEGORY:
            if cate in label_temp:
                label = CATEGORY[cate]

        return img, label

    def __len__(self):
        return len(self.path)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])


data_path = "./dataset/"
train_data = QCDataset(data_path, "train", transform=train_transform)
test_data = QCDataset(data_path, "test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model, size = models.initialize_model("resnet18", 18, use_pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

epochs = 10
val_every = 1
save_weights_dir = "./weight"
os.makedirs(save_weights_dir, exist_ok=True)

# train
def train(num_epoch, model, train_loader, test_loader, criterion, optimizer,
          save_dir, val_every, device):

    print("String... train !!! ")
    best_loss = 9999
    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels  = imgs.to(device), labels.to(device)
            output = model(imgs)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()

            print("Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}, Acc : {:.2f}%".format(
                epoch + 1, num_epoch, i +
                1, len(train_loader), loss.item(), acc.item() * 100
            ))

            if (epoch + 1) % val_every == 0:
                avg_loss = validation(
                    epoch + 1, model, test_loader, criterion, device)
                if avg_loss < best_loss:
                    print("Best prediction at epoch : {} ".format(epoch + 1))
                    print("Save model in", save_dir)
                    best_loss = avg_loss
                    save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")


def validation(epoch, model, test_loader, criterion, device):
    print("Start validation # {}".format(epoch))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avg_loss = total_loss / cnt
        print("Validation # {} Acc : {:.2f}% Average Loss : {:.4f}%".format(
            epoch, correct / total * 100, avg_loss
        ))

    model.train()
    return avg_loss


def save_model(model, save_dir, file_name="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)


def eval(model, test_loader, device):
    print("Starting evaluation")
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (imgs, labels) in tqdm(enumerate(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            # 점수가 가장 높은 클래스 선택
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()

        print("Test acc for image : {} ACC : {:.2f}".format(
            total, correct / total * 100))
        print("End test.. ")


"""model load => model test"""
# model.load_state_dict(torch.load("./weight/best.pt"))

if __name__ == "__main__":
    train(epochs, model, train_loader, test_loader, criterion, optimizer, save_weights_dir, val_every, device)
    # eval(model, test_loader, device)