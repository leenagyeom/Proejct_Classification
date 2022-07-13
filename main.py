import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import numpy as np
import random

import dataset, models, utils

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


train_data = dataset.QCDataset("./Training", transform=train_transform)
test_data = dataset.QCDataset("./Validation", transform=test_transform)


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

modelname = "ghostnet"
model, size = models.initialize_model(modelname, 54, use_pretrained=True)
model = model.to(device)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

epochs = 10
val_every = 10
save_weights_dir = "./weight"
os.makedirs(save_weights_dir, exist_ok=True)


"""model load => model test"""
# model.load_state_dict(torch.load("./weight/best.pt", map_location='cpu'))

if __name__ == "__main__":
    model, data = utils.train(epochs, model, train_loader, test_loader, criterion, optimizer, save_weights_dir, val_every, device)
    utils.loss_acc_visualize(data, modelname)
    utils.visual_predict(model, modelname, test_data)
    # utils.eval(model, test_loader, device)