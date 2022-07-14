import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

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

data = []
def train(modelname, num_epoch, model, train_loader, test_loader, criterion, optimizer,
          save_dir, val_every, device):

    print("String... train !!! ")
    best_loss = 9999
    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
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

            global avg_acc, avg_loss
            avg_acc = 0
            avg_loss = 0

            if (epoch + 1) % val_every == 0:
                avg_acc, avg_loss = validation(
                    epoch + 1, model, test_loader, criterion, device)
                if avg_loss < best_loss:
                    print("Best prediction at epoch : {} ".format(epoch + 1))
                    print("Save model in", save_dir)
                    best_loss = avg_loss
                    save_model(model, save_dir, file_name=f"{modelname}_best.pt")
            data.append([acc.item() * 100, loss.item(), avg_acc, avg_loss])
            print()

        pd_data = pd.DataFrame(data, columns=['train_accu', 'train_loss', 'test_accu', 'test_loss'])

        save_model(model, save_dir, file_name=f"{modelname}_last.pt")
        return model, pd_data


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
    return correct / total * 100, avg_loss.item()


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

def loss_acc_visualize(history, modelname, path="./results"):
    os.makedirs("./results", exist_ok=True)

    plt.figure(figsize=(20, 10))

    plt.suptitle(f"SGD; 0.01")

    plt.subplot(121)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['test_loss'], label='test_loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(122)
    plt.plot(history['train_accu'], label='train_accu')
    plt.plot(history['test_accu'], label='test_accu')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.savefig(os.path.join(str(path),f'loss_acc_{modelname}.png'))


def visual_predict(model, modelname, data, path="./results"):
    os.makedirs("./results", exist_ok=True)

    c = np.random.randint(0, len(data))
    img, labels = data[c]
    print(labels)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img.view(1, 3, 224, 224).cuda())
        out = torch.exp(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title(labels)
    plt.subplot(122)

    for k, v in CATEGORY.items():
        if v == labels:
           key = k

    plt.barh(key, out.cpu().numpy()[0])

    plt.savefig(os.path.join(str(path),f'predict_{modelname}.png'))