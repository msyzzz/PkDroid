import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
expno = "ttttttt"

def printlog(s):
    print(s)
    with open(f"./log/exp{expno}.txt", "a") as f:
        f.write(s+"\n")


class CNN_16bit_Grayscale(nn.Module):
    def __init__(self):
        super(CNN_16bit_Grayscale, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dilated_conv = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4)
        # self.dilated_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.dilated_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.dilated_conv(x))
        x = F.relu(self.dilated_conv2(x))
        # x = F.adaptive_avg_pool2d(x, (64, 64))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        return focal_loss.mean()


class GrayscaleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = self.load_image(self.image_paths[idx])
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        # return image, label
        return image, label, image_path 

    def load_image(self, path):
        image = Image.open(path).convert('I')  # 'I' 表示16位灰度图
        image = np.array(image)/65536
        return image.astype(np.float32)


from torch.utils.data import Sampler
import random

class HardNegativeSampler(Sampler):
    def __init__(self, dataset, hard_sample_ratio=0.2):
        self.dataset = dataset
        self.hard_sample_ratio = hard_sample_ratio
        self.hard_samples = set() 

    def update(self, misclassified_paths):
        self.hard_samples.update(misclassified_paths)

    def __iter__(self):
        all_indices = list(range(len(self.dataset)))
        hard_indices = [i for i in all_indices 
                        if self.dataset.image_paths[i] in self.hard_samples]

        other_indices = list(set(all_indices) - set(hard_indices))

        sample_size = int(self.hard_sample_ratio * len(all_indices))
        selected_others = random.sample(other_indices, min(sample_size, len(other_indices)))

        selected_indices = hard_indices + selected_others
        random.shuffle(selected_indices)
        return iter(selected_indices)

    def __len__(self):
        return len(self.dataset)
    

# 数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 假设16位灰度图范围在0-65535，可以根据实际情况调整
])

image_paths = []
labels = []
exp = 2 
benum = 0
malnum = 0
path_nb = "./data/image_non_packed_benign"
path_nm = "./data/image_non_packed_malware"
path_pb = "./data/image_packed_benign"
path_pm = "./data/image_packed_malware"
path_ub = "./data/image_unpacked_benign"
path_um = "./data/image_unpacked_malware"

for f in os.listdir(path_nb):
        path = os.path.join(path_nb, f)
        image_paths.append(path)

if exp == 1:
    for f in os.listdir(path_pm):
        path = os.path.join(path_pm, f)
        image_paths.append(path)
    benum = 6200
    malnum = 5305
if exp == 2:
    for f in os.listdir(path_um):
        path = os.path.join(path_um, f)
        image_paths.append(path)
    benum = 6200
    malnum = 5305
if exp == 3:
    for f in os.listdir(path_pb):
        path = os.path.join(path_pb, f)
        image_paths.append(path)
    for f in os.listdir(path_nm):
        path = os.path.join(path_nm, f)
        image_paths.append(path)
    for f in os.listdir(path_pm):
        path = os.path.join(path_pm, f)
        image_paths.append(path)
    benum = 6200 + 806
    malnum = 5305 + 837
if exp == 4:
    for f in os.listdir(path_ub):
        path = os.path.join(path_ub, f)
        image_paths.append(path)
    for f in os.listdir(path_nm):
        path = os.path.join(path_nm, f)
        image_paths.append(path)
    for f in os.listdir(path_um):
        path = os.path.join(path_um, f)
        image_paths.append(path)
    benum = 6200 + 208
    malnum = 5305 + 837
labels = [0.0] * benum + [1.0] * malnum

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = GrayscaleDataset(train_paths, train_labels, transform)
val_dataset = GrayscaleDataset(val_paths, val_labels, transform)
sampler = HardNegativeSampler(train_dataset, hard_sample_ratio=0.4)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8, shuffle=False)

# 模型实例化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_16bit_Grayscale().to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = FocalLoss(alpha=0.5, gamma=2.0)
criterion = nn.BCEWithLogitsLoss()

# 训练过程
num_epochs = 60
max_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    error_samples1 = []
    total = 0
    for images, labels,  paths in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs.squeeze(dim=1), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # predicted = torch.round(torch.sigmoid(outputs))
        threshold = 0.7 

        probs = torch.sigmoid(outputs)  
        predicted = (probs >= threshold).int() 
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i].item() != labels[i].item():
                error_samples1.append(paths[i])

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # 验证过程
    model.eval() 
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    error_samples = []
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            # predicted = torch.round(torch.sigmoid(outputs))
            threshold = 0.7

            probs = torch.sigmoid(outputs) 
            predicted = (probs >= threshold).int() 
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

            for i in range(len(labels)):
                if predicted[i].item() != labels[i].item():
                    error_samples.append(paths[i])

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    sampler.update(error_samples1)
    error_samples_file = f'error_epoch_{epoch+1}.txt'
    # with open(os.path.join("./log", error_samples_file), 'w') as f:
    #     for path in error_samples:
    #         f.write(f"{path}\n")

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()


    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    recall = recall_score(all_preds, all_labels)
    precision = precision_score(all_preds, all_labels)
    # if val_accuracy > max_acc:
    #     fprr, tprr, thresholds = roc_curve(all_labels, all_probs)
    #     auc = roc_auc_score(all_labels, all_probs)

    #     # 绘图
    #     plt.figure()
    #     plt.plot(fprr, tprr, color='red', label=f'ROC curve (AUC = {auc:.4f})')
    #     plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('EXP3 ROC Curve')
    #     plt.legend()
    #     plt.grid()
    #     # plt.show()
    #     plt.savefig(f"./ROC/exp{expno}_epoch{epoch+1}.png")
    #     plt.close()
    #     max_acc = val_accuracy

    printlog(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    printlog(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    printlog(f"Epoch [{epoch+1}/{num_epochs}], FPR: {fpr:.4f}, FNR: {fnr:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")


now = datetime.now().strftime("%m_%d_%H_%M")

torch.save(model.state_dict(), './model/16bit_{}.pth'.format(now))
print("Model saved!")
