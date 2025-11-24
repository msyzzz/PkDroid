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
expno = "hsratio"


def printlog(s):
    print(s)
    with open(f"./log/exp{expno}.txt", "a") as f:
        f.write(s+"\n")



# 定义CNN模型
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
        
        self.fc1 = nn.Linear(128 * 64 * 64, 512)  # 输入特征图展平后的维度
        self.fc2 = nn.Linear(512, 1)  # 二分类输出
        
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

# 假设已经定义好数据集
# 自定义数据集类（需要根据你自己的数据格式进行调整）
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
        return image, label, image_path  # 返回路径

    def load_image(self, path):
        # 这里使用PIL库来加载图像，可以根据需要修改为加载16位灰度图
        image = Image.open(path).convert('I')  # 'I' 表示16位灰度图
        image = np.array(image)/65536
        return image.astype(np.float32)


from torch.utils.data import Sampler
import random

class HardNegativeSampler(Sampler):
    def __init__(self, dataset, hard_sample_ratio=0.2):
        self.dataset = dataset
        self.hard_sample_ratio = hard_sample_ratio
        self.hard_samples = set()  # 存储路径字符串

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
shells = pd.read_csv("./data/LookPack-master/mal2020_shells.csv")
be_shells = pd.read_csv("./data/LookPack-master/be2020_shells.csv")
path_b = "./data/image_be2020_512"
path_m = "./data/image_mal2020_512"
path_m1 = "/Data2/masy/apk/image_mal2020_512"

shells2 = pd.read_csv("./data/LookPack-master/mal2022_shells.csv")
be_shells2 = pd.read_csv("./data/LookPack-master/be2022_shells.csv")
path_b2 = "./data/image_be2022_512"
path_m2 = "./data/image_mal2022_512"
path_m12 = "/Data2/masy/apk/image_mal2022_512"
path_be_jiagu = "./data/image_be_jiagu_512"
path_be_jiagu_unpack = "/Data2/masy/apk/image_be_jiagu_512"
path_jiagu = "/Data2/masy/apk/image_jiagu"
image_paths = []
be_selected = []


files = os.listdir(path_b)
#3914
be_num = 3000
count = 0
for imagefile in files:
    shell_item = be_shells[be_shells['sha256'] == imagefile.split('.')[0]]
    # print(shell_item)
    if shell_item['shell'].values != "unpackaged":
        continue
    image_paths.append(os.path.join(path_b ,imagefile))
    be_selected.append(shell_item.iloc[0])
    count += 1
    if count == be_num:
        break
df_selected = pd.DataFrame(be_selected)
df_selected.to_csv("be20_selected.csv", index=False)


files = os.listdir(path_b2)
#3669
be_num = 3200
count = 0
for imagefile in files:
    shell_item = be_shells2[be_shells2['sha256'] == imagefile.split('.')[0]]
    # print(shell_item)
    if shell_item['shell'].values != "unpackaged":
        continue
    image_paths.append(os.path.join(path_b2 ,imagefile))
    be_selected.append(shell_item.iloc[0])
    count += 1
    if count == be_num:
        break

df_selected = pd.DataFrame(be_selected)
df_selected.to_csv("be22_selected.csv", index=False)

files = os.listdir(path_be_jiagu_unpack)
for imagefile in files:
    image_paths.append(os.path.join(path_be_jiagu_unpack ,imagefile))
    



# files = os.listdir(path_jiagu)
# for imagefile in files:
#     image_paths.append(os.path.join(path_jiagu ,imagefile))

files = os.listdir(path_m)
for imagefile in files:
    shell_item = shells[shells['sha256'] == imagefile.split('.')[0]]
    # print(shell_item)
    if shell_item['shell'].values == "unpackaged":
        image_paths.append(os.path.join(path_m ,imagefile))
files = os.listdir(path_m1)
for imagefile in files:  
    image_paths.append(os.path.join(path_m1 ,imagefile))
    # image_paths.append(os.path.join(path_m ,imagefile))

files = os.listdir(path_m2)
for imagefile in files:
    shell_item = shells2[shells2['sha256'] == imagefile.split('.')[0]]
    # print(shell_item)
    if shell_item['shell'].values == "unpackaged":
        image_paths.append(os.path.join(path_m2 ,imagefile))
files = os.listdir(path_m12)
for imagefile in files:  
    image_paths.append(os.path.join(path_m12 ,imagefile))
    # image_paths.append(os.path.join(path_m2 ,imagefile))
# 3. 数据准备
# 假设你有图像路径列表 image_paths 和对应的标签列表 labels
# image_paths = ["path_to_image1.jpg", "path_to_image2.jpg", "..."]  # 这里替换为你的图像路径

# 725 2228 112 3077

labels = [0.0] * (6200+208) + [1.0] * (5305+837)
# 使用train_test_split划分训练集和验证集
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = GrayscaleDataset(train_paths, train_labels, transform)
val_dataset = GrayscaleDataset(val_paths, val_labels, transform)
sampler = HardNegativeSampler(train_dataset, hard_sample_ratio=1)
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
num_epochs = 60  # 可以根据实际需要调整
max_acc = 0

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    error_samples1 = []
    total = 0
    for images, labels,  paths in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        
        loss = criterion(outputs.squeeze(dim=1), labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # predicted = torch.round(torch.sigmoid(outputs))
        threshold = 0.7  # 举例：调高阈值，模型更保守，不轻易判正类

        probs = torch.sigmoid(outputs)             # 转为概率
        predicted = (probs >= threshold).int()     # 自定义阈值后分类
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i].item() != labels[i].item():
                error_samples1.append(paths[i])

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # 验证过程
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    error_samples = []  # 存储预测错误的样本路径
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            # predicted = torch.round(torch.sigmoid(outputs))
            threshold = 0.7  # 举例：调高阈值，模型更保守，不轻易判正类

            probs = torch.sigmoid(outputs)             # 转为概率
            predicted = (probs >= threshold).int()     # 自定义阈值后分类
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

    # 计算假阳性率（FPR）和假阴性率（FNR）
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    recall = recall_score(all_preds, all_labels)
    precision = precision_score(all_preds, all_labels)
    if val_accuracy > max_acc:
        fprr, tprr, thresholds = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)

        # 绘图
        plt.figure()
        plt.plot(fprr, tprr, color='red', label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EXP3 ROC Curve')
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig(f"./ROC/exp{expno}_epoch{epoch+1}.png")
        plt.close()
        max_acc = val_accuracy
    # 打印每个epoch的训练和验证结果
    printlog(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    printlog(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    printlog(f"Epoch [{epoch+1}/{num_epochs}], FPR: {fpr:.4f}, FNR: {fnr:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")



# 获取当前时间
now = datetime.now().strftime("%m_%d_%H_%M")

# 模型保存
torch.save(model.state_dict(), './model/16bit_{}.pth'.format(now))
print("Model saved!")
