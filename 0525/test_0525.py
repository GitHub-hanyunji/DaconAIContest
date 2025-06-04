import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
from timm.models.convnext import convnext_tiny
from timm import create_model
import matplotlib.pyplot as plt
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F  # Softmax 쓰기 위해

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchvision.transforms.functional import to_pil_image

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 50,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 64,
    'SEED': 41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

all_img_list = glob.glob('./train/*/*')
df = pd.DataFrame(columns=['img_path', 'rock_type'])
df['img_path'] = all_img_list
df['rock_type'] = df['img_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

train_df, val_df, _, _ = train_test_split(df, df['rock_type'], test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])
le = preprocessing.LabelEncoder()
train_df['rock_type'] = le.fit_transform(train_df['rock_type'])
val_df['rock_type'] = le.transform(val_df['rock_type'])

# Dropout 추가된 convnext_tiny 클래스 정의
class ConvNeXtWithDropout(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.3):
        super().__init__()
        self.model = convnext_tiny(pretrained=True)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            # nn.Dropout(p=dropout_p),
            # nn.Linear(in_features, num_classes),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    

class PadSquare(ImageOnlyTransform):
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w, c = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

train_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.RandomResizedCrop(CFG['IMG_SIZE'], CFG['IMG_SIZE'], scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ], p=0.6),
    A.Sharpen(alpha=(0.1, 0.2), lightness=(0.9, 1.0), p=0.3),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.05, rotate_limit=10, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_transform2 = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=(0.2, 0.1), scale_limit=(0.7, 1.0), rotate_limit=90, p=0.5),
    A.Blur(blur_limit=8, p=0.4),
    A.CoarseDropout(max_holes=30, max_height=16, max_width=16, p=0.2),
    A.OneOf([
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
    ], p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_dataset = CustomDataset(train_df['img_path'].values, train_df['rock_type'].values, train_transform2)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)

val_dataset = CustomDataset(val_df['img_path'].values, val_df['rock_type'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)


def train(model, optimizer, train_loader, val_loader, scheduler, device,freeze_backbone=True, epochs=20,start_epoch=1, best_score=0,stage=1):
    model.to(device)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    # 수정
    # 클래스 불균형 대응 -> 기존 loss 교체
    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['rock_type']),
                    y=train_df['rock_type']
                    )
    
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    #criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    #criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None

    os.makedirs("./output", exist_ok=True)
    best_model_save_path = "./output/best_model.pth"

            
    print(f"Start training [Stage{stage}]")
    x_arr=[]
    rec_loss = [[],[]]
    rec_acc=[[],[]]
    class_acc=[[],[],[],[],[],[],[]]
    
    
    
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_preds, true_train_labels = [], []
        train_loss = []
        model_save_path = f"./output/model_epoch{epoch}.pth"

        for imgs, labels in tqdm(iter(train_loader), desc=f"[Stage{stage}] Epoch {epoch}"):
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            train_preds += output.argmax(dim=1).detach().cpu().numpy().tolist()
            true_train_labels += labels.detach().cpu().numpy().tolist()

        _train_loss = np.mean(train_loss)
        _train_score = f1_score(true_train_labels, train_preds, average='macro')  # Train F1 계산

        _val_loss, _val_score,class_accuracy = validation(model, criterion, val_loader, device)

        print(f'Train Loss: {_train_loss:.5f}, Train Marco F1: {_train_score:.5f}')
        print(f'Val Loss: {_val_loss:.5f}, Val Macro F1: {_val_score:.5f}')

        # 모델 가중치 및 학습 상태 저장
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_score': best_score
        }
        torch.save(checkpoint, model_save_path)


        rec_loss[0].append(_train_loss)
        rec_loss[1].append(_val_loss)

        rec_acc[0].append(_train_score)
        rec_acc[1].append(_val_score)


        for i, acc in enumerate(class_accuracy):
            class_acc[i].append(acc)
            print(f"  Class {i+1}: {acc:.4f}")
        
        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            
            # 모델 가중치 저장
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {best_model_save_path}")

        plot(rec_loss,rec_acc,epoch,stage)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"Training time {total_time_str}")

    return best_model

def validation(model, criterion, val_loader, device, threshold=0.4):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()
            
            output = model(imgs)
            loss = criterion(output, labels)
            val_loss.append(loss.item())

            probs = F.softmax(output, dim=1)  # 확률로 변환
            confs, pred_labels = torch.max(probs, dim=1)

            # 🔥 조건: 자신 없으면 → 클래스 3 (즉 index == 3)
            pred_labels[confs < threshold] = 3

            preds += pred_labels.detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')

        cm = confusion_matrix(true_labels, preds)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

    return _val_loss, _val_score, class_accuracy

# 그래프 그리기 
def plot(rec_loss,rec_acc,epoch,stage):
    to_numpy_loss = np.array(rec_loss)
    to_numpy_acc=np.array(rec_acc)
    #길이 자동 추정
    x_arr = np.arange(len(rec_loss[0]))
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))
    # 손실 그래프
    ax1.plot(x_arr, to_numpy_loss[0], '-', label='Train loss', marker='o')      
    ax1.plot(x_arr, to_numpy_loss[1], '--', label='Valid loss', marker='o')
    ax1.legend(fontsize=15)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch', size=15)
    ax1.set_ylabel('Loss', size=15)
    ax1.legend()
    # 정확도 그래프
    ax2.plot(x_arr, to_numpy_acc[0], '-', label='Train F1', marker='o')
    ax2.plot(x_arr, to_numpy_acc[1], '--', label='Valid F1', marker='o')
    ax2.legend(fontsize=15)
    ax2.set_title('F1 Score')
    ax2.set_xlabel('Epoch', size=15)
    ax2.set_ylabel('F1', size=15)
    ax2.legend()
    plt.tight_layout()
    if(stage==1):
        plt.savefig(f"./graph/1/graph{epoch}.png")
    else:
        plt.savefig(f"./graph/2/graph{epoch}.png")
    plt.close()  # 메모리 누수 방지

# 답안지 작성
def inference(model, test_loader, device,threshold=0.4):
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            output = model(imgs)
            probs = F.softmax(output, dim=1)
            confs, pred_labels = torch.max(probs, dim=1)

            # 🔥 자신 없으면 → 클래스 3
            pred_labels[confs < threshold] = 3

            preds += pred_labels.detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds


test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()
    
    os.makedirs('./graph/1', exist_ok=True)
    os.makedirs('./graph/2', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    start_epoch = 1
    best_score = 0
    checkpoint_path = './output/model_epoch50.pth'  # 이어서 학습할 pth 경로

    resume_training = True  # <- 이어서 할지 여부

    # # 모델 
    model = ConvNeXtWithDropout(num_classes=7,dropout_p=0.3).to(device)

    # # <- 이걸로 할 경우 위에 모델 정의 할 때 pretrained=False로 해줘야함
    #model.load_state_dict(torch.load('./output/model_epoch15.pth', map_location=device)) 
    optimizer = optim.Adam(model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=1e-4)
    # 스케줄러 변경
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold_mode='abs', min_lr=1e-8, verbose=True)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
    

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', 0)
        print(f"✅ 체크포인트에서 epoch {start_epoch}부터 재개합니다.")

    # Stage 1: Freeze backbone
    #model = train(model, optimizer, train_loader, val_loader, scheduler, device, freeze_backbone=True, start_epoch=1,epochs=15, best_score=best_score,stage=1)


    # # Stage 2: Unfreeze and fine-tune all
    # for param in model.parameters():
    #     param.requires_grad = True

    
    # model = train(model, optimizer, train_loader, val_loader, scheduler, device,  freeze_backbone=False, start_epoch=1, epochs=50,best_score=best_score,stage=2)

    preds = inference(model, test_loader, device)
    submit = pd.read_csv('./sample_submission.csv')
    submit['rock_type'] = preds
    submit.to_csv('./baseline_submit3.csv', index=False)
