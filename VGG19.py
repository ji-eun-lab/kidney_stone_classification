


import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import splitfolders
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from torchsummary import summary as summary_
from torchvision.transforms import Compose, Normalize, ToTensor

import copy
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



### train, val, test 셋으로 분류
import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir, target_dir, train_size=0.6):
    # 처리할 클래스만 지정
    classes = ['Cyst', 'Normal', 'Stone', 'Tumor']
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        
        # 학습, 검증 및 테스트 세트로 분할
        train_files, test_files = train_test_split(files, train_size=train_size, random_state=42)
        val_files, test_files = train_test_split(test_files, train_size=0.5, random_state=42)
        
        # 각 세트별로 폴더 생성 및 파일 복사
        for file_set, set_name in zip([train_files, val_files, test_files], ['train', 'validation', 'test']):
            set_dir = os.path.join(target_dir, set_name, cls)
            os.makedirs(set_dir, exist_ok=True)
            for file in file_set:
                shutil.copy(os.path.join(source_dir, cls, file), os.path.join(set_dir, file))

source_dir = '/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone'
target_dir = '/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone/dataset'

split_data(source_dir, target_dir)





# use the ImageNet transformation
transform = transforms.Compose([# 데이터 증강
    transforms.RandomRotation(degrees=(-30, 30)),  # 랜덤 회전
    transforms.RandomAffine(degrees=0, shear=(10, 10)),  # 랜덤 전단
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # 랜덤 줌
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 랜덤 이동
    transforms.RandomHorizontalFlip(),  # 수평 뒤집기
    # transforms.RandomVerticalFlip(),  # 수직 뒤집기 (필요한 경우)
    
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])





# 학습 데이터셋 로드
train_dataset = ImageFolder(root='/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone/dataset/train2', transform=transform)
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 검증 데이터셋 로드
val_dataset = ImageFolder(root='/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone/dataset/val', transform=transform)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = ImageFolder(root='/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone/dataset/test', transform=transform)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
  
classes = train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        num_features = self.vgg.classifier[0].in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),  # Change the output features to 4
        )
        
        
        
        # placeholder for the gradients
        self.gradients = None
        
        
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        if x.requires_grad:
            # register the hook
            h = x.register_hook(self.activations_hook)
        
        # register the hook
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)





# initialize the VGG model
vgg = VGG().to(device)

# 손실 함수 및 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.parameters(), lr=0.00001)

loss_values = []

## 여기 돌리기
for epoch in range():  
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    
    # Training
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    # 에포크별 평균 손실 계산 및 저장
    epoch_loss = running_loss / len(trainloader)
    loss_values.append(epoch_loss)
    
    # Validation
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = vgg(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    print(f'Epoch {epoch + 1}/{70}, Loss: {running_loss:.4f}, Train Accuracy: {(100 * correct_train / total_train):.2f}%, Val Accuracy: {(100 * correct_val / total_val):.2f}%')
    
# 모델 평가 (수정하기)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))    


print('Finished Training')

# 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(vgg.state_dict(), 'vgg19_weights.pth')

vgg.load_state_dict(torch.load('vgg19_weights.pth'))
vgg.eval()

# output = vgg(testloader)
# pred = output.argmax(dim=1)



# 공식 깃헙에서 가져옴
def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


# 원하는 레이어의 활성화 맵에 대한 그래디언트 계산
target_layer = [vgg.features_conv[-2]]  # 예시로 마지막 컨볼루션 레이어를 선택
grad_cam = GradCAM(model=vgg, target_layers = target_layer)


# 원본 이미지 로드 및 전처리
rgb_img = cv2.imread('/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone/Stone/Stone- (2).jpg', 1)[:, :, ::-1]
rgb_img = np.float32(cv2.resize(rgb_img, (224, 224))) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# 관심 있는 카테고리를 지정
targets = [ClassifierOutputTarget(2)]

# GradCAM 실행 및 히트맵 생성
heatmap = grad_cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = heatmap[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.axis('off')  # 축 정보 제거
plt.show(block=True)


# # 히트맵을 0과 1 사이로 정규화
# heatmap_min = heatmap.min()
# heatmap_max = heatmap.max()
# heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

# # 정규화된 히트맵을 0에서 255 사이의 값으로 스케일링
# heatmap_scaled = np.uint8(255 * heatmap_norm)
# heatmap_scaled_2d = heatmap_scaled.squeeze()

# # 컬러맵 적용
# heatmap_colored = cv2.applyColorMap(heatmap_scaled_2d, cv2.COLORMAP_JET)

# plt.imshow(heatmap_colored)
# plt.axis('off')  # 축 제거
# plt.show()

# # 원본 이미지 로드 (예시)
# img = cv2.imread('/home/cvip12/Dataset/Normal-Cyst-Tumor-Stone/Stone/Stone- (2).jpg')
# img = cv2.resize(img, (224, 224))  # 컬러맵과 동일한 크기로 조정

# # 히트맵을 원본 이미지에 오버레이
# superimposed_img = heatmap * 0.4 + img
# superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

# # 결과 이미지 표시
# cv2.imshow('Heatmap Overlay', superimposed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 히트맵을 원본 이미지에 오버레이
# heatmap_on_image = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

# # 시각화
# plt.imshow(heatmap_on_image)
# plt.show()