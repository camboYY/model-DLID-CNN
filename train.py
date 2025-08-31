import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.dlidcnn import DLID_CNN
from models.loss import DLIDLoss

# Config
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.1
EMB_DIM = 512
DATASET_PATH = "./data/train"

# Dataset
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Model
num_classes = len(train_dataset.classes)
model = DLID_CNN(embedding_dim=EMB_DIM, num_classes=num_classes).cuda()
criterion = DLIDLoss(num_classes=num_classes, feat_dim=EMB_DIM).cuda()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        feats, _ = model(imgs)
        loss = criterion(feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "dlidcnn.pth")
print("✅ Model saved: dlidcnn.pth")
