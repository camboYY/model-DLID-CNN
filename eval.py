import torch
import torchvision.transforms as transforms
import numpy as np
import cv2, os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from models.dlidcnn import DLID_CNN

# Config
MODEL_PATH = "dlidcnn.pth"
LFW_PATH = "./data/lfw"
EMB_DIM = 512

# Model
model = DLID_CNN(embedding_dim=EMB_DIM, num_classes=10000).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def get_embedding(img_path):
    img = cv2.imread(img_path)[:,:,::-1]
    img = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat, _ = model(img)
    return feat.cpu().numpy()

# Example: compare 2 faces
emb1 = get_embedding(f"{LFW_PATH}/person1/1.jpg")
emb2 = get_embedding(f"{LFW_PATH}/person1/2.jpg")
sim = np.dot(emb1, emb2.T).item()
print("Cosine similarity:", sim)
