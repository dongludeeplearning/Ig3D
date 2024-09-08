import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from tqdm import tqdm
import time 
from datetime import datetime
from pdb import set_trace as st



# **** Data source 

IMAGE_FOLDER = "/Work/Datesets/AffectNet/train_set_clean/images/"
IMAGE_FOLDER_TEST = "/Work/Datesets/AffectNet/val_set_clean/images/"

train_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_train_file.csv"
)
valid_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_val_file.csv"
)

train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)



# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, balance=False):
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.balance = balance

        if self.balance:
            self.dataframe = self.balance_dataset()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.root_dir, f"{self.dataframe['number'].iloc[idx]}.jpg"
        )
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.new(
                "RGB", (224, 224), color="white"
            )  # Handle missing image file

        classes = torch.tensor(self.dataframe["exp"].iloc[idx], dtype=torch.long)  #dim 1
        va_labels = torch.tensor(self.dataframe.iloc[idx, 2:4].values, dtype=torch.float32) #dim 2
        expressions = torch.tensor(self.dataframe.iloc[idx, 4:160].values, dtype=torch.float32)  #dim 156

        if self.transform:
            image = self.transform(image)

        return image, classes, va_labels, expressions

    def balance_dataset(self):
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomGrayscale(0.01),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ), 
        transforms.RandomPerspective(
            distortion_scale=0.2, p=0.5
        ),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
        ), 
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Set parameters for dataloader 
BATCHSIZE = 64
NUM_EPOCHS = 20

train_dataset = CustomDataset(
    dataframe=train_annotations_df,
    root_dir=IMAGE_FOLDER,
    transform=transform,
    balance=True, # balanced sampler
)
valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    transform=transform_valid,
    balance=False,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=12
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=12
)

# ***** Define the model *****

MODEL = models.maxvit_t(weights="DEFAULT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LayerNorm(block_channels),
    nn.Linear(block_channels, block_channels),
    nn.Tanh(),  # (-1, 1)
)


#fusion the two features
class CustomClassifier(nn.Module):
    def __init__(self, block_channels, additional_feature_dim, num_classes=10):
        super(CustomClassifier, self).__init__()
        
        self.additional_feature_transform = nn.Linear(additional_feature_dim, block_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(block_channels * 2, block_channels),  # Concatenate features, so dimension is doubled
            nn.Tanh(),
            nn.Linear(block_channels, num_classes, bias=False)
        )
    
    def forward(self, x, additional_feature):
        additional_feature = self.additional_feature_transform(additional_feature)
        combined_features = torch.cat((x, additional_feature), dim=1)
        out = self.classifier(combined_features)
        return out

# suppose additional_feature_dim is the dim to add for susion
additional_feature_dim = 156  # emoca156
num_classes = 10

# change the classifer 
MODEL_cls= CustomClassifier(block_channels, additional_feature_dim, num_classes)

MODEL.to(DEVICE)  
MODEL_cls.to(DEVICE)
LR = 1e-5

# Define (weighted) loss function
weights = torch.tensor(
    [0.015605, 0.008709, 0.046078, 0.083078, 0.185434, 0.305953, 0.046934, 0.30821]
)
criterion_cls = nn.CrossEntropyLoss(weights.to(DEVICE))
criterion_cls_val = (
    nn.CrossEntropyLoss()
)  
criterion_reg = nn.MSELoss()

optimizer = optim.AdamW(MODEL.parameters(), lr=LR)
# lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=BATCHSIZE * NUM_EPOCHS)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=234, mode='triangular',cycle_momentum=False) #2241

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()

best_valid_loss = 100
best_valid_acc = (0.5)*100


# set log and checkpoint name 
current_time = datetime.now().strftime("%Y%m%d")
log_file_path = f"log/midfusion_affectnet_emoca_combine_{current_time}.txt"
save_MODEL_ckpt = f"ckpt/midfusion_affectnet_emoca_combine_{current_time}.pt"
save_CLS_ckpt = f"ckpt/midfusion_affectnet_emoca_combine_cls_{current_time}.pt"


with open(log_file_path, "a") as log_file:
    def log_and_print(message):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_and_print(f"Epoch {epoch + 1} start time: {epoch_start_time}")

        MODEL.train()
        total_train_correct = 0
        total_train_samples = 0
        for images, classes, labels, expressions in tqdm(train_loader, desc="Epoch train_loader progress" ):
            images, classes, labels, expressions = (images.to(DEVICE),classes.to(DEVICE),labels.to(DEVICE),expressions.to(DEVICE))
   
            optimizer.zero_grad()
            out_features = MODEL(images)
            outputs= MODEL_cls(out_features, expressions)
            outputs_cls = outputs[:, :8]
            outputs_reg = outputs[:, 8:]
            loss1 = criterion_cls_val(outputs_cls.cuda(), classes.cuda())  # weighted sampler
            loss2 = criterion_reg(outputs_reg.cuda(), labels.cuda())      
            loss = loss1 + 5*loss2 
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            _, train_predicted = torch.max(outputs_cls, 1)
            total_train_samples += classes.size(0)
            total_train_correct += (train_predicted == classes).sum().item()

        train_accuracy = (total_train_correct / total_train_samples) * 100
        log_and_print(f"Epoch {epoch + 1} train accuracy: {train_accuracy:.2f}% LR {optimizer.param_groups[0]['lr']}" )


        MODEL.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, classes, labels, expressions in valid_loader:
                images, classes, labels, expressions = (images.to(DEVICE),classes.to(DEVICE),labels.to(DEVICE),expressions.to(DEVICE))

                
                out_features = MODEL(images)
                outputs= MODEL_cls(out_features, expressions)

                outputs_cls = outputs[:, :8]
                outputs_reg = outputs[:, 8:]
                loss1 = criterion_cls_val(outputs_cls.cuda(), classes.cuda()) # note weight or not 
                loss2 = criterion_reg(outputs_reg.cuda(), labels.cuda())
                loss = loss1 + 5*loss2

                valid_loss += loss.item()
                _, predicted = torch.max(outputs_cls, 1)
                total += classes.size(0)
                correct += (predicted == classes).sum().item()

        val_acc = (correct / total) * 100
        cur_loss = valid_loss/len(valid_loader)
        log_message=(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
            f"Validation Loss: {cur_loss:.4f}, "
            f"Validation Accuracy: {val_acc:.2f}%"
            f", Training Accuracy: {train_accuracy:.2f}%, "
        )
        log_and_print(log_message)

        if cur_loss < best_valid_loss:
            best_valid_loss = cur_loss
            torch.save(MODEL.state_dict(), save_MODEL_ckpt )  # Save the best model
            torch.save(MODEL_cls.state_dict(), save_CLS_ckpt) 
            log_and_print(f"Saving model at epoch {epoch+1}, Best loss changes to : {best_valid_loss}, ckpt_name: {save_MODEL_ckpt, save_CLS_ckpt} ")  
        
