# ***** Train the model *****
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

# Load the annotations for training and validation from separate CSV files
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
            self.dataframe = self.minbalance_dataset()


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

    def minbalance_dataset(self):
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df
    
    def balance_dataset(self):   #weight balance
        class_counts = self.dataframe['exp'].value_counts()
        class_weights = 1.0 / class_counts
        sample_weights = self.dataframe['exp'].apply(lambda x: class_weights[x])
        balanced_dataframe = self.dataframe.sample(n=len(self.dataframe), weights=sample_weights, replace=True, random_state=42)
        return balanced_dataframe.reset_index(drop=True)


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
    balance=True, # weighted sampler
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
        # x = self.feature_extractor(x)
        additional_feature = self.additional_feature_transform(additional_feature)
        combined_features = torch.cat((x, additional_feature), dim=1)
        out = self.classifier(combined_features)
        return out


additional_feature_dim = 156  # emoca156
num_classes = 10

MODEL_cls= CustomClassifier(block_channels, additional_feature_dim, num_classes)

# load pre-trained combined model
MODEL.load_state_dict(torch.load("ckpt/midfusion_affectnet_emoca_combine_20240807.pt"))
MODEL_cls.load_state_dict(torch.load("ckpt/midfusion_affectnet_emoca_combine_cls_20240807.pt"))


MODEL_cls.classifier = nn.Sequential(
    nn.Linear(block_channels * 2, block_channels),  # Concatenate features, so dimension is doubled
    nn.Tanh(),
    nn.Linear(block_channels, 2, bias=False)  
)


MODEL.to(DEVICE) 
MODEL_cls.to(DEVICE)
LR = 1e-5


def CCCLoss(x, y):
    # Compute means
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    # Compute variances
    x_var = torch.var(x, dim=0)
    y_var = torch.var(y, dim=0)
    # Compute covariance matrix
    cov_matrix = torch.matmul(
        (x - x_mean).permute(*torch.arange(x.dim() - 1, -1, -1)), y - y_mean
    ) / (x.size(0) - 1)
    # Compute CCC
    numerator = 2 * cov_matrix
    denominator = x_var + y_var + torch.pow((x_mean - y_mean), 2)
    ccc = torch.mean(numerator / denominator)
    return 1-ccc


val_loss = nn.MSELoss()
aro_loss = nn.MSELoss()


optimizer = optim.AdamW(MODEL.parameters(), lr=LR)
# lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=BATCHSIZE * NUM_EPOCHS)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=234, mode='triangular',cycle_momentum=False)  #2241 or 234

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
l2_lambda = 0.00001  # L1 Regularization
l1_lambda = 0.00001  # L2 Regularization

best_valid_loss = 100
best_valid_acc = (0.5)*100

# set log and checkpoint name 
current_time = datetime.now().strftime("%Y%m%d")
log_file_path = f"log/midfusion_affectnet_emoca_va_{current_time}.txt"
save_MODEL_ckpt = f"ckpt/midfusion_affectnet_emoca_va_{current_time}.pt"
save_MODEL_cls_ckpt = f"ckpt/midfusion_affectnet_emoca_va_cls_{current_time}.pt"

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
            train_loss = 0
            l2_reg = 0
            l1_reg = 0

            val_true = labels[:, 0]
            aro_true = labels[:, 1]
            out_features = MODEL(images)
            outputs= MODEL_cls(out_features, expressions)
            val_pred = outputs[:, 0]
            aro_pred = outputs[:, 1]
            loss = (
                    3 * val_loss(val_pred.cuda(), val_true.cuda())
                    + 3 * aro_loss(aro_pred.cuda(), aro_true.cuda())
                    + CCCLoss(val_pred.cuda(), val_true.cuda())
                    + CCCLoss(aro_pred.cuda(), aro_true.cuda())
                    )
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            total_train_samples += classes.size(0)

        log_and_print(f"Epoch {epoch + 1} train loss: {train_loss:.2f} LR {current_lr }")

        MODEL.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, classes, labels, expressions in valid_loader:
                images, classes, labels, expressions = (images.to(DEVICE),classes.to(DEVICE),labels.to(DEVICE),expressions.to(DEVICE))

                val_true = labels[:, 0]
                aro_true = labels[:, 1]
                out_features = MODEL(images)
                outputs= MODEL_cls(out_features, expressions)
                val_pred = outputs[:, 0]
                aro_pred = outputs[:, 1]
                loss = (
                        3 * val_loss(val_pred.cuda(), val_true.cuda())
                        + 3 * aro_loss(aro_pred.cuda(), aro_true.cuda())
                        + CCCLoss(val_pred.cuda(), val_true.cuda())
                        + CCCLoss(aro_pred.cuda(), aro_true.cuda())
                        )
                valid_loss += loss.item()
                total += classes.size(0)

        cur_loss = valid_loss/len(valid_loader)
        log_message=(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
            f"Training Loss: {train_loss/len(train_loader):.4f}, " 
            f"Validation Loss: {cur_loss:.4f}, "
            f"Learning Rate: {current_lr:.8f}, "
        )
        log_and_print(log_message)

        if cur_loss < best_valid_loss:
            best_valid_loss = cur_loss
            torch.save(MODEL.state_dict(), save_MODEL_ckpt  ) 
            torch.save(MODEL_cls.state_dict(), save_MODEL_cls_ckpt )  
            log_and_print(f"Saving model at epoch {epoch+1}, Best loss changes to : {best_valid_loss}, ckpt_name: {save_MODEL_ckpt, save_MODEL_cls_ckpt} ")  
        
