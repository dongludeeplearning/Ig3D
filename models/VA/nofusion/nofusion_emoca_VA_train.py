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

IMAGE_FOLDER = "/mnt/pub/Facial_Emotion_Dataset/AffectNet/train_set/images/"
IMAGE_FOLDER_TEST = "/mnt/pub/Facial_Emotion_Dataset/AffectNet/val_set/images/"

train_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_train_file.csv"
)
valid_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_val_file.csv"
)

train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)

class EmocaFER(nn.Module):
    def __init__(self, input_size=156, hidden_size=2048, output_size=10, dropout_probs=[0.5, 0.4, 0.3]):
        super(EmocaFER, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_probs[0])
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_probs[1])
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_probs[2])
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):  # residual connection # input is 358 dims json file
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        return output


# Set parameters
BATCHSIZE = 64
NUM_EPOCHS = 20
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, balance=False): # add 3d label 
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.balance = balance

        if self.balance:
            self.dataframe = self.balance_dataset()
            # self.dataframe = self.minbalance_dataset()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        classes = torch.tensor(self.dataframe["exp"].iloc[idx], dtype=torch.long)             
        va_labels = torch.tensor(self.dataframe.iloc[idx, 2:4].values, dtype=torch.float32)   
        expressions = torch.tensor(self.dataframe.iloc[idx, 4:160].values, dtype=torch.float32)  #dim 156 out of 334
        return classes, va_labels, expressions 

    def minbalance_dataset(self):  #min balance
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df
    
    def balance_dataset(self):   #weighted balance
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
        ),  # Should help overfitting
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_dataset = CustomDataset(
    dataframe=train_annotations_df,
    root_dir=IMAGE_FOLDER,
    transform=transform,
    balance=True,  #   weighted sample or min sample
)
valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    transform=transform_valid,
    balance=False,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=48
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=48
)

# ***** Define the model *****

# Initialize the model
MODEL = EmocaFER()
MODEL.load_state_dict(torch.load("ckpt/ckpt_affectnet_emoca_acc_20240805.pt")) 
MODEL.output= nn.Linear(2048, 2)  # change the output size
MODEL.to(DEVICE)

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
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=2241, mode='triangular',cycle_momentum=False) # cycle is better

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
best_valid_loss =5.0
l2_lambda = 0.00001  # L1 Regularization
l1_lambda = 0.00001  # L2 Regularization


# set checkpoint name 
current_time = datetime.now().strftime("%Y%m%d")
ckptname_loss = f"ckpt/nofusion_affectnet_emoca_va_{current_time}.pt"
# set log file name
log_file_path = f"log/nofusion_affectnet_emoca_va_{current_time}.txt"


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
        current_lr = optimizer.param_groups[0]["lr"]
        for classes, labels, expressions in tqdm(train_loader, desc="Epoch train_loader progress"):
            classes, labels, expressions = (
                classes.to(DEVICE),
                labels.to(DEVICE),  # va
                expressions.to(DEVICE),    
            )
            optimizer.zero_grad()
            train_loss = 0
            l2_reg = 0
            l1_reg = 0

            with torch.autocast(device_type="cuda", dtype=torch.float16):

                val_true = labels[:, 0]
                aro_true = labels[:, 1]
                outputs = MODEL(expressions)  #only train exp features
                val_pred = outputs[:, 0]
                aro_pred = outputs[:, 1]
                for param in MODEL.parameters():
                    l2_reg += torch.norm(param, 2)  # **2
                    l1_reg += torch.norm(param, 1)
                loss = (
                    3 * val_loss(val_pred.cuda(), val_true.cuda())
                    + 3 * aro_loss(aro_pred.cuda(), aro_true.cuda())
                    + CCCLoss(val_pred.cuda(), val_true.cuda())
                    + CCCLoss(aro_pred.cuda(), aro_true.cuda())
                )
              
                train_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # lr_scheduler.step()
                scheduler.step()


        log_and_print(f"Epoch {epoch + 1} train loss: {train_loss:.2f}")

        MODEL.eval()
        valid_loss = 0.0
        total_valid_correct = 0
        total_valid_samples = 0
        with torch.no_grad():
            for classes, labels, expressions in valid_loader:
                classes, labels, expressions = (
                    classes.to(DEVICE),
                    labels.to(DEVICE),  # va
                    expressions.to(DEVICE),   
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):

                    val_true = labels[:, 0]
                    aro_true = labels[:, 1]
                    outputs = MODEL(expressions)  #only train exp features
                    val_pred = outputs[:, 0]
                    aro_pred = outputs[:, 1]
                    loss = (
                        3 * val_loss(val_pred.cuda(), val_true.cuda())
                        + 3 * aro_loss(aro_pred.cuda(), aro_true.cuda())
                        + CCCLoss(val_pred.cuda(), val_true.cuda())
                        + CCCLoss(aro_pred.cuda(), aro_true.cuda())
                    )
                    valid_loss += loss.item()
    
        cur_loss = valid_loss/len(valid_loader)
        log_message=(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
            f"Training Loss: {train_loss/len(train_loader):.4f}, "      # loss/4482
            f"Validation Loss: {cur_loss:.4f}, "    # loss/63
            f"Learning Rate: {current_lr:.8f}, "
        )
        log_and_print(log_message)
        

        if cur_loss < best_valid_loss:
            best_valid_loss =  cur_loss            
            torch.save(MODEL.state_dict(), ckptname_loss)  # Save the best model
            log_and_print(f"Saving model at epoch {epoch+1}, cur_loss {cur_loss}, ckpt {ckptname_loss}")
