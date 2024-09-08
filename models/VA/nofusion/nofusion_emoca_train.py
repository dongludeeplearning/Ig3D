import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from tqdm import tqdm
import time 
from pdb import set_trace as st
from datetime import datetime

# Load the annotations for training and validation from separate CSV files

IMAGE_FOLDER = "/mnt/pub/Facial_Emotion_Dataset/AffectNet/train_set/images/"
IMAGE_FOLDER_TEST = "/mnt/pub/Facial_Emotion_Dataset/AffectNet/val_set/images/"


train_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_train_file.csv"
)
valid_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_val_file.csv"
)


# 'emoca334_merged_train_file.csv'
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

    def forward(self, x):  # residual connection # 358 dims
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        return output


# Set parameters and initial model
BATCHSIZE = 64
NUM_EPOCHS = 20
LR = 1e-5
MODEL = EmocaFER ()
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
        expressions = torch.tensor(self.dataframe.iloc[idx, 4:160].values, dtype=torch.float32)  #dim 156 out pf 334
        return classes, va_labels, expressions 

    def minbalance_dataset(self):  # min balance
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df
    
    def balance_dataset(self):   # weighted balance
        class_counts = self.dataframe['exp'].value_counts()
        class_weights = 1.0 / class_counts
        sample_weights = self.dataframe['exp'].apply(lambda x: class_weights[x])
        balanced_dataframe = self.dataframe.sample(n=len(self.dataframe), weights=sample_weights, replace=True, random_state=42)
        return balanced_dataframe.reset_index(drop=True)



# no used 
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

def collate_fn(batch):
    # filter out all 0 in data of expressions
    batch = [item for item in batch if not torch.all(item[2] == 0)]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    classes, labels, expressions = zip(*batch)

    # images = torch.stack(images)
    classes = torch.stack(classes)
    labels = torch.stack(labels)
    expressions = torch.stack(expressions)
    return classes, labels, expressions

train_dataset = CustomDataset(
    dataframe=train_annotations_df,
    root_dir=IMAGE_FOLDER,
    transform=transform,
    balance=False,  # balabce training 
)
valid_dataset = CustomDataset(
    dataframe = valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    transform=transform_valid,
    balance=False,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
)

# ***** Define the model *****
MODEL.to(DEVICE)  

# Define (weighted) loss function
weights = torch.tensor(
    [0.015605, 0.008709, 0.046078, 0.083078, 0.185434, 0.305953, 0.046934, 0.30821]
)
criterion_cls = nn.CrossEntropyLoss(weights.to(DEVICE))
criterion_cls_val = (nn.CrossEntropyLoss())  # Use two loss functions, as the validation dataset is balanced
criterion_reg = nn.MSELoss()




optimizer = optim.AdamW(MODEL.parameters(), lr=LR)
# lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=BATCHSIZE * NUM_EPOCHS)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=2241, mode='triangular',cycle_momentum=False)



# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
best_valid_loss = 50
best_valid_acc = (0.5)*100

# set checkpoint name 
current_time = datetime.now().strftime("%Y%m%d")
ckptname_acc = f"ckpt/nofusion_affectnet_emoca_acc_{current_time}.pt"
# set log file name
log_file_path = f"log/nofusion_affectnet_emoca_acc_{current_time}.txt"


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
        for classes, labels, expressions in tqdm(    # add exp data 
            train_loader, desc="Epoch train_loader progress"
        ):
            
            classes, labels, expressions = (
                
                classes.to(DEVICE),
                labels.to(DEVICE),  # va
                expressions.to(DEVICE),    # add exp data 
            )
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = MODEL(expressions)
                outputs_cls = outputs[:, :8]
                outputs_reg = outputs[:, 8:10]

                loss1 = criterion_cls(outputs_cls.cuda(), classes.cuda())  # weighted los
                loss2 = criterion_reg(outputs_reg.cuda(), labels.cuda())
                loss = loss1 + 5*loss2 
        


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                # lr_scheduler.step()
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
            for classes, labels, expressions in valid_loader: # add exp data 
                classes, labels, expressions = (
                    classes.to(DEVICE),
                    labels.to(DEVICE),
                    expressions.to(DEVICE),    # add exp data 
                )
                outputs = MODEL(expressions)
                outputs_cls = outputs[:, :8]
                outputs_reg = outputs[:, 8:10]

                loss1 = criterion_cls_val(outputs_cls.cuda(), classes.cuda()) # note weight or not 
                loss2 = criterion_reg(outputs_reg.cuda(), labels.cuda())
                loss = loss1 + 5*loss2 


                valid_loss += loss.item() # loss1 only for classification
                _, predicted = torch.max(outputs_cls, 1)
                total += classes.size(0)
                correct += (predicted == classes).sum().item()


        val_acc = (correct / total) * 100
        val_loss = valid_loss/len(valid_loader)
        log_message=(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_acc:.2f}%"
            f", Training Accuracy: {train_accuracy:.2f}%, "
        )
        log_and_print(log_message)

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            print(f"Saving model at epoch {epoch+1}")      
            torch.save(MODEL.state_dict(), ckptname_acc)  #Save the best model
            log_and_print(f"Saving model at epoch {epoch+1}, Best_Acc changes to : {best_valid_acc:.2f}, ckpt_name: {ckptname_acc}")  
            

