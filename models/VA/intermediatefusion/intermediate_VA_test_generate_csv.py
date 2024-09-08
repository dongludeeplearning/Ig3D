import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from PIL import Image
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
            # self.dataframe = self.balance_dataset()
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


class CustomClassifier(nn.Module):
    def __init__(self, block_channels, additional_feature_dim, num_classes=2):
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


additional_feature_dim = 156  # emoca156
num_classes = 2


MODEL_cls= CustomClassifier(block_channels, additional_feature_dim, num_classes)
# load pre-trained models
MODEL.load_state_dict(torch.load("ckpt/midfusion_affectnet_emoca_va_20240807.pt"))
MODEL_cls.load_state_dict(torch.load("ckpt/midfusion_affectnet_emoca_va_cls_20240807.pt"))
print("load ckpt success")

 
MODEL.to(DEVICE)  
MODEL_cls.to(DEVICE)
MODEL.eval()
MODEL_cls.eval()

# set the name 
current_time = datetime.now().strftime("%Y%m%d")
save_csv = f"../infer_csv/midfusion_VA_inference_{current_time}.csv"


all_val_true_values = []
all_val_predicted_values = []
all_aro_true_values = []
all_aro_predicted_values = []

# Start inference on test set
with torch.no_grad():
    for images, classes, labels, expressions in valid_loader:
        images, classes, labels, expressions = (images.to(DEVICE),classes.to(DEVICE),labels.to(DEVICE),expressions.to(DEVICE))

        val_true = labels[:, 0]
        aro_true = labels[:, 1]
        out_features = MODEL(images)
        outputs= MODEL_cls(out_features, expressions)
        val_pred = outputs[:, 0]
        aro_pred = outputs[:, 1]

        # Append to the lists --> Regression
        true_val_values = val_true.cpu().numpy()
        true_aro_values = aro_true.cpu().numpy()
        pred_val_values = val_pred.cpu().numpy()
        pred_aro_values = aro_pred.cpu().numpy()
        all_val_true_values.extend(true_val_values)
        all_aro_true_values.extend(true_aro_values)
        all_val_predicted_values.extend(pred_val_values)
        all_aro_predicted_values.extend(pred_aro_values)
df = pd.DataFrame(
    {
        "val_pred": all_val_predicted_values,
        "val_true": all_val_true_values,
        "aro_pred": all_aro_predicted_values,
        "aro_true": all_aro_true_values,
    }
)
df.to_csv(save_csv , index=False)
print("save successfully!") 

