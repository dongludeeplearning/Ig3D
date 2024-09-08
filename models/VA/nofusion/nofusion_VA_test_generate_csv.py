import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from datetime import datetime
from PIL import Image

# Load the annotations for training and validation from separate CSV files

IMAGE_FOLDER = "/mnt/pub/Facial_Emotion_Dataset/AffectNet/train_set/images/"
IMAGE_FOLDER_TEST = "/mnt/pub/Facial_Emotion_Dataset/AffectNet/val_set/images/"

valid_annotations_path = (
    "../../../data/affectnet_annotations/emoca334_merged_val_file.csv"
)
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

    def forward(self, x):  # residual connection #输入358 json file
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output(x)
        return output

# Set parameters
BATCHSIZE = 128
# MODEL = models.maxvit_t(weights="DEFAULT")
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
    
    def balance_dataset(self):   #weight balance
        class_counts = self.dataframe['exp'].value_counts()
        class_weights = 1.0 / class_counts
        sample_weights = self.dataframe['exp'].apply(lambda x: class_weights[x])
        balanced_dataframe = self.dataframe.sample(n=len(self.dataframe), weights=sample_weights, replace=True, random_state=42)
        return balanced_dataframe.reset_index(drop=True)



transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    transform=transform_valid,
    balance=False,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=48
)

# ***** Define the model *****
MODEL = EmocaFER()
MODEL.output= nn.Linear(2048, 2)  # change the output size
model_name = "ckpt/ckpt_affectnet_emoca_va_20240805.pt"
# save infer file 
current_time = datetime.now().strftime("%Y%m%d")
save_infer = f"../infer_csv/nofusion_VA_inference_{current_time}.csv"
MODEL.load_state_dict(torch.load(model_name)) 
MODEL.to(DEVICE)
MODEL.eval()

all_labels_cls = []
all_predicted_cls = []

all_true_val = []
all_pred_val = []
all_true_aro = []
all_pred_aro = []

all_val_true_values = []
all_val_predicted_values = []
all_aro_true_values = []
all_aro_predicted_values = []

# Start inference on test set
with torch.no_grad():
    for classes, labels, expressions in valid_loader:
        classes, labels, expressions  = ( classes.to(DEVICE), labels.to(DEVICE),expressions.to(DEVICE))
        val_true = labels[:, 0]
        aro_true = labels[:, 1]
        outputs = MODEL(expressions)
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
df.to_csv(save_infer , index=False)
print("save inference success.")
