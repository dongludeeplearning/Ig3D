import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.utils.data as data
from networks.DDAM import DDAMNet as DDAM
from networks.SER import EmocaEmoRecognition_S as EmocaRecognition
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import json
from PIL import Image
import pandas as pd
import random
from datetime import datetime
import pdb

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/home/hbml-user/XiaoWang/FaceData/affectnetAndEmoca/', help='Emoca dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--no_mask_in_val', action='store_true', help='Disable feature masking in validation.')
    parser.add_argument('--mask_prob', type=float, default=0.1, help='Probability of masking each feature.')
    parser.add_argument('--base_model_path', default = 'final_checkpoint/affecnet8sota.pth')
    parser.add_argument('--featured_model_path', default = 'final_checkpoint/Affecnet_Emoca_S.pth')
    parser.add_argument('--val_excel_path', type=str, default='../../../data/val_set_annotation_without_lnd.csv', help='Path to Excel file containing validation labels, val, and aro.')
    return parser.parse_args()

class CustomImageFolder(Dataset):
    def __init__(self, root, excel_path, transform=None, num_classes=8, mask_prob=0.0, mask=True):
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        self.mask_prob = mask_prob
        self.mask = mask
        self.imgs = self._load_images(root)
        self.labels_df = pd.read_csv(excel_path)  
        self.default_keys = {
            'exp': [0.0] * 50,#
            'pose': [0.0] * 6, #
            'shape': [0.0] * 100 #
        }

    def _load_images(self, root):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        imgs = []
        for subdir, dirs, files in os.walk(root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    path = os.path.join(subdir, file)
                    label = self._get_label(subdir)
                    imgs.append((path, label))
        return imgs
    def _log_missing_csv(self, filename):
        log_file_path = './logs.txt'
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as log_file:
                log_file.write("Log for missing CSV files:\n")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Missing CSV for image: {filename}\n")
    
    def _mask_features(self, features):
        if not self.mask:
            return features
        
        for key, value in features.items():
            if random.random() < self.mask_prob:
                num_elements = len(value)
                num_mask = random.randint(1, num_elements)
                mask_indices = random.sample(range(num_elements), num_mask)
                for idx in mask_indices:
                    value[idx] = 0.0
        return features

    def _get_label(self, subdir):
        try:
            label = int(os.path.basename(subdir))
            if label < 0 or label >= self.num_classes:
                raise ValueError(f"Label {label} out of range for num_classes {self.num_classes}")
        except ValueError as e:
            print(f"Error parsing label for directory {subdir}: {e}")
            label = -1  
        return label

    def __getitem__(self, index): 
        path, label = self.imgs[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        json_path = os.path.splitext(path)[0] + '.json'
        features = {}
         
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                features = json.load(f)
            if 'cam' in features:
                del features['cam']
            if 'tex' in features:
                del features['tex']
            if 'detail' in features:
                del features['detail']
        for key, default_value in self.default_keys.items():
            if key not in features:
                features[key] = default_value
        features = self._mask_features(features)

        if label < 0 or label >= self.num_classes:
            raise ValueError(f"Label {label} out of range for num_classes {self.num_classes}")

        filename = os.path.basename(path)
        number = int(os.path.splitext(filename)[0])
        row = self.labels_df[self.labels_df['number'] == number]
        if not row.empty:
            val = row['val'].values[0]
            aro = row['aro'].values[0]
        else:
            self._log_missing_csv(filename)

        return image, label, features, val, aro
    
    def __len__(self):
        return len(self.imgs)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()

class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

def get_dataloader(args, root, excel_path, transform, batch_size, workers, sampler=None, shuffle=False, mask=True):
    dataset = CustomImageFolder(root, excel_path, transform=transform, mask_prob=args.mask_prob, mask=mask)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=workers,
                                       sampler=sampler,
                                       shuffle=shuffle,
                                       pin_memory=True)
    

def run_test():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_base = DDAM()
    model_emoca = EmocaRecognition()
    base_model_checkpoint = torch.load(args.base_model_path, map_location=device)
    smirk_model_checkpoint = torch.load(args.featured_model_path, map_location=device)
    model_base.load_state_dict(base_model_checkpoint['model_state_dict'])
    model_emoca.load_state_dict(smirk_model_checkpoint['model_state_dict'])
    model_base.to(device)
    model_base.eval()
    model_emoca.to(device)
    model_emoca.eval()
    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_base_loader = get_dataloader(args, f'{args.aff_path}/val', args.val_excel_path, data_transforms_val, args.batch_size, args.workers, mask=False)

    val_smirk_loader = get_dataloader(args, f'{args.aff_path}/val', args.val_excel_path, data_transforms_val, args.batch_size, args.workers, mask=False)

    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
   
    for (imgs_base, targets_base, features_base,val, aro), (imgs_smirk, targets_smirk, features_smirk,val, aro) in zip(val_base_loader, val_smirk_loader):
        imgs_base = imgs_base.to(device)
        
        imgs_smirk = imgs_smirk.to(device)
        targets_base = targets_base.to(device)
        out_base = model_base(imgs_base, features_base)
        out_emoca = model_emoca(imgs_smirk, features_smirk)
        out_emoca_primary = out_emoca[2]
        out_fused = out_emoca_primary
        _, predicts = torch.max(out_fused, 1)
        correct_num = torch.eq(predicts, targets_base)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out_fused.size(0)

        if iter_cnt == 0:
            all_predicted = predicts
            all_targets = targets_base
        else:
            all_predicted = torch.cat((all_predicted, predicts), 0)
            all_targets = torch.cat((all_targets, targets_base), 0)
        iter_cnt += 1
    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)
    print("Validation accuracy:%.4f. " % (acc))
  
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class8_names, normalize=True, title='Affecnet_Emoca_S_Confusion Matrix (acc: %0.2f%%)' % (acc * 100))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join('FusionConfusionMatrix', f"Affecnet_Emoca_S_acc{acc}_{timestamp}.png"))
    plt.close()
    all_targets_np = all_targets.data.cpu().numpy()
    all_predicted_np = all_predicted.cpu().numpy()
    
    precision = precision_score(all_targets_np, all_predicted_np, average='weighted')
    recall = recall_score(all_targets_np, all_predicted_np, average='weighted')
    f1 = f1_score(all_targets_np, all_predicted_np, average='weighted')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    class_report = classification_report(all_targets_np, all_predicted_np, target_names=class8_names)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    run_test()
