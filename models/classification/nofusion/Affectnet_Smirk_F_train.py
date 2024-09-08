import os
import sys
import json
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import random
from torchvision import transforms, datasets
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from networks.SER import SmirkEmoRecognition_F as SmirkEmoRecognition
import torch.optim.lr_scheduler as lr_scheduler


eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/home/hbml-user/XiaoWang/FaceData/affectnetAndSmirk', help='AffectNet dataset path.')
    parser.add_argument('--train_excel_path', type=str, default='/../../../data/affectnet_annotations/train_set_annotation_without_lnd.csv', help='Path to Excel file containing training labels, val, and aro.')
    parser.add_argument('--val_excel_path', type=str, default='../../../data/affectnet_annotations/val_set_annotation_without_lnd.csv', help='Path to Excel file containing validation labels, val, and aro.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--mask_prob', type=float, default=0, help='Probability of masking each feature.')
    parser.add_argument('--no_mask_in_val', action='store_true', help='Disable feature masking in validation.')


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
            'pose_params': [0.0] * 3,
            'shape_params': [0.0] * 300,
            'expression_params': [0.0] * 50,
            'eyelid_params': [0.0] * 2,
            'jaw_params': [0.0] * 3
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


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
       
        if isinstance(dataset, datasets.ImageFolder) or isinstance(dataset, CustomImageFolder):
            labels = [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            labels = [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError("Unsupported dataset type")
        
        return labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def smirk_loss(valence_true, arousal_true, expression_true, valence_pred, arousal_pred, expression_pred, alpha, beta, gamma):
    ce_loss = F.cross_entropy(expression_pred, expression_true)
    mse_valence = F.mse_loss(valence_pred, valence_true)
    mse_arousal = F.mse_loss(arousal_pred, arousal_true)
    mse = mse_valence + mse_arousal
    pcc_valence = torch.corrcoef(torch.stack([valence_true, valence_pred]))[0, 1] # 0行1列
    pcc_arousal = torch.corrcoef(torch.stack([arousal_true, arousal_pred]))[0, 1]
    pcc = (pcc_valence + pcc_arousal) / 2
    var_valence_true = torch.var(valence_true)
    var_valence_pred = torch.var(valence_pred)
    mean_valence_true = torch.mean(valence_true)
    mean_valence_pred = torch.mean(valence_pred)
    ccc_valence = 2.0 * var_valence_true * var_valence_pred * pcc_valence / (
        var_valence_true + var_valence_pred + (mean_valence_true - mean_valence_pred) ** 2.0)

    var_arousal_true = torch.var(arousal_true)
    var_arousal_pred = torch.var(arousal_pred)
    mean_arousal_true = torch.mean(arousal_true)
    mean_arousal_pred = torch.mean(arousal_pred)
    ccc_arousal = 2 * var_arousal_true * var_arousal_pred * pcc_arousal / (
        var_arousal_true + var_arousal_pred + (mean_arousal_true - mean_arousal_pred) ** 2)

    ccc = (ccc_valence + ccc_arousal) / 2
    return ce_loss + (alpha/(alpha+beta+gamma)) * mse +  (beta/(alpha+beta+gamma)) * (1 - pcc) +  (gamma/(alpha+beta+gamma))  * (1 - ccc), ce_loss


def get_dataloader(args, root, excel_path, transform, batch_size, workers, sampler=None, shuffle=False, mask=True):
    dataset = CustomImageFolder(root, excel_path, transform=transform, mask_prob=args.mask_prob, mask=mask)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=workers,
                                       sampler=sampler,
                                       shuffle=shuffle,
                                       pin_memory=True)

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = SmirkEmoRecognition()
    model.to(device)
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])
    
    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomImageFolder(root=f'{args.aff_path}/train', excel_path=args.train_excel_path, transform=data_transforms)
    train_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = get_dataloader(args, f'{args.aff_path}/train', args.train_excel_path, data_transforms, args.batch_size, args.workers, sampler=train_sampler, mask=False)
    val_loader = get_dataloader(args, f'{args.aff_path}/val_clean', args.val_excel_path, data_transforms_val, args.batch_size, args.workers, mask=False)
    model = SmirkEmoRecognition()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=2241, mode='triangular',cycle_momentum=False)
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        ce_loss_all = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for imgs, targets, features, val, aro in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device).float()
            valence_true = val.to(device).float()
            arousal_true = aro.to(device).float()
            expression_true= targets.to(device).long() 
            valence_pred,arousal_pred,expression_pred = model(imgs, features)
            alpha, beta, gamma = np.random.rand(3)
            loss,ce_loss = smirk_loss(valence_true,arousal_true, expression_true, valence_pred,arousal_pred,expression_pred , alpha,beta,gamma)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ce_loss_all += ce_loss.item()
            _, predicts = torch.max(expression_pred, 1) 
            correct_num = torch.eq(predicts, expression_true).sum()
            correct_sum += correct_num
            optimizer.step()
        acc = correct_sum.float() / float(len(train_loader.dataset))
        running_loss = running_loss / iter_cnt
        ce_loss_all = ce_loss_all / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f CE_loss: %.3f. LR %.6f' % (epoch, acc, running_loss,ce_loss_all ,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            ce_loss_all = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets, features, val, aro in val_loader:
                iter_cnt += 1
                optimizer.zero_grad()
                imgs = imgs.to(device).float()
                valence_true = val.to(device).float()
                arousal_true = aro.to(device).float()
                expression_true= targets.to(device).long()
                valence_pred,arousal_pred,expression_pred = model(imgs, features)
                alpha, beta, gamma = np.random.rand(3)
                loss,ce_loss = smirk_loss(valence_true,arousal_true, expression_true, valence_pred,arousal_pred,expression_pred , alpha,beta,gamma)
                _, predicts = torch.max(expression_pred, 1)
                correct_num = torch.eq(predicts, expression_true)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += expression_pred.size(0)
                running_loss += loss.item() 
                ce_loss_all += ce_loss.item()
            running_loss = running_loss / iter_cnt
            ce_loss_all = ce_loss_all / iter_cnt    
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            tqdm.write("[Epoch %d] Validation accuracy: %.4f. CE_Loss: %.3f Loss: %.3f" % (epoch, acc, ce_loss_all,running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if acc > best_acc:
                best_acc = acc
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join('final_checkpoint', "Affecnet_F_epoch"+str(epoch)+"_acc"+str(acc)+".pth"))
                tqdm.write('Model saved.')

if __name__ == "__main__":                    
    run_training()
