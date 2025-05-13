import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision import transforms
import random

def set_seed(seed=42):
    """
    Define a semente apenas para a divisão de treino/validação.
    """
    random.seed(seed)  # Semente para operações aleatórias do Python
    np.random.seed(seed)  # Semente para NumPy
    torch.manual_seed(seed)  # Semente para PyTorch

# Define a semente ANTES do random_split
set_seed(42)

class UniqueDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None, scale='mm'):
        """
        Dataset para imagens Landsat e máscaras correspondentes.
        :param img_dir: Diretório com as imagens de entrada (7 bandas do Landsat).
        :param gt_dir: Diretório com as máscaras de ground truth.
        :param transform: Transformações a serem aplicadas.
        :param scale: Tipo de normalização ('mm', 'ss' ou 'mmpc').
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.scale = scale
        self.img_names = os.listdir(img_dir)  # Lista de nomes das imagens

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        gt_path = os.path.join(self.gt_dir, self.img_names[idx])


                # Verifica se os arquivos existem antes de tentar carregar
        if not os.path.exists(img_path):
            print(f"ERRO: Arquivo de imagem não encontrado: {img_path}")
        
        if not os.path.exists(gt_path):
            print(f"ERRO: Arquivo de máscara não encontrado: {gt_path}")

        #image = np.load(img_path).astype(np.float32) # (7, 256, 256)

        try:
            image = np.load(img_path).astype(np.float32)  # (7, 256, 256)
        except ValueError as e:
            print(f"[ERRO] Falha ao carregar: {img_path}")
            raise e

        mask = np.load(gt_path).astype(np.uint8)

        # Normalização
        if self.scale == 'mm':
            for i in range(image.shape[0]):  
                min_val = image.min()
                max_val = image.max()
                image[i] = ((image[i] - min_val)) / (max_val - min_val) if max_val > min_val else 0
        elif self.scale == 'ss':
            for i in range(image.shape[0]):  
                mean_val = image[i].mean()
                std_val = image[i].std()
                image[i] = ((image[i] - mean_val)) / (std_val)
        elif self.scale == 'mmpc':
            for i in range(image.shape[0]):  
                lower = np.percentile(image[i], 2)
                upper = np.percentile(image[i], 98)
                image[i] = np.clip(image[i], lower, upper)
                image[i] = (image[i] - lower) / (upper - lower) if upper > lower else 0
        
        elif self.scale == 'div255':
            image = image / 255.0
        

        # Aplica transformações na imagem
        if self.transform:
            image = image.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
            image = self.transform(image)
        else:
            ###### (tratamento para img transformados de tif)
            image = image.transpose(2, 0, 1)  # (C, H, W) → (H, W, C)
            ######
            image = torch.from_numpy(image)  

        # Converte máscara para tensor
        #mask = torch.from_numpy(mask)
        ###### tratamento para mask gerados de tif
        mask = torch.from_numpy(mask).unsqueeze(0)
        ######
        #print(mask.shape)
        #print(image.shape)
  

        return image, mask


class AugmentDataset(Dataset):
    def __init__(self, dataset, augmentation="both"):
        """
        Aplica data augmentation ao dataset original.
        :param dataset: Dataset original (treino ou validação).
        :param augmentation: Tipo de augmentation ('rotation', 'flip', 'both' ou None).
        """
        self.dataset = dataset
        self.augmentation = augmentation
        self.rotations = [0, 90, 180, 270]
        self.transforms = []

        if augmentation in ["rotation", "both"]:
            self.transforms.extend([T.RandomRotation(degrees=[angle, angle]) for angle in self.rotations])

        if augmentation in ["flip", "both"]:
            self.transforms.append(T.RandomHorizontalFlip(p=1.0))
            self.transforms.append(T.RandomVerticalFlip(p=1.0))

    def __len__(self):
        return len(self.dataset) * (len(self.transforms) + 1)

    def __getitem__(self, idx):
        original_idx = idx % len(self.dataset)  
        image, mask = self.dataset[original_idx]  

        # Aplica uma transformação específica com base no índice
        transform_idx = idx // len(self.dataset)
        if transform_idx > 0:
            transform = self.transforms[transform_idx - 1]
            image = transform(image)
            mask = transform(mask)

        return image, mask


def get_dataloaders(train_img_dir, train_gt_dir, val_img_dir=None, val_gt_dir=None, batch_size=8, n_workers=4, 
                    transforms_flag=False, scale='mm', split_data=False, split_ratio=0.8, aug_t=None, aug_v=None):
    """
    Cria dataloaders para treino e validação (ou treino e teste).
    
    :param train_img_dir: Diretório das imagens de treino.
    :param train_gt_dir: Diretório das máscaras de treino.
    :param val_img_dir: Diretório das imagens de validação (pode ser None se split_data=True).
    :param val_gt_dir: Diretório das máscaras de validação (pode ser None se split_data=True).
    :param batch_size: Tamanho do batch.
    :param n_workers: Número de workers para carregamento de dados.
    :param transforms_flag: Se True, aplica normalização específica nas imagens.
    :param scale: Método de normalização ('mm', 'ss', 'mmpc').
    :param split_data: Se True, divide automaticamente o conjunto de treino em treino e validação.
    :param split_ratio: Percentual do conjunto de treino usado para treino (ex: 0.8 = 80% treino, 20% validação).
    :param aug_t: Tipo de augmentation no treinamento ('rotation', 'flip', 'both' ou None).
    :param aug_v: Tipo de augmentation na validação ('rotation', 'flip', 'both' ou None).
    
    :return: Dataloaders para treino e validação.
    """

    means = [9575.1400, 9333.1825, 8279.4832, 17096.3165, 14481.1129, 11567.9320]
    stds = [2269.6487, 1782.7761, 1505.7824, 3603.5337, 3851.3550, 3271.9373]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
    ]) if transforms_flag else None

    full_dataset = UniqueDataset(train_img_dir, train_gt_dir, transform=transform, scale=scale)

    if split_data:
        train_size = int(split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        if val_img_dir and val_gt_dir:
            val_dataset = UniqueDataset(val_img_dir, val_gt_dir, transform=transform, scale=scale)
        else:
            raise ValueError("Se 'split_data' for False, val_img_dir e val_gt_dir devem ser fornecidos!")

    # Aplica augmentation apenas ao conjunto de treino
    if aug_t:
        train_dataset = AugmentDataset(train_dataset, aug_t)
    if aug_v:
        val_dataset = AugmentDataset(val_dataset, aug_v)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return train_loader, val_loader
