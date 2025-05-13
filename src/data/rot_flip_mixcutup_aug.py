import os
import numpy as np
import random

# Diretórios de entrada (imagens e máscaras originais 256x256)
img_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/train/img/'
gt_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/train/gt/'

# Diretórios de saída (augmentados 256x256)
aug_img_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/train/aug_mc/img'
aug_gt_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/train/aug_mc/gt'

os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_gt_dir, exist_ok=True)

# Parâmetros de transformação
angulos = [0, 90, 180, 270]
flips = ['nenhum', 'horizontal', 'vertical']

def aplicar_transformacao(img_array, mask_array, angulo, flip):
    """
    Aplica rotação e flip a uma imagem e máscara.
    """
    # Rotacionar
    img_rot = np.rot90(img_array, k=angulo // 90, axes=(1, 2))
    mask_rot = np.rot90(mask_array, k=angulo // 90, axes=(1, 2))
    
    # Aplicar flip
    if flip == 'horizontal':
        img_rot = np.flip(img_rot, axis=2)
        mask_rot = np.flip(mask_rot, axis=2)
    elif flip == 'vertical':
        img_rot = np.flip(img_rot, axis=1)
        mask_rot = np.flip(mask_rot, axis=1)
    
    return img_rot, mask_rot

def mixup(img1, mask1, img2, mask2, alpha=0.2):
    """
    Aplica mixup em imagens e máscaras binárias.
    """
    # Gerar lambda de forma aleatória seguindo distribuição Beta
    lam = np.random.beta(alpha, alpha)
    
    # Misturar imagens
    mixed_img = lam * img1 + (1 - lam) * img2
    
    # Para máscaras binárias, usa-se a média ponderada 
    mixed_mask = lam * mask1 + (1 - lam) * mask2
    
    return mixed_img, mixed_mask

def cutmix(img1, mask1, img2, mask2, size=64):
    """
    Aplica CutMix em imagens e máscaras binárias.
    """
    h, w = img1.shape[1], img1.shape[2]
    
    # Posição aleatória para o corte
    x = np.random.randint(0, w - size + 1)
    y = np.random.randint(0, h - size + 1)
    
    # Criar cópia das imagens
    mixed_img = img1.copy()
    mixed_mask = mask1.copy()
    
    # Cortar e colar região
    mixed_img[:, y:y+size, x:x+size] = img2[:, y:y+size, x:x+size]
    mixed_mask[y:y+size, x:x+size] = mask2[y:y+size, x:x+size]
    
    return mixed_img, mixed_mask

# Processar cada imagem
imagens = [f for f in os.listdir(img_dir) if f.endswith('.npy')]
for i in range(len(imagens)):
    img_name = imagens[i]
    
    # Caminhos dos arquivos
    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(gt_dir, img_name)  # Assume mesmo nome
    
    # Carregar imagem e máscara
    img_array = np.load(img_path)  # Esperado formato (6, 256, 256)
    mask_array = np.load(mask_path)  # Esperado formato (256, 256)
    
    # Transformações originais
    for angulo in angulos:
        for flip in flips:
            img_transf, mask_transf = aplicar_transformacao(img_array, mask_array, angulo, flip)
            
            # Nome único para cada transformação
            nome_base = os.path.splitext(img_name)[0]
            sufixo = f'_r{angulo}_f{flip[0]}'  # Ex: _r90_fh
            novo_nome = f"{nome_base}{sufixo}.npy"
            
            # Salvar imagens e máscaras transformadas
            np.save(os.path.join(aug_img_dir, novo_nome), img_transf)
            np.save(os.path.join(aug_gt_dir, novo_nome), mask_transf)
    
    # Mixup e CutMix
    if i + 1 < len(imagens):
        # Carregar próxima imagem para mixup/cutmix
        img_next = np.load(os.path.join(img_dir, imagens[i+1]))
        mask_next = np.load(os.path.join(gt_dir, imagens[i+1]))
        
        # Mixup
        mixed_img_mixup, mixed_mask_mixup = mixup(img_array, mask_array, img_next, mask_next)
        np.save(os.path.join(aug_img_dir, f"{nome_base}_mixup.npy"), mixed_img_mixup)
        np.save(os.path.join(aug_gt_dir, f"{nome_base}_mixup.npy"), mixed_mask_mixup)
        
        # # CutMix
        # mixed_img_cutmix, mixed_mask_cutmix = cutmix(img_array, mask_array, img_next, mask_next)
        # np.save(os.path.join(aug_img_dir, f"{nome_base}_cutmix.npy"), mixed_img_cutmix)
        # np.save(os.path.join(aug_gt_dir, f"{nome_base}_cutmix.npy"), mixed_mask_cutmix)

print("Augmentação concluída!")