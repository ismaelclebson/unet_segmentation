import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Diretórios de entrada (imagens e máscaras originais 256x256)
img_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/val/img/'
gt_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/val/gt/'

# Diretórios de saída (augmentados 256x256)
aug_img_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/val/aug/img'
aug_gt_dir = '/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_filter/val/aug/gt'

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

# Processar cada imagem
for img_name in os.listdir(img_dir):
    if not img_name.endswith('.npy'):
        continue  # Pula arquivos que não são npy

    # Caminhos dos arquivos
    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(gt_dir, img_name)  # Assume mesmo nome
    
    # Carregar imagem e máscara
    img_array = np.load(img_path)  # Esperado formato (6, 256, 256)
    mask_array = np.load(mask_path)  # Esperado formato (256, 256)
    
    # Aplicar transformações
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

print("Augmentação concluída!")
