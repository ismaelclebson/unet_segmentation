import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Diretórios das imagens aumentadas
aug_img_dir = '/home/clebson/Documentos/dataset_palmls4claymodel/data/train/aug/gt'

# Listar todas as imagens disponíveis
arquivos = [f for f in os.listdir(aug_img_dir) if f.endswith('.npy')]

if not arquivos:
    print("Nenhuma imagem encontrada no diretório de imagens aumentadas.")
else:
    # Seleciona um nome de arquivo base aleatório (sem a parte de transformação)
    nomes_bases = set(["_".join(f.split("_")[:-2]) for f in arquivos])
    nome_base = random.choice(list(nomes_bases))

    # Filtra as versões transformadas da mesma imagem
    versoes = sorted([f for f in arquivos if f.startswith(nome_base)])

    # Carregar e exibir as imagens
    fig, axes = plt.subplots(1, len(versoes), figsize=(15, 5))

    for ax, nome_arquivo in zip(axes, versoes):
        caminho = os.path.join(aug_img_dir, nome_arquivo)
        img = np.load(caminho)  # (6, 256, 256)

        print(img.shape)

        #img = ((img - img.min()) / (img.max() - img.min())*255).astype(np.uint8)

        # Exibir apenas a primeira banda como referência
        #ax.imshow(img[:3].transpose(1, 2, 0), cmap='gray')
        ax.imshow(img[0], cmap='gray')
        ax.set_title(nome_arquivo)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
