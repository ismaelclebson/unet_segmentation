# prompt: quero visualizar imagem e label lado a lado. Quero ver 5 pelo menos

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Diretórios de entrada e saída (ajuste conforme necessário)
output_img_dir = '/content/imagens'
output_label_dir = '/content/rotulos'

# Número de imagens a exibir
num_images_to_show = 14

# Função para exibir imagens e rótulos lado a lado
def show_images_and_labels(image_dir, label_dir, num_images):
    image_files = sorted(os.listdir(image_dir))  # Ordena os arquivos para garantir consistência
    label_files = sorted(os.listdir(label_dir))

    num_images = min(num_images, len(image_files), len(label_files))

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        image_path = os.path.join(image_dir, image_files[i])
        label_path = os.path.join(label_dir, label_files[i])

        try:
            image = np.load(image_path)
            label = np.load(label_path)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(label, cmap='gray')  # Exibe o rótulo em escala de cinza
            axes[i, 1].set_title(f"Label {i+1}")
            axes[i, 1].axis('off')

        except FileNotFoundError:
            print(f"Arquivo não encontrado: {image_path} ou {label_path}")
        except Exception as e:
            print(f"Erro ao exibir imagens: {e}")

    plt.tight_layout()
    plt.show()

# Chama a função para exibir as imagens e rótulos
show_images_and_labels(output_img_dir, output_label_dir, num_images_to_show)
