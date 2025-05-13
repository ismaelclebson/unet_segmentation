import os

import numpy as np

#from sklearn.model_selection import train_test_split

def resize_image_array(image_array, target_shape):
    """Ajusta o formato do array de imagem para o shape desejado.

    Args:
        image_array (np.ndarray): Array de imagem original.
        target_shape (tuple): Shape desejado (C, H, W).

    Returns:
        np.ndarray: Array ajustado.
    """
    from skimage.transform import resize

    resized = np.zeros(target_shape, dtype=image_array.dtype)
    for c in range(target_shape[0]):
        resized[c] = resize(
            image_array[c] if c < image_array.shape[0] else np.zeros_like(image_array[0]),
            target_shape[1:],
            mode='constant',
            preserve_range=True
        )
    return resized

def resize_label_array(label_array, target_shape):
    """Ajusta o formato do array de rótulo para o shape desejado.

    Args:
        label_array (np.ndarray): Array de rótulo original.
        target_shape (tuple): Shape desejado (1, H, W).

    Returns:
        np.ndarray: Array ajustado.
    """
    from skimage.transform import resize

    resized = resize(
        label_array[0],
        target_shape[1:],
        mode='constant',
        preserve_range=True
    )
    return resized[np.newaxis, ...]

# Define o diretório raiz dos dados
data_dir = '../../dataset_citrus/raw_data/val_raw'

# Lista para armazenar os caminhos dos arquivos .npz
npz_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]

# Define as proporções para o split
#train_ratio = 0.7
#val_ratio = 0.2
#test_ratio = 0.1

# Divide os dados em treino, validação e teste
#train_files, test_val_files = train_test_split(npz_files, train_size=train_ratio, random_state=42)
#val_files, test_files = train_test_split(test_val_files, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

def process_and_split_npz(filepath, output_dir, split, n_band, threshold=0.01, counters=None, proportion_counts=None):
    """Processa um arquivo .npz e salva as imagens e máscaras no formato .npy nos diretórios img e gt.
    Filtra imagens com base na representatividade do rótulo 1 e realiza contagem para diferentes limiares.

    Args:
        filepath (str): Caminho do arquivo .npz.
        output_dir (str): Diretório base para salvar os arquivos processados.
        split (str): Nome do conjunto ('train', 'val' ou 'test').
        n_band (int): Número de bandas totais no arquivo.
        threshold (float): Limiar mínimo de proporção de pixels com valor 1 na máscara.
        counters (dict): Dicionário para rastrear contagem total e imagens filtradas.
        proportion_counts (dict): Dicionário para contagens em diferentes limiares.
    """
    bands_feature = int(n_band - 1)
    band_label = 1
    filename = os.path.basename(filepath).split('.')[0]

    # Carrega os dados do arquivo .npz
    data = np.load(filepath)
    image_array = data['array']

    # Separar a última banda como rótulo e as demais como imagem
    class_array = image_array[-1:, :, :] # Rótulo (1, H, W)
    image_array = image_array[:-1, :, :] # Imagem (C-1, H, W)

    # Incrementa o contador total
    if counters is not None:
        counters["total"] += 1

    # Verificar a proporção do rótulo 1
    label_1_proportion = np.sum(class_array == 1) / class_array.size

    # Atualizar contagens para diferentes limiares
    if proportion_counts is not None:
        for threshold_s in proportion_counts.keys():
            if label_1_proportion >= threshold_s:
                proportion_counts[threshold_s] += 1

    # Aplicar filtro baseado no limiar
    if label_1_proportion < threshold:
        print(f"Arquivo {filename} descartado por baixa representatividade do rótulo 1 ({label_1_proportion:.2%}).")
        return

    # Incrementa o contador de imagens válidas
    if counters is not None:
        counters["filtered"] += 1

    # Ajustar o formato da imagem, se necessário
    if image_array.shape != (bands_feature, 256, 256):
        image_array = resize_image_array(image_array, (bands_feature, 256, 256))

    # Ajustar o formato do rótulo, se necessário
    if class_array.shape != (band_label, 256, 256):
        class_array = resize_label_array(class_array, (band_label, 256, 256))

    # Criar diretórios para imagens e rótulos
    img_dir = os.path.join(output_dir, split, "img")
    gt_dir = os.path.join(output_dir, split, "gt")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Salvar como .npy
    np.save(os.path.join(img_dir, f"{filename}.npy"), image_array)
    np.save(os.path.join(gt_dir, f"{filename}.npy"), class_array)

    print(f"Arquivo {filename} processado e salvo. Proporção do rótulo 1: {label_1_proportion:.2%}")

"""
Este script processa arquivos .npz contendo imagens de satélite e suas respectivas máscaras de segmentação.
O script carrega os arquivos .npz, separa as imagens das máscaras, redimensiona-as para um tamanho padrão,
filtra as imagens com base na proporção de pixels da classe 1 nas máscaras, e salva as imagens e máscaras processadas
como arquivos .npy em diretórios separados para treino, validação e teste.
"""
# Adiciona contadores para rastrear estatísticas
counters = {"total": 0, "filtered": 0}

# Contadores para diferentes limiares de proporção
proportion_counts = {0.05:0, 0.1: 0, 0.15: 0, 0.2: 0, 0.25: 0, 0.3: 0, 0.4: 0, 0.5: 0}

# Define o diretório de saída
output_dir = '../../dataset_citrus/raw_data/'

# Processa e salva os arquivos
for file in npz_files:
    process_and_split_npz(
        file,
        output_dir,
        'train',
        7,
        threshold=1, # Ajuste conforme necessário
        counters=counters,
        proportion_counts=proportion_counts,
    )

# Exibe as estatísticas finais
print(f"\nProcessamento concluído!")
print(f"Total de imagens: {counters['total']}")
print(f"Imagens após aplicação do limiar: {counters['filtered']}")
print(f"Imagens descartadas: {counters['total'] - counters['filtered']}")
print("\nContagem de imagens por limiares de proporção do rótulo 1:")
for threshold, count in proportion_counts.items():
    print(f"- Proporção >= {threshold:.0%}: {count} imagens")
