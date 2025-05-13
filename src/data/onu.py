import os
import numpy as np
from skimage.transform import resize

def resize_image_array(image_array, target_shape):
    """Ajusta o formato do array de imagem para o shape desejado."""
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
    """Ajusta o formato do array de rótulo para o shape desejado."""
    resized = resize(
        label_array[0],
        target_shape[1:],
        mode='constant',
        preserve_range=True
    )
    return resized[np.newaxis, ...]

def process_and_split_npz(filepath, output_dir, split, n_band, threshold=0.01, 
                         counters=None, proportion_counts=None, unique_coords=None):
    """Processa arquivos .npz com coleta de coordenadas únicas."""
    bands_feature = int(n_band - 1)
    band_label = 1
    filename = os.path.basename(filepath).split('.')[0]

    data = np.load(filepath)
    image_array = data['array']
    
    # Extrair coordenadas
    lat = data['lat'].item()
    lon = data['lon'].item()

    class_array = image_array[-1:, :, :]
    image_array = image_array[:-1, :, :]

    if counters is not None:
        counters["total"] += 1

    label_1_proportion = np.sum(class_array == 1) / class_array.size

    if proportion_counts is not None:
        for threshold_s in proportion_counts.keys():
            if label_1_proportion >= threshold_s:
                proportion_counts[threshold_s] += 1

    if label_1_proportion < threshold:
        print(f"Arquivo {filename} descartado. Proporção: {label_1_proportion:.2%}")
        return

    # Registrar coordenadas se passar no filtro
    if unique_coords is not None:
        unique_coords.add((lat, lon))

    if counters is not None:
        counters["filtered"] += 1

    if image_array.shape != (bands_feature, 256, 256):
        image_array = resize_image_array(image_array, (bands_feature, 256, 256))

    if class_array.shape != (band_label, 256, 256):
        class_array = resize_label_array(class_array, (band_label, 256, 256))

    img_dir = os.path.join(output_dir, split, "img")
    gt_dir = os.path.join(output_dir, split, "gt")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    np.save(os.path.join(img_dir, f"{filename}.npy"), image_array)
    np.save(os.path.join(gt_dir, f"{filename}.npy"), class_array)

    print(f"Processado: {filename} | Proporção 1: {label_1_proportion:.2%}")

# Configurações principais
data_dir = '../../dataset_citrus/raw_data/citros_median_filter/median_raw'
output_dir = '../../dataset_citrus/raw_data/citros_median_filter'
npz_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]

# Inicializar estruturas de monitoramento
counters = {"total": 0, "filtered": 0}
proportion_counts = {0.01:0, 0.05:0, 0.1:0, 0.15:0, 0.2:0, 0.25:0, 0.3:0, 0.4:0, 0.5:0}
unique_coords = set()

# Processar todos os arquivos
for file in npz_files:
    process_and_split_npz(
        file,
        output_dir,
        split='val',
        n_band=7,
        threshold=1,
        counters=counters,
        proportion_counts=proportion_counts,
        unique_coords=unique_coords
    )

# Relatório final
print("\n=== Estatísticas Finais ===")
print(f"Total de imagens processadas: {counters['total']}")
print(f"Imagens válidas após filtro: {counters['filtered']}")
print(f"Coordenadas únicas identificadas: {len(unique_coords)}")
print("\nDistribuição por limiares:")
for thresh, count in sorted(proportion_counts.items()):
    print(f"- ≥{thresh:.0%}: {count} imagens")

# Opcional: Mostrar coordenadas únicas
print("\nCoordenadas únicas (lat, lon):")
print(len(unique_coords))
