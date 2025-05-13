import os
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import GroupShuffleSplit

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

def process_npz(filepath, n_band, threshold=0.005):
    """Processa um único arquivo NPZ e retorna dados + coordenadas."""
    bands_feature = int(n_band - 1)
    filename = os.path.basename(filepath).split('.')[0]

    data = np.load(filepath)
    image_array = data['array']
    lat = data['lat'].item()
    lon = data['lon'].item()

    class_array = image_array[-1:, :, :]
    image_array = image_array[:-1, :, :]

    label_1_proportion = np.sum(class_array == 1) / class_array.size

    if label_1_proportion < threshold:
        print(f"Arquivo {filename} descartado. Proporção: {label_1_proportion:.2%}")
        return None, None, None, None

    # Redimensionar se necessário
    if image_array.shape != (bands_feature, 256, 256):
        image_array = resize_image_array(image_array, (bands_feature, 256, 256))

    if class_array.shape != (1, 256, 256):
        class_array = resize_label_array(class_array, (1, 256, 256))

    return image_array, class_array, (lat, lon), filename

# Configurações
data_dir = '../../dataset_citrus/raw_data/citros_temporal_filter/temporal_raw'
output_dir = '../../dataset_citrus/raw_data/citros_temporal_filter'
npz_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]

# Estruturas para armazenamento
data_records = []
location_groups = {}
group_counter = 0

# Processar arquivos e agrupar por localização
for file in npz_files:
    img, gt, coords, filename = process_npz(file, n_band=7, threshold=0.1)
    
    if img is not None and gt is not None:
        # Criar grupos únicos por coordenada
        if coords not in location_groups:
            location_groups[coords] = group_counter
            group_counter += 1
        
        data_records.append({
            'image': img,
            'label': gt,
            'filename': filename,
            'group': location_groups[coords]
        })

# Extrair grupos para split
groups = [item['group'] for item in data_records]
filenames = [item['filename'] for item in data_records]
images = [item['image'] for item in data_records]
labels = [item['label'] for item in data_records]

# Split por grupos (garante mesma localização no mesmo conjunto)
splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_idx, val_idx = next(splitter.split(images, groups=groups))

# Função para salvar arrays
def save_arrays(indices, split_name):
    img_dir = os.path.join(output_dir, split_name, "img")
    gt_dir = os.path.join(output_dir, split_name, "gt")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    for idx in indices:
        np.save(os.path.join(img_dir, f"{filenames[idx]}.npy"), images[idx])
        np.save(os.path.join(gt_dir, f"{filenames[idx]}.npy"), labels[idx])
        print(f"Salvo {split_name}: {filenames[idx]}")

# Salvar conjuntos
save_arrays(train_idx, "train")
save_arrays(val_idx, "val")

# Estatísticas
print("\n=== Estatísticas ===")
print(f"Total de imagens processadas: {len(npz_files)}")
print(f"Imagens válidas após filtro: {len(data_records)}")
print(f"  - Treino: {len(train_idx)}")
print(f"  - Validação: {len(val_idx)}")
print(f"Localizações únicas: {len(location_groups)}")
print(f"  - No treino: {len(set(groups[i] for i in train_idx))}")
print(f"  - Na validação: {len(set(groups[i] for i in val_idx))}")