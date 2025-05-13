import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from src.models.model import UNet
import os

class GeoTIFFPredictor:
    def __init__(self, model, device, window_size=256, overlap=64):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.ramp = np.linspace(0, 1, overlap//2)
        self.band_stats = {}

    def compute_global_stats(self, src: rasterio.DatasetReader):
        """Calcula estatísticas globais para normalização"""
        self.band_stats = {}
        for band in range(1, src.count + 1):
            data = src.read(band)
            self.band_stats[band] = {
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'std': data.std()
            }

    def normalize_band(self, band, scale='mm'):
        if scale == 'mm':
            """Normalização individual de cada banda (igual ao dataset)"""
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:
                return (band - min_val) / (max_val - min_val)
            return np.zeros_like(band, dtype=np.float32)

        elif scale == 'ss':
            mean_val = band.mean()
            std_val = band.std()
            return (band - mean_val) / std_val

        elif scale == 'global_ss':
            """Normalização Z-score global (valores fixos de média e desvio padrão)"""
            global_stats = {
                0: {'mean': 9575.1400, 'std': 2269.6487},
                1: {'mean': 9333.1825, 'std': 1782.7761},
                2: {'mean': 8279.4832, 'std': 1505.7824},
                3: {'mean': 17096.3165, 'std': 3603.5337},
                4: {'mean': 14481.1129, 'std': 3851.3550},
                5: {'mean': 11567.9320, 'std': 3271.9373},
            }

            band_index = getattr(self, 'current_band', 0)  # Assume que self.current_band está sendo atualizado no loop
            if band_index in global_stats:
                mean_val = global_stats[band_index]['mean']
                std_val = global_stats[band_index]['std']
                return (band - mean_val) / std_val
            else:
                raise ValueError(f"Não há estatísticas globais definidas para a banda {band_index}")

        elif scale == 'mmpc':
            """Normalização Percentílica (Min-Max ajustado pelos percentis 2% e 98%)"""
            lower = np.percentile(band, 2)
            upper = np.percentile(band, 98)
            band = np.clip(band, lower, upper)  # Limita os valores aos percentis escolhidos
            if upper > lower:  # Evita divisão por zero
                return (band - lower) / (upper - lower)
            return np.zeros_like(band, dtype=np.float32)

        else:
            raise ValueError("Opção de normalização inválida. Use 'mm', 'ss', 'global_ss' ou 'mmpc'.")

    
    def get_blend_weights(self, window_height, window_width, y_start, x_start, src_height, src_width):
        """Gera pesos adaptativos considerando bordas da imagem"""
        weights = np.ones((window_height, window_width), dtype=np.float32)
        
        # Aplica rampa apenas se não estiver na borda correspondente
        # Topo
        if y_start > 0:
            top_ramp = self.ramp[:, np.newaxis]
            weights[:self.overlap//2, :] *= top_ramp[:window_height, :]
        
        # Base
        if y_start + window_height < src_height:
            bottom_ramp = self.ramp[::-1, np.newaxis]
            weights[-self.overlap//2:, :] *= bottom_ramp[:window_height, :]
        
        # Esquerda
        if x_start > 0:
            left_ramp = self.ramp[np.newaxis, :]
            weights[:, :self.overlap//2] *= left_ramp[:, :window_width]
        
        # Direita
        if x_start + window_width < src_width:
            right_ramp = self.ramp[np.newaxis, ::-1]
            weights[:, -self.overlap//2:] *= right_ramp[:, :window_width]
            
        return torch.from_numpy(weights).to(self.device)

    def predict_geotiff(self, input_path, output_path, return_probs=True, minmax='ss'):
        """Executa a predição completa em um arquivo GeoTIFF"""
        with rasterio.open(input_path) as src:
            # Normaliza a imagem completa antes do janelamento
            self.compute_global_stats(src)
            full_image = src.read().astype(np.float32)
            normalized_image = np.zeros_like(full_image, dtype=np.float32)
            
            for b in range(full_image.shape[0]):
                normalized_image[b] = self.normalize_band(full_image[b], minmax)
            
            full_pred = np.zeros((src.height, src.width), dtype=np.float32)
            full_count = np.zeros((src.height, src.width), dtype=np.float32)
            
            offsets = []
            # Gera coordenadas com overlap negativo para cobrir bordas
            for y in range(-self.overlap//2, src.height, self.stride):
                for x in range(-self.overlap//2, src.width, self.stride):
                    y_start = max(0, y)
                    x_start = max(0, x)
                    y_end = min(src.height, y_start + self.window_size)
                    x_end = min(src.width, x_start + self.window_size)
                    
                    # Apenas adiciona janelas válidas
                    if (y_end - y_start) > self.overlap//2 and (x_end - x_start) > self.overlap//2:
                        offsets.append((y_start, x_start, y_end, x_end))

            for y_start, x_start, y_end, x_end in tqdm(offsets, desc="Processando janelas"):
                window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                chip = normalized_image[:, y_start:y_end, x_start:x_end]
                
                # Converte para tensor
                input_tensor = torch.from_numpy(chip).unsqueeze(0).to(self.device)
                
                # Predição
                with torch.no_grad():
                    output = self.model(input_tensor)
                    pred = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Obtém dimensões reais
                h, w = pred.shape
                
                # Gera pesos adaptativos
                weights = self.get_blend_weights(
                    window_height=h,
                    window_width=w,
                    y_start=y_start,
                    x_start=x_start,
                    src_height=src.height,
                    src_width=src.width
                )
                
                # Aplica pesos
                weighted_pred = pred * weights.cpu().numpy()
                
                # Atualiza acumuladores
                full_pred[y_start:y_end, x_start:x_end] += weighted_pred
                full_count[y_start:y_end, x_start:x_end] += weights.cpu().numpy()

            # Calcula média ponderada
            full_pred = np.divide(full_pred, full_count, where=full_count>0)
            
            # Binarização
            if not return_probs:
                full_pred = (full_pred > 0.5).astype(np.uint8)

            # Salva resultado
            self.save_geotiff(output_path, full_pred, src.profile, return_probs)

    def save_geotiff(self, output_path, data, profile, return_probs):
        """Salva o resultado mantendo a projeção original"""
        profile.update({
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': 'float32' if return_probs else 'uint8',
            'nodata': None,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

def process_directory(input_dir, output_dir, predictor, return_probs=True, minmax = 'ss'):
    """Processa todos os arquivos .tif em um diretório"""
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista todos os arquivos .tif no diretório de entrada
    input_files = list(Path(input_dir).glob("*.tif"))
    
    for input_file in tqdm(input_files, desc="Processando arquivos"):
        # Define o caminho de saída
        output_file = Path(output_dir) / f"{input_file.stem}_pred.tif"
        
        # Executa a predição
        predictor.predict_geotiff(str(input_file), str(output_file), return_probs, minmax)

if __name__ == "__main__":
    model_name = 'checkpoint_epoch_9_acc_0.7405_loss_0.2396_ss_augtt_30'
    checkpoint_path = f"src/models/checkpoints/{model_name}.pth"
    input_dir = "image/img_point_median"  # Diretório contendo os arquivos .tif
    output_dir = f"image/img_point_median/{model_name}"  # Diretório para salvar as predições
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega o modelo
    model = UNet(in_channels=6).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Cria o predictor
    predictor = GeoTIFFPredictor(model, device, window_size=256, overlap=64)

    # Processa todos os arquivos no diretório
    process_directory(input_dir, output_dir, predictor, return_probs=True, minmax='ss')
