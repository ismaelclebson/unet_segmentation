"""
Training Script for U-Net Model with Enhanced Progress Tracking

Melhorias incluem:
- Barras de progresso com tqdm
- Tracking de tempo de execução
- Monitoramento de memória GPU
- Exibição de estatísticas detalhadas
- Estimativa de tempo restante (ETA)
- Formatação profissional das métricas
- Early Stopping com paciência de 10 épocas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
from src.models.model import UNet
from src.data.uniqueDataset import get_dataloaders
import segmentation_models_pytorch as smp
from src.scripts.utils import JointLoss, ModelCheckpoint
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import torch
import gc

# Libera memória antes de iniciar o treinamento
gc.collect()
torch.cuda.empty_cache()

# Configurações globais
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "src/models/checkpoints"
EARLY_STOP_PATIENCE = 15  # Número de épocas sem melhoria para parada antecipada

# Configuração inicial
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Função para salvar checkpoint
def save_last_checkpoint(model, optimizer, epoch, checkpoint_dir):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, "checkpoint_last.pth"))

# Carregamento de dados
train_loader, val_loader = get_dataloaders(
    train_img_dir="/home/clebson/Documentos/rice_dataset/img/",
    train_gt_dir="/home/clebson/Documentos/rice_dataset/gt/",
    # val_img_dir="/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_sp/val/aug/img",
    # val_gt_dir="/home/clebson/Documentos/dataset_citrus/raw_data/citros_temporal_sp/val/aug/gt/",
    batch_size=BATCH_SIZE,
    scale='div255',
    split_data=True,
    split_ratio=0.7,
    aug_t=None,
    aug_v=None
)

# Inicialização do modelo
model = UNet(in_channels=3).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
metric_1 = BinaryJaccardIndex().to(DEVICE)
metric_2 = BinaryF1Score().to(DEVICE)

# Definição das funções de perda
#focal_loss = smp.losses.FocalLoss(mode="binary")
#iou_loss = smp.losses.JaccardLoss(mode="binary")
#criterion = JointLoss(focal_loss, iou_loss, weight1=0.5, weight2=0.5).to(DEVICE)

criterion = smp.losses.FocalLoss(mode='binary')

# Verificação de checkpoints existentes
start_epoch = 0
last_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_last.pth")
if os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"\n▶ Retomando treinamento da época {start_epoch+1}")

# Configuração inicial do treinamento
checkpoint_callback = ModelCheckpoint(CHECKPOINT_DIR, max_saves=1)
total_start_time = time.time()
best_val_loss = float('inf')
patience_counter = 0  # Contador para early stopping
lr_patience_counter = 0  # Novo contador para redução do learning rate

# Cabeçalho informativo
print(f"\n{'='*60}")
print(f"Treinamento Iniciado | Dispositivo: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE:.0e}")
print(f"Épocas: {NUM_EPOCHS} | Checkpoints: {CHECKPOINT_DIR}")
print(f"Early Stopping: {EARLY_STOP_PATIENCE} épocas sem melhoria")
print(f"{'='*60}\n")

# Loop principal de treinamento
for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    
    # Containers para métricas
    batch_times, batch_losses, batch_accs, batch_accs_2 = [], [], [], []

    # Barra de progresso do treino
    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1:03d}/{NUM_EPOCHS:03d} [Treino]",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        unit="batch"
    )

    for images, masks in train_bar:
        batch_start_time = time.time()
        # Movimentação dos dados
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Cálculo de métricas
        with torch.no_grad():
            acc_1 = metric_1(outputs, masks)
            acc_2 = metric_2(outputs, masks)
        
        # Atualização de estatísticas
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        batch_losses.append(loss.item())
        batch_accs.append(acc_1.item())
        batch_accs_2.append(acc_2.item())
        
        # Atualização da barra de progresso
        train_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{acc_1.item():.4f}",
            'dice': f"{acc_2.item():.4f}",
            'btime': f"{batch_time:.2f}s",
            'lr': optimizer.param_groups[0]['lr']
        })

    # Cálculo de métricas do treino
    avg_train_loss = sum(batch_losses) / len(train_loader)
    avg_train_acc = sum(batch_accs) / len(train_loader)
    avg_train_acc_2 = sum(batch_accs_2) / len(train_loader)
    epoch_time = time.time() - epoch_start_time

    # Validação
    model.eval()
    val_loss, val_acc, val_acc_2 = 0.0, 0.0, 0.0
    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1:03d}/{NUM_EPOCHS:03d} [Validação]",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        unit="batch"
    )

    with torch.no_grad():
        for images, masks in val_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            acc_1 = metric_1(outputs, masks)
            acc_2 = metric_2(outputs, masks)
            
            val_loss += loss.item()
            val_acc += acc_1.item()
            val_acc_2 += acc_2.item()
            
            val_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{acc_1.item():.4f}",
                'dice': f"{acc_2.item():.4f}"
            })

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    avg_val_acc_2 = val_acc_2 / len(val_loader)

    # Lógica de Early Stopping e redução do learning rate
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        lr_patience_counter = 0  # Resetar ambos os contadores
        checkpoint_callback(model, avg_val_acc, avg_val_loss, epoch + 1)
    else:
        patience_counter += 1
        lr_patience_counter += 1  # Incrementar contador do learning rate

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early Stopping ativado na época {epoch+1}! Sem melhoria há {EARLY_STOP_PATIENCE} épocas consecutivas.")
            break

        # Redução do learning rate
        if lr_patience_counter >= 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5  # Reduzir pela metade
            print(f"\n⏬ Learning rate reduzido para {param_group['lr']} após 5 épocas sem melhoria.")
            lr_patience_counter = 0  # Resetar o contador

    # Cálculo de estatísticas de tempo
    total_time = time.time() - total_start_time
    avg_batch_time = sum(batch_times)/len(batch_times)
    epochs_left = NUM_EPOCHS - (epoch + 1)
    eta = epochs_left * (total_time / (epoch + 1 - start_epoch)) if epoch > start_epoch else 0

    # Monitoramento de memória
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        torch.cuda.reset_peak_memory_stats()
    else:
        gpu_mem = 0

    # Exibição de resultados
    print(f"\nEPOCH {epoch+1:03d}/{NUM_EPOCHS:03d} [{epoch_time:.1f}s]")
    print(f"  Train => Loss: {avg_train_loss:.4f} | iou: {avg_train_acc:.4f} | dice: {avg_train_acc_2:.4f}")
    print(f"  Valid => Loss: {avg_val_loss:.4f} | iou: {avg_val_acc:.4f} | dice: {avg_val_acc_2:.4f}")
    print(f"  Paciência: {patience_counter}/{EARLY_STOP_PATIENCE}")
    print(f"  Batch Time: {avg_batch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
    if gpu_mem > 0:
        print(f"  GPU Memory: {gpu_mem:.2f}GB")
    print(f"  Elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"  ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}\n")

    # Salvamento de checkpoints
    save_last_checkpoint(model, optimizer, epoch + 1, CHECKPOINT_DIR)

# Finalização
best_model_path = sorted(checkpoint_callback.best_losses)[0][1]
print(f"\n{'='*60}")
print(f"Treinamento Concluído!")
print(f"Tempo Total: {time.strftime('%H:%M:%S', time.gmtime(time.time()-total_start_time))}")
print(f"Melhor Modelo: {os.path.basename(best_model_path)}")
print(f"{'='*60}")