import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
import time
import os
import signal

# File to store process ID
pid_file = "/tmp/one_new_trainer.pid"

def create_pid_file():
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

def delete_pid_file():
    if os.path.exists(pid_file):
        os.remove(pid_file)

def signal_handler(signum, frame):
    print("Training stopped by signal.")
    delete_pid_file()
    exit(0)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated Dataset Class
class StockDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = []
        total_length = 0

        for file in sorted(os.listdir(data_dir)):
            if file.endswith('.npz'):
                data_npz = np.load(os.path.join(data_dir, file))
                chunk_values = data_npz['data']
                length = len(chunk_values)
                self.data_files.append({
                    'file_path': os.path.join(data_dir, file),
                    'start_idx': total_length,
                    'end_idx': total_length + length,
                    'length': length
                })
                total_length += length

        self.total_length = total_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for file_info in self.data_files:
            if file_info['start_idx'] <= idx < file_info['end_idx']:
                relative_idx = idx - file_info['start_idx']
                data_npz = np.load(file_info['file_path'])
                chunk_values = data_npz['data']
                features = chunk_values[relative_idx, :-1].astype(np.float32)
                target = chunk_values[relative_idx, -1].astype(np.float32)
                return torch.tensor(features), torch.tensor(target)
        raise IndexError("Index out of range")

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Residual Block for Improved Gradient Flow
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return x + self.linear2(self.relu(self.linear1(x)))

# Improved Feedforward Regressor Model
class ImprovedFeedforwardRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super(ImprovedFeedforwardRegressor, self).__init__()
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.batch_norm(self.initial_layer(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = self.dropout(x)
        return self.tanh(self.output_layer(x))

# Weight Initialization Function
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

# Training Function with Latest Checkpoint Saving and Gradient Clipping
def train(
    model, dataloader, criterion_mae, criterion_focal, optimizer, scheduler, device,
    num_epochs=10, loss_ce_weight=1.0, start_epoch=0, start_batch=0, checkpoint_dir='checkpoints/',
    checkpoint_interval_batches=10
):
    model.train()

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_batches = len(dataloader)
    batch_num = start_batch

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        num_batches = 0

        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            initial=(batch_num % total_batches) if epoch == start_epoch else 0,
            total=total_batches
        )
        for features, target in progress_bar:
            batch_num += 1
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features).squeeze()

            # Apply tanh to target to match the output activation
            target_tanh = torch.tanh(target)

            # MAE Loss between model output and tanh-transformed target
            loss_mae = criterion_mae(output, target_tanh)

            # Focal Loss on the direction of the return
            prob_positive = (output + 1) / 2  # Scale from (-1,1) to (0,1)
            target_direction = (target > 0).float()
            loss_focal = criterion_focal(prob_positive, target_direction)

            # Total Loss: weighted sum of MAE and Focal losses
            loss = loss_mae + loss_ce_weight * loss_focal

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            num_batches += 1

            # Update progress bar with batch loss
            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

            # Save checkpoint every specified number of batches
            if batch_num % checkpoint_interval_batches == 0:
                checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'batch_num': batch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logger.info(f"Latest checkpoint saved at {checkpoint_path}")

            # Free up memory
            del features, target, output, loss_mae, loss_focal, loss, target_tanh, prob_positive, target_direction
            torch.cuda.empty_cache()  # If using CUDA

        scheduler.step()

        # Calculate and display average loss for the epoch
        epoch_loss = running_loss / num_batches
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.6f}')

    logger.info("Training complete.")

if __name__ == "__main__":
    # Check if PID file exists (to prevent multiple instances)
    if os.path.exists(pid_file):
        print("Training script is already running.")
        exit(0)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)

    # Create a PID file
    create_pid_file()

    # Load Data
    updated_processed_data_dir = '/home/a/Fin Project/Financial Web Scraping/updated_processed_chunks'
    dataset = StockDataset(updated_processed_data_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Reduced batch size

    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Model, Loss, and Optimizer with Best Practices
    input_dim = dataset[0][0].shape[0]
    hidden_dim = 64  # Reduced hidden dimension
    num_layers = 4   # Reduced number of layers
    dropout_rate = 0.1  # Increased dropout for regularization
    model = ImprovedFeedforwardRegressor(input_dim, hidden_dim, num_layers, dropout_rate).to(device)
    model.apply(initialize_weights)

    # Use MAE Loss and Focal Loss
    criterion_mae = nn.L1Loss()
    criterion_focal = FocalLoss()

    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Check for existing latest checkpoint
    checkpoint_dir = '/home/a/Fin Project/Financial Web Scraping/checkpoints/'
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    start_epoch = 0
    start_batch = 0

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_num']
        logger.info(f"Loaded latest checkpoint from {checkpoint_path}, starting from epoch {start_epoch+1}, batch {start_batch+1}")

    # Train the Model
    try:
        print("Training started...")
        train(
            model, dataloader, criterion_mae, criterion_focal, optimizer, scheduler, device,
            num_epochs=20, loss_ce_weight=0.7, start_epoch=start_epoch, start_batch=start_batch,
            checkpoint_dir=checkpoint_dir, checkpoint_interval_batches=100
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    finally:
        # Clean up PID file
        # Save the final Trained Model
        model_save_path = '/home/a/Fin Project/Financial Web Scraping/checkpoints/trained_transformer_model.pth'
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        delete_pid_file()

