import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import DriftDataset, DriftValidationDataset
import os 
import matplotlib.pyplot as plt
from model import *
from scipy.stats import binned_statistic 

BATCH_STRATEGIES = 128
EPOCHS = 50
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DRIFT_TRAIN = "dataset/DriftReduced/train"
DRIFT_VAL = "dataset/DriftReduced/validation"

IO_TRAIN = "dataset/InputOutputAugmented/train"
IO_VAL = "dataset/InputOutputAugmented/validation"

SAVE_DIR = "output"

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, device="cuda", batch_size=1, lr=1e-3, epochs=50, 
                 reduce_patience=5, reduce_rate=0.25, save_dir="checkpoints"):
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                     patience=reduce_patience,
                                                                     factor=reduce_rate, min_lr=1e-6)
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_loader = DataLoader(train_dataset, num_workers=16, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_dataset, num_workers=16, batch_size=batch_size, shuffle=False)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_path = os.path.join(save_dir, "best_model.pt")
        self.final_model_path = os.path.join(save_dir, "final_model.pt")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        skipped_batches = 0
        pbar = tqdm(self.train_loader, desc="Train", unit="batch")
        for drift, batch_inputs, batch_outputs in pbar:
            drift, batch_inputs, batch_outputs = drift.to(self.device), batch_inputs.to(self.device), batch_outputs.to(self.device)
            
            if torch.isnan(drift).any() or torch.isnan(batch_inputs).any() or torch.isnan(batch_outputs).any():
                print("NaN detected in train batch, skipping...")
                skipped_batches+=1
                continue

            if drift.shape[2] != 577:
                print(f"Unexpected trajectory length: {drift.shape[2]}, skipping...")
                skipped_batches += 1
                continue

            self.optimizer.zero_grad()
            preds = self.model(drift, batch_inputs)
            loss = self.criterion(preds, batch_outputs)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                "loss": running_loss / (pbar.n + 1),
                "pred_std": preds.std().item(),
                "POS_std": batch_outputs.std().item(),
                "lr": self.optimizer.param_groups[0]['lr']
            })
        epoch_loss = running_loss / (len(self.train_loader)-skipped_batches)
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        skipped_batches = 0
        pbar = tqdm(self.val_loader, desc="Validation", unit="batch")

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for drift, batch_inputs, batch_outputs in pbar:
                drift, batch_inputs, batch_outputs = drift.to(self.device), batch_inputs.to(self.device), batch_outputs.to(self.device)
                
                if torch.isnan(drift).any() or torch.isnan(batch_inputs).any() or torch.isnan(batch_outputs).any():
                    print("NaN detected in train batch, skipping...")
                    skipped_batches+=1
                    continue

                if drift.shape[2] != 577:
                    print(f"Unexpected trajectory length: {drift.shape[2]}, skipping...")
                    skipped_batches += 1
                    continue

                preds = self.model(drift, batch_inputs)
                loss = self.criterion(preds, batch_outputs)
                val_loss += loss.item()
                pbar.set_postfix({"val_loss": val_loss / (pbar.n + 1)})

                all_preds.append(preds.detach().cpu())
                all_targets.append(batch_outputs.detach().cpu())

        epoch_loss = val_loss / (len(self.val_loader) - skipped_batches)
        self.val_losses.append(epoch_loss)
        return epoch_loss, all_preds, all_targets

    def fit(self, early_stopping_patience=None):
        no_improve_epoch = 0
        last_val_preds = []
        last_val_targets = []
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.epochs} ===")
            train_loss = self.train_one_epoch()
            val_loss, last_val_preds, last_val_targets = self.validate()
            print(f"Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1

            if early_stopping_patience is not None and no_improve_epoch >= early_stopping_patience:
                print(f"Early stopping triggered after {no_improve_epoch} epochs without improvement.")
                break
        torch.save(self.model.state_dict(), self.final_model_path)
        self.plot_correlation(last_val_preds, last_val_targets)
        self.plot_losses()

    def plot_correlation(self, preds_list, targets_list):
        preds = torch.cat(preds_list).view(-1).numpy()
        targets = torch.cat(targets_list).view(-1).numpy()

        plt.figure(figsize=(8, 8))
        
        plt.scatter(targets, preds, alpha=0.05, s=1, color='#1f77b4', label='Prédictions', rasterized=True)
        
        min_val = float(min(targets.min(), preds.min()))
        max_val = float(max(targets.max(), preds.max()))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Idéal (y=x)')
        
        num_bins = 50
        bin_means, bin_edges, _ = binned_statistic(targets, preds, statistic='mean', bins=num_bins)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        plt.plot(bin_centers, bin_means, color='orange', linestyle='-', linewidth=2.5, label='Moyenne des prédictions')
        
        plt.xlabel("Vraies valeurs (Ground Truth)")
        plt.ylabel("Prédictions")
        plt.title("Corrélation avec évolution de la moyenne")
        
        leg = plt.legend(loc="upper left")
        for handle in leg.legend_handles:
            if hasattr(handle, 'set_alpha'):
                handle.set_alpha(1.0)
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([20.0])
                
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "correlation_scatter_plot.png"), dpi=300)
        plt.show()

    def plot_losses(self):
        plt.figure(figsize=(8,6))
        plt.plot(range(1, len(self.train_losses)+1), self.train_losses, label="Train Loss")
        plt.plot(range(1, len(self.val_losses)+1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss_plot.png"))
        plt.show()

def main():
    train_dataset = DriftDataset(DRIFT_TRAIN, IO_TRAIN, batch_strategies=BATCH_STRATEGIES)
    val_dataset   = DriftValidationDataset(DRIFT_VAL, IO_VAL, batch_strategies=BATCH_STRATEGIES)

    model = SurrogateModel(d_model=128, num_head=8)

    trainer = Trainer(model, train_dataset, val_dataset,device=DEVICE, batch_size=16, lr=LR, epochs=EPOCHS, 
                      reduce_patience=5, reduce_rate=0.1, save_dir=SAVE_DIR)

    trainer.fit(early_stopping_patience=10)


if __name__ == "__main__":
    main()