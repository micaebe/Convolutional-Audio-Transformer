import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


class ESC50Trainer:
    """
    Trainer for the ESC-50 dataset using PyTorch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        path_name: str = "esc50classifier",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCEWithLogitsLoss()
        self.path_name = path_name

    def train(self, epochs: int = 10):
        print(f"Training for {epochs} epochs")
        total_steps = len(self.train_loader) * epochs
        warmup_steps = len(self.train_loader) * 2
        print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")

        def lr_lambda(current_step):
            """cosine annealing with warmup"""
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        for epoch in range(epochs):
            self.model.train()
            correct = 0
            total = 0
            total_loss = 0.0
            for _, (inputs, targets) in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm**0.5

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

                self.optimizer.step()
                scheduler.step()

                _, predicted = outputs.max(1)
                _, targets = targets.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                accuracy = 100.0 * correct / total
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_accuracy": accuracy,
                        "grad_norm": grad_norm,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )
                total_loss += loss.item()

            val_loss, val_accuracy = self.evaluate(self.val_loader)
            print(
                f"Epoch: {epoch+1}/{epochs}   Loss: {total_loss / len(self.train_loader):.4f} Accuracy: {100.0 * correct / total:.2f}%"
            )
            print(
                f"Validation Loss: {val_loss:.4f} Validation Accuracy: {val_accuracy:.2f}%"
            )
            wandb.log({"val_accuracy": val_accuracy})

    def evaluate(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                _, targets = targets.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                wandb.log({"val_loss": loss.item()})
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def test(self, test_loader: DataLoader):
        avg_loss, accuracy = self.evaluate(test_loader)
        print(f"Test Accuracy: {accuracy:.2f}%")
        wandb.log({"test_accuracy": accuracy, "test_loss": avg_loss})

    def save_checkpoint(self, epoch: int):
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/{self.path_name}_epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        print(f"Loaded checkpoint: {checkpoint_path}")
