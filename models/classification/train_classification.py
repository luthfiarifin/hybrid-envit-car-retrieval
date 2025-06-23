import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.classification.hybrid_efficient_net_vit_model import HybridEfficientNetViT


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=5, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        import os

        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CarAugmentation:
    """
    An augmentation pipeline specifically tuned for
    high-angle, real-world CCTV footage with various weather and occlusions.
    """

    def __init__(self):
        self.aug = A.Compose(
            [
                # --- 1. MANDATORY: Resize Every Image First ---
                # This ensures all images entering the pipeline are the correct size.
                A.Resize(height=224, width=224),
                # --- 2. OPTIONAL: Apply Random Augmentations ---
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-10, 10), p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.75
                ),
                A.RandomRain(p=0.2),
                A.RandomShadow(p=0.3),
                A.ColorJitter(p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 25),
                    hole_width_range=(8, 25),
                    p=0.5,
                ),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.ImageCompression(
                    compression_type="jpeg",
                    quality_range=(75, 95),
                    p=0.5,
                ),
                # --- 3. MANDATORY: Final Conversion ---
                # These must always be applied at the end.
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        return self.aug(image=img)["image"]


class CarClassifierTrainer:
    """
    CarClassifierTrainer: A class to train a hybrid ResNet-ViT model for car classification.
    """

    def __init__(
        self,
        train_dir="dataset/train",
        val_dir="dataset/val",
        num_classes=8,
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=25,
        device=None,
        result_path="carvit_model.pth",
        use_weighted_loss=True,
        use_class_balancing=False,
        num_workers=0,
        early_stopping_patience=7,
        early_stopping_delta=0.001,
        early_stopping_verbose=True,
        pretrained=False,
        freeze_stem=False,
    ):
        self.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.DEVICE}")
        self.NUM_CLASSES = num_classes
        self.EMBED_DIM = embed_dim
        self.NUM_HEADS = num_heads
        self.DROPOUT = dropout
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.TRAIN_DIR = train_dir
        self.VAL_DIR = val_dir
        self.RESULT_PATH = result_path
        self.USE_WEIGHTED_LOSS = use_weighted_loss
        self.USE_CLASS_BALANCING = use_class_balancing
        self.NUM_WORKERS = num_workers
        self.PRETRAINED = pretrained
        self.FREEZE_STEM = freeze_stem

        # Early stopping parameters
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience
        self.EARLY_STOPPING_DELTA = early_stopping_delta
        self.EARLY_STOPPING_VERBOSE = early_stopping_verbose

        # Initialize tracking variables
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_losses = []
        self.epoch_times = []
        self.learning_rates = []
        self.early_stopping_triggered = False
        self.stopped_epoch = None
        self.early_stopping_counter_history = []

        # TensorBoard writer setup
        run_name = f"run_{int(time.time())}"
        self.writer = SummaryWriter(f"logs/train_classification/{run_name}")
        print(f"Logging to TensorBoard: logs/train_classification/{run_name}")

        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        # Data transformations
        self.train_transforms = CarAugmentation()
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_dataset = ImageFolder(
            root=self.TRAIN_DIR, transform=self.train_transforms
        )
        self.val_dataset = ImageFolder(root=self.VAL_DIR, transform=self.val_transforms)

        # Create samplers for class balancing if enabled
        if self.USE_CLASS_BALANCING:
            from torch.utils.data import WeightedRandomSampler

            # Calculate class weights for sampling
            class_counts = torch.zeros(self.NUM_CLASSES)
            for _, target in self.train_dataset:
                class_counts[target] += 1

            # Inverse frequency weighting
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[target] for _, target in self.train_dataset]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.BATCH_SIZE,
                sampler=sampler,
                num_workers=self.NUM_WORKERS,
            )
            print("Using WeightedRandomSampler for class balancing")
        else:
            self.train_loader = DataLoader(
                dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True
            )

        self.val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )

    def _init_model(self):
        self.model = HybridEfficientNetViT(
            num_classes=self.NUM_CLASSES,
            embed_dim=self.EMBED_DIM,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT,
            pretrained=self.PRETRAINED,
            freeze_stem=self.FREEZE_STEM,
        ).to(self.DEVICE)

        if self.USE_WEIGHTED_LOSS:
            # Calculate class weights based on inverse frequency
            class_counts = torch.zeros(self.NUM_CLASSES)
            for _, target in self.train_dataset:
                class_counts[target] += 1

            # Inverse frequency weighting
            total_samples = len(self.train_dataset)
            class_weights = total_samples / (self.NUM_CLASSES * class_counts)
            class_weights = class_weights.to(self.DEVICE)

            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)

        # Print class distribution info
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution and imbalance statistics"""
        class_counts = torch.zeros(self.NUM_CLASSES)
        for _, target in self.train_dataset:
            class_counts[target] += 1

        print(f"\n📊 Class Distribution Analysis:")
        print(f"{'Class':<12} {'Count':<8} {'Percentage':<10} {'Imbalance Ratio':<15}")
        print("-" * 50)

        max_count = class_counts.max().item()
        for i, (class_name, count) in enumerate(
            zip(self.train_dataset.classes, class_counts)
        ):
            percentage = (count / len(self.train_dataset)) * 100
            imbalance_ratio = max_count / count.item()
            print(
                f"{class_name:<12} {int(count):<8} {percentage:<10.1f}% {imbalance_ratio:<15.2f}x"
            )

        # Calculate overall imbalance metrics
        min_count = class_counts.min().item()
        imbalance_factor = max_count / min_count
        print(
            f"\n📈 Imbalance Factor: {imbalance_factor:.2f}x (Most frequent / Least frequent)"
        )

        if imbalance_factor > 5:
            print(
                "⚠️  High imbalance detected! Consider using weighted loss or resampling."
            )
        elif imbalance_factor > 2:
            print("⚠️  Moderate imbalance detected. Weighted loss recommended.")
        else:
            print("✅ Relatively balanced dataset.")

    def train_one_epoch(self):
        self.model.train()
        loop = tqdm(self.train_loader, leave=True)
        running_loss = 0.0
        num_correct = 0
        num_samples = 0

        for _, (data, targets) in enumerate(loop):
            data, targets = data.to(self.DEVICE), targets.to(self.DEVICE)
            scores = self.model(data)
            loss = self.loss_fn(scores, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum().item()
            num_samples += predictions.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(self.train_loader)
        accuracy = (num_correct / num_samples) * 100 if num_samples > 0 else 0.0
        return avg_loss, accuracy

    def check_accuracy(self):
        self.model.eval()
        num_correct = 0
        num_samples = 0
        running_val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                scores = self.model(x)
                loss = self.loss_fn(scores, y)
                running_val_loss += loss.item()

                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = (num_correct / num_samples) * 100
        avg_val_loss = running_val_loss / len(self.val_loader)

        print(f"Validation accuracy: {accuracy:.2f}%")
        print(f"Validation loss: {avg_val_loss:.4f}")

        return accuracy, avg_val_loss, all_predictions, all_targets

    def train(self):
        best_accuracy = 0.0
        best_model_state = None

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.EARLY_STOPPING_PATIENCE,
            verbose=self.EARLY_STOPPING_VERBOSE,
            delta=self.EARLY_STOPPING_DELTA,
            path=self.RESULT_PATH,
            trace_func=print,
        )

        print("Starting training with detailed tracking and early stopping...")
        print(
            f"Early Stopping - Patience: {self.EARLY_STOPPING_PATIENCE}, Delta: {self.EARLY_STOPPING_DELTA}"
        )
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        for epoch in range(self.NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.NUM_EPOCHS}")
            print(f"{'='*50}")

            # Training phase
            train_loss, train_accuracy = self.train_one_epoch()

            # Validation phase
            val_accuracy, val_loss, predictions, targets = self.check_accuracy()

            # Track metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy.item())
            self.val_losses.append(val_loss)
            self.epoch_times.append(epoch_time)
            self.learning_rates.append(current_lr)
            self.early_stopping_counter_history.append(early_stopping.counter)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            # Save best model based on accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
                print(f"🎉 New best accuracy: {best_accuracy:.2f}%")

            # Check early stopping
            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {early_stopping.val_loss_min:.6f}")
                self.early_stopping_triggered = True
                self.stopped_epoch = epoch + 1
                break

        print(f"\n{'='*50}")
        print("Training completed!")
        if self.early_stopping_triggered:
            print(
                f"Training stopped early at epoch {self.stopped_epoch} due to no improvement in validation loss"
            )
        print(f"Best validation accuracy: {best_accuracy:.2f}%")

        # Load the best model saved by early stopping (based on validation loss)
        if early_stopping.val_loss_min < np.inf:
            print(
                f"Loading best model with validation loss: {early_stopping.val_loss_min:.6f}"
            )

        # Also save the best accuracy model separately if it's different
        if best_model_state:
            best_acc_path = self.RESULT_PATH.replace(".pth", "_best_acc.pth")
            torch.save(best_model_state, best_acc_path)
            print(f"Best accuracy model saved to {best_acc_path}")

        self.writer.close()
        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "val_losses": self.val_losses,
            "epoch_times": self.epoch_times,
            "learning_rates": self.learning_rates,
            "best_accuracy": best_accuracy,
            "final_predictions": predictions,
            "final_targets": targets,
            "early_stopping_triggered": self.early_stopping_triggered,
            "stopped_epoch": self.stopped_epoch,
            "early_stopping_counter_history": self.early_stopping_counter_history,
            "best_val_loss": early_stopping.val_loss_min,
            "early_stopping_patience": self.EARLY_STOPPING_PATIENCE,
        }
