import argparse

import torch
import wandb
from augmentations import (
    random_frequency_masking,
    random_time_masking,
    random_time_shift,
    random_noise,
)
from torch.utils.data import DataLoader

from dataset import ESC50Dataset
from model import CompactConvolutionalTransformer
from trainer import ESC50Trainer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mel_augmentations = [
        random_frequency_masking,
        random_time_masking,
    ]
    augmentations = [
        random_time_shift,
        random_noise,
    ]

    train_dataset = ESC50Dataset(
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        folds=args.train_folds,
        augmentations=augmentations,
        mel_augmentations=mel_augmentations,
        augmentation_prob=args.augmentation_prob,
        resample_rate=args.resample,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        mel_size=args.mel_size,
        mean=None,
        std=None,
    )
    test_dataset = ESC50Dataset(
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        folds=[args.test_fold],
        augmentations=None,
        mel_augmentations=None,
        augmentation_prob=0.0,
        resample_rate=args.resample,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        mel_size=args.mel_size,
        mean=train_dataset.mean, # use mean and std of training set
        std=train_dataset.std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = CompactConvolutionalTransformer(
        d_model=args.d_model,
        num_classes=args.num_classes,
        image_size=args.image_size,
        image_channels=1,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout_rate=args.dropout,
        conv_out_channels=args.conv_out_channels,
        conv_kernels=args.conv_kernels,
        conv_strides=args.conv_strides,
        pool_kernels=args.pool_kernels,
        pool_strides=args.pool_strides,
        project=args.project,
        max_pool=args.max_pool,
    )

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    wandb.init(project="CCT_ESC50_", name=args.model_name)
    wandb.watch(model, log="all")
    wandb.config.update(args)

    trainer = ESC50Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader, # using the same as ESC50 is split into CV folds anyways
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        path_name=args.model_name,
    )

    trainer.train(epochs=args.epochs)
    trainer.test()

    wandb.finish()

    if args.save_checkpoint:
        trainer.save_checkpoint(args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Transformer-based classifier on ESC-50 dataset."
    )

    parser.add_argument("--csv_path", type=str, help="Path to the ESC-50 CSV file")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--train_folds", type=int, nargs="+", default=[1,2,3,4], help="Folds to use for training")
    parser.add_argument("--test_fold", type=int, default=5, help="Fold to use for testing")
    parser.add_argument("--augmentation_prob", type=float, default=0.95, help="Probability of applying augmentation")
    parser.add_argument("--resample", type=int, default=22050 * 2, help="Audio resample rate")
    parser.add_argument("--n_mels", type=int, default=96, help="Number of mel bands")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=512, help="Number of samples between successive frames")
    parser.add_argument("--mel_size", type=int, nargs=2, default=[96, 256], help="Size of mel spectrogram")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_classes", type=int, default=50, help="Number of classes")
    parser.add_argument("--image_size", type=int, nargs=2, default=[96, 256], help="Input image size")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--transformer_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--conv_out_channels", type=int, nargs="+", default=[64, 128, 256], help="Output channels for conv layers")
    parser.add_argument("--conv_kernels", type=int, nargs="+", default=[3, 3, 3], help="Kernel sizes for conv layers")
    parser.add_argument("--conv_strides", type=int, nargs="+", default=[1, 1, 1], help="Strides for conv layers")
    parser.add_argument("--pool_kernels", type=int, nargs="+", default=[3, 3, 3], help="Kernel sizes for pooling layers")
    parser.add_argument("--pool_strides", type=int, nargs="+", default=[2, 2, 2], help="Strides for pooling layers")
    parser.add_argument("--project", action="store_true", default=True, help="Whether to use projection layer")
    parser.add_argument("--max_pool", action="store_true", default=True, help="Whether to use max pooling")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00075, help="Learning rate") # 0.002, 5e-4
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--model_name", type=str, default="esc50classifier", help="Name for saving model")
    parser.add_argument("--save_checkpoint", action="store_true", help="Whether to save model checkpoint")

    args = parser.parse_args()
    main(args)
