import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CNNEncoder(nn.Module):
    """
    CNN Encoder for Compact Convolutional Transformer
    """
    def __init__(
        self,
        channels: list[int] = [32, 64, 128],
        conv_kernels: list[int] = [3, 3, 3],
        conv_strides: list[int] = [1, 1, 1],
        max_pool: bool = True,
        pool_kernels: list[int] = [3, 3, 3],
        pool_strides: list[int] = [2, 2, 2],
        project: bool = False,
        d_model: int = 128,
    ):
        """
        Args:
            channels: list of integers, the number of output channels for each convolutional layer
            conv_kernels: list of integers, the kernel size for each convolutional layer
            conv_strides: list of integers, the stride for each convolutional layer
            max_pool: bool, whether to use max pooling after each convolutional layer
            pool_kernels: list of integers, the kernel size for each max pooling layer
            pool_strides: list of integers, the stride for each max pooling layer
            project: bool, whether to project the output to the dimension of the model
                    (if False, last conv_out_channel must be equal to d_model)
            d_model: int, the dimension of the model
        """
        super(CNNEncoder, self).__init__()
        layers = []
        in_channels = 1
        for i in range(len(channels)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=channels[i],
                        kernel_size=conv_kernels[i],
                        stride=conv_strides[i],
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                    #nn.BatchNorm2d(channels[i]),
                )
            )
            if max_pool:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_kernels[i], stride=pool_strides[i], padding=1
                    )
                )
            in_channels = channels[i]  # Update in_channels for next layer
        self.project_layer = None
        if project:
            self.project_layer = nn.Linear(channels[-1], d_model)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        out = self.model(x)  # (batch_size, channels, width_conv, height_conv)
        out = rearrange(
            out, "b c w h -> b (w h) c"
        )  # (batch_size, sequence_length, channels)
        if self.project_layer:
            out = self.project_layer(out)
        return out


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with pre-layer normalization
    """
    def __init__(
        self, d_model: int, n_heads: int, dim_mlp: int, dropout: float = 0.1
    ):
        """
        Args:
            d_model: int, the dimension of the model
            n_heads: int, the number of attention heads
            dim_mlp: int, the dimension of the MLP
            dropout: float, the dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, d_model)
        x_norm = self.norm1(x)
        x_norm = rearrange(x_norm, "b s d -> s b d")
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        attn_output = rearrange(attn_output, "s b d -> b s d")
        x = x + attn_output
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x
    

class PositionEmbedding(nn.Module):
    def __init__(self, sequence_length: int, d_model: int):
        """
        Args:
            sequence_length: int, the length of the sequence
            d_model: int, the dimension of the model
        """
        super(PositionEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(
            torch.randn(1, sequence_length, d_model)
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, d_model)
        return x + self.position_embeddings


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        transformer_layers: int = 4,
        num_heads: int = 2,
        projection_dim: int = 128,
        n_classes: int = 50,
        dropout: float = 0.1,
        sequence_length: int = 64,
    ):
        """
        Args:
            transformer_layers: int, the number of transformer layers
            num_heads: int, the number of attention heads
            projection_dim: int, the dimension of the projection
            n_classes: int, the number of classes
            dropout: float, the dropout rate
            sequence_length: int, the length of the sequence
        """
        super(TransformerClassifier, self).__init__()
        self.attention_pool = nn.Linear(projection_dim, 1)
        self.positional_embedding = PositionEmbedding(sequence_length, projection_dim)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoder(
                    d_model=projection_dim,
                    n_heads=num_heads,
                    dim_mlp=projection_dim,
                    dropout=dropout,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.head = nn.Linear(projection_dim, n_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, d_model)
        x = self.positional_embedding(x)
        for encoder in self.encoder_layers:
            x = encoder(x)
        # Attention pooling
        attn_weights = F.softmax(
            self.attention_pool(x), dim=1
        )  # (batch_size, sequence_length, 1)
        x = torch.sum(attn_weights * x, dim=1)  # (batch_size, d_model)
        x = self.head(x)
        return x


class CompactConvolutionalTransformer(nn.Module):
    """
    Compact Convolutional Transformer
    """
    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 50,
        image_size: tuple[int, int] = (64, 64),
        image_channels: int = 1,
        num_heads: int = 2,
        transformer_layers: int = 4,
        dropout_rate: float = 0.1,
        conv_out_channels: list[int] = [32, 64, 128],
        conv_kernels: list[int] = [3, 3, 3],
        conv_strides: list[int] = [1, 1, 1],
        pool_kernels: list[int] = [3, 3, 3],
        pool_strides: list[int] = [2, 2, 2],
        project: bool = False,
        max_pool: bool = True,
    ):
        """
        Args:
            num_classes: int, the number of classes
            image_size: tuple of ints, the size of the image
            image_channels: int, the number of channels in the image
            num_heads: int, the number of attention heads
            transformer_layers: int, the number of transformer layers
            dropout_rate: float, the dropout rate
            conv_out_channels: list of ints, the number of output channels for each convolutional layer
            conv_kernels: list of ints, the kernel size for each convolutional layer
            conv_strides: list of ints, the stride for each convolutional layer
            pool_kernels: list of ints, the kernel size for each max pooling layer
            pool_strides: list of ints, the stride for each max pooling layer
        """
        if not project:
            assert (
                conv_out_channels[-1] == d_model
            ), "Last conv_out_channel must be equal to d_model when project is False"
        assert (
            len(conv_out_channels)
            == len(conv_kernels)
            == len(conv_strides)
            == len(pool_kernels)
            == len(pool_strides)
        ), "All layer configs must be of the same length"
        super(CompactConvolutionalTransformer, self).__init__()
        self.tokenizer = CNNEncoder(
            channels=conv_out_channels,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            pool_kernels=pool_kernels,
            pool_strides=pool_strides,
            project=project,
            d_model=d_model,
            max_pool=max_pool,
        )
        # Calculate the sequence length after tokenizer
        dummy_input = torch.zeros(1, image_channels, image_size[0], image_size[1])
        with torch.no_grad():
            sequence_length = self.tokenizer(dummy_input).shape[1]
        print(f"Sequence length: {sequence_length}")
        print(f"Dimension of model: {d_model}")
        self.transformer = TransformerClassifier(
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            projection_dim=d_model,
            n_classes=num_classes,
            dropout=dropout_rate,
            sequence_length=sequence_length,
        )

    def forward(self, x):
        # x: (batch_size, image_channels, image_height, image_width)
        x = self.tokenizer(x)  # (batch_size, sequence_length, d_model)
        x = self.transformer(x)  # (batch_size, num_classes)
        return x
