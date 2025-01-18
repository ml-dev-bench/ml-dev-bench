import torch
from channel_vit import ChannelViT


def test_channel_vit_sequence_lengths_and_pos_embeddings():
    # Test parameters
    batch_size = 2
    num_channels = 6
    img_size = 224
    patch_size = 16

    # Calculate expected sequence length
    # Each channel is processed independently, so patches per channel = (H/P) * (W/P)
    patches_per_channel = (img_size // patch_size) * (img_size // patch_size)
    expected_seq_length = patches_per_channel * num_channels

    # Create model
    model = ChannelViT(
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=num_channels,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )

    # Create dummy input
    x = torch.randn(batch_size, num_channels, img_size, img_size)

    # Forward pass
    output = model(x)

    assert model.pos_embed.shape == (
        1,
        patches_per_channel + 1,
        384,
    ), (
        'Positional embedding shape mismatch, use same learnable positional embeddings across all channels'
    )

    # Verify input shape
    assert x.shape == (batch_size, num_channels, img_size, img_size), (
        f'Input shape mismatch. Expected {(batch_size, num_channels, img_size, img_size)}, got {x.shape}'
    )

    assert output.shape[1] == expected_seq_length + 1, (
        f'Sequence length mismatch. Expected {expected_seq_length}, got {output.shape[1]}'
    )

    # Additional test to verify patch embedding shape
    patches = model.patch_embed(x)
    assert patches.shape == (batch_size, expected_seq_length, model.embed_dim), (
        f'Patch embedding shape mismatch. Expected {(batch_size, expected_seq_length, model.embed_dim)}, got {patches.shape}'
    )


if __name__ == '__main__':
    test_channel_vit_sequence_lengths_and_pos_embeddings()
