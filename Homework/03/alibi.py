import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """
    slopes = torch.tensor(
        [2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)]
    ).view(num_heads, 1, 1)

    relative_positions = torch.arange(seq_len).view(1, -1) - torch.arange(seq_len).view(
        -1, 1
    )
    
    biases = (slopes * relative_positions).float()

    return biases


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
