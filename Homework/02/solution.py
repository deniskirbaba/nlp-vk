import torch
import torch.nn.functional as F


def compute_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """

    # return (
    #     F.softmax(queries @ keys.transpose(1, 2) / queries.shape[2] ** 0.5, dim=2)
    #     @ values
    # )
    return F.scaled_dot_product_attention(queries, keys, values)


def compute_multihead_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    projection_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """

    BATCH_SIZE, _, SEQ_LENGTH, _ = queries.shape

    att = F.scaled_dot_product_attention(queries, keys, values)
    return att.transpose(1, 2).view(BATCH_SIZE, SEQ_LENGTH, -1) @ projection_matrix.T

    # atthi = (
    #     F.softmax(
    #         queries @ keys.transpose(2, 3) / DIM_PER_HEAD ** 0.5,
    #         dim=3,
    #         dtype=torch.float32,
    #     )
    #     @ values
    # )
    # return (
    #     atthi.transpose(1, 2).reshape(
    #         queries.shape[0], queries.shape[2], -1)
    #     @ projection_matrix.T
    # )


def compute_rotary_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    _, SEQ_LENGTH, _, DIM_PER_HEAD = x.shape

    i = torch.arange(1, DIM_PER_HEAD // 2 + 1).repeat_interleave(2)
    theta = 10000**(-2*(i-1)/DIM_PER_HEAD)
    m = torch.arange(SEQ_LENGTH)

    cosmat = torch.outer(m, theta).cos()
    sinmat = torch.outer(m, theta).sin()

    x = x.transpose(1, 2)

    # permi = torch.tensor([[i+1,i] for i in range(0, DIM_PER_HEAD, 2)]).ravel()
    # rotx = x[:, :, :, permi]
    # rotx[:, :, :, ::2] *= -1

    rotx = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x)
    
    x = (x * cosmat) + (rotx * sinmat)

    return x.transpose(1, 2)