import torch
import torch.nn.functional as F


def scaled_dot_product_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = True,
    need_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """

    _, seq_len, num_heads, hid_dim = query.shape
    _, kv_seq_len, kv_heads, _ = key.shape

    if num_heads % kv_heads != 0:
        raise ValueError(
            "'num_heads' must be divisible by 'kv_heads' for grouped attention."
        )

    head_ratio = num_heads // kv_heads
    scale = hid_dim**-0.5

    if kv_heads < num_heads:
        key = key.repeat_interleave(head_ratio, dim=2)
        value = value.repeat_interleave(head_ratio, dim=2)

    # using einsum because it's more concise, but was harder to understand how to use it :) 
    attn_logits = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale

    if is_causal:
        causal_mask = torch.tril(
            torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=query.device)
        )
        attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_logits, dim=-1)

    # another einsum...
    attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)

    if need_weights:
        return attn_output, attn_weights
    return attn_output, None
