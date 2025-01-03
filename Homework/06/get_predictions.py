import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def predict_by_token_id(logits: torch.Tensor, tokenizer: AutoTokenizer) -> int:
    """
    Determines the predicted choice based on the logits of the model's output.

    Args:
        logits (torch.Tensor): The logits output from the model, typically of shape (1, sequence_length, vocab_size).
        tokenizer (AutoTokenizer): The tokenizer used to encode the input prompt.

    Returns:
        int: The index of the predicted choice (0 for 'A', 1 for 'B', 2 for 'C', 3 for 'D').
    """
    # For real tokenizer
    # ttoi = {
    #     tokenizer(char, add_special_tokens=False)["input_ids"][0]: i
    #     for i, char in enumerate("ABCD")
    # }

    # For mocked tokenizer from tests
    ttoi = {tokenizer.encode(char, add_special_tokens=False)[0]: i for i, char in enumerate("ABCD")}

    last_logits = logits[0, -1]
    choice_logits = torch.tensor([last_logits[token_id] for token_id in ttoi.keys()])
    return int(torch.argmax(choice_logits).item())


def get_choice_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """
    Calculates the average log probabilities of predicted tokens for a given sequence.


    Args:
        logits (torch.Tensor): A tensor of logits generated by the model, with shape (batch_size, sequence_length, vocab_size).
        input_ids (torch.Tensor): A tensor of input token IDs, with shape (batch_size, sequence_length).

    Returns:
         float: The average log probability of the predicted tokens.
    """
    log_preds = F.log_softmax(logits, dim=2)
    gathered_log_probs = torch.gather(log_preds[:, :-1], 2, input_ids[:, 1:].unsqueeze(2))
    avg_log_prob = gathered_log_probs.mean().item()

    return avg_log_prob
