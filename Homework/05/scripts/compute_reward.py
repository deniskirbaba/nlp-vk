from torch import Tensor, no_grad, tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MovieReview(Dataset):
    """
    Represents dataset of movie review raw texts without labels
    """

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def compute_reward(reward_model, reward_tokenizer, texts: list[str], device="cpu") -> Tensor:
    """
    Compute the reward scores for a list of texts using a specified reward model and tokenizer.

    Parameters:
    reward_model: The model used to compute the reward scores
    reward_tokenizer: The tokenizer for reward_model
    texts (list[str]): A list of text strings for which the reward scores are to be computed.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    torch.Tensor: A tensor containing the reward scores for each input text. The scores are extracted
                  from the logits of the reward model.

    Example:
    >>> compute_reward(my_reward_model, my_reward_tokenizer, ["text1", "text2"])
    tensor([ 5.1836, -4.8438], device='cpu')
    """
    dataset = MovieReview(texts)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    rewards = []
    with no_grad():
        for batch in tqdm(dataloader, desc="Computing rewards", leave=False):
            inputs = reward_tokenizer(
                batch, truncation=True, padding="max_length", return_tensors="pt"
            ).to(device)
            logits = reward_model(**inputs).logits[:, 0].detach().cpu().tolist()
            rewards.extend(logits)
    return tensor(rewards)
