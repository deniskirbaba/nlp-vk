import numpy as np
from tqdm import tqdm


def eval_reward_model(reward_model, reward_tokenizer, test_dataset, target_label, device="cpu"):
    """
    Evaluate the performance of a reward model by comparing reward scores for chosen and rejected reviews.

    This function selects reviews from a test dataset based on a target label and evaluates the reward model's
    ability to assign higher scores to chosen reviews compared to rejected ones. The evaluation is performed
    in batches for efficiency.
    Note that reward scores are compared on corresponding chosen and rejected reviews:
        chosen_reviews[0] vs rejected_reviews[0],
        chosen_reviews[1] vs rejected_reviews[1],
        etc.

    Parameters:
    reward_model: The model used to compute the reward scores
    reward_tokenizer: The tokenizer for reward_model
    test_dataset: test Dataset
    target_label (0 or 1): The label used to select chosen reviews. Reviews with this label are considered chosen,
                  while others are considered rejected.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    float: The accuracy of the reward model, calculated as the proportion of times the model assigns a higher
           reward score to the chosen review compared to the rejected review.

    Example:
    >>> accuracy = eval_reward_model(my_reward_model, my_reward_tokenizer, test_data, target_label=1)
    >>> print(f"Model accuracy: {accuracy:.2%}")
    """
    # Compute rewards of all test dataset
    rewards = compute_reward(
        reward_model, reward_tokenizer, [review["text"] for review in test_dataset]
    )

    # Define chosen and rejected indices
    labels = np.array([review["label"] for review in test_dataset])
    ch_ids = np.argwhere(labels == target_label)
    rej_ids = np.argwhere(labels != target_label)
    assert len(ch_ids) == len(rej_ids)

    # Calculate accuracy
    total = len(ch_ids) * len(rej_ids)
    correct = 0
    for ch_idx in tqdm(ch_ids, desc="Calculating accuracy", leave=False):
        correct += (rewards[rej_ids] < rewards[ch_ids]).sum().item()

    return correct / total
