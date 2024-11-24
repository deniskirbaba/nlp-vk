from torch.utils.data import Dataset
from tqdm import tqdm


class IMDBPairwiseDataset(Dataset):
    """
    A dataset of all possible pairs of chosen and rejected texts for TRL reward training format.

    This dataset is designed to facilitate the training of a reward model by providing pairs of
    texts where one is preferred (chosen) and the other is not (rejected). Each sample in the dataset
    is a dictionary containing tokenized input IDs and attention masks for both the chosen and rejected
    texts.

    Parameters:
    imdb: dataset to pairwise
    tokenizer: The tokenizer used to preprocess the texts
    accepted_label (int): The label that indicates a chosen text. Texts with this label are considered
                          preferred, while others are considered rejected.

    Methods:
    __len__(): Returns the total number of possible pairs of chosen and rejected texts.
    __getitem__(index): Returns a dictionary containing tokenized inputs for a specific pair of chosen
                        and rejected texts.
    """

    def __init__(self, imdb, tokenizer, accepted_label: int, device="cpu"):
        super().__init__()
        self.tokenizer = tokenizer
        self.chosen_texts = []
        self.rejected_texts = []
        for review in tqdm(imdb, leave=False):
            if review["label"] == accepted_label:
                self.chosen_texts.append(review["text"])
            else:
                self.rejected_texts.append(review["text"])

        assert self.chosen_texts, f"no texts with label {accepted_label}"
        print(
            f"Found {len(self.chosen_texts)} chosen and {len(self.rejected_texts)} rejected texts, {len(self)} pairs"
        )

        self.column_names = [
            "input_ids_chosen",
            "attention_mask_chosen",
            "input_ids_rejected",
            "attention_mask_rejected",
        ]

    def __len__(self):
        return len(self.chosen_texts) * len(self.rejected_texts)  # all pairs

    def __getitem__(self, index: int):
        ch_i, rej_i = index // len(self.chosen_texts), index % len(self.chosen_texts)
        ch_tok = self.tokenizer(self.chosen_texts[ch_i], return_tensors="pt")
        rej_tok = self.tokenizer(self.rejected_texts[rej_i], return_tensors="pt")
        return dict(
            input_ids_chosen=ch_tok["input_ids"].squeeze(),
            attention_mask_chosen=ch_tok["attention_mask"].squeeze(),
            input_ids_rejected=rej_tok["input_ids"].squeeze(),
            attention_mask_rejected=rej_tok["attention_mask"].squeeze(),
        )
