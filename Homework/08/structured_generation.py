import random

from fsm import FSM, build_odd_zeros_fsm


def get_valid_tokens(vocab: dict[int, str], eos_token_id: int, fsm: FSM, state: int) -> list[int]:
    """Filter tokens from the vocabulary based on the given state in the FSM.
    1. Retain only tokens that can be achieved from the given state.
    2. If the current state is terminal, then add the EOS token.

    Args:
        vocab (dict): vocabulary, id to token
        eos_token_id (int): index of EOS token
        fsm (FSM): Finite-State Machine
        state (int): start state
    Returns:
        valid tokens (list): list of possible tokens
    """
    valid_ids = [id for id, tok in vocab.items() if fsm.validate_continuation(state, tok)]

    if fsm.states[state].is_terminal:
        valid_ids.append(eos_token_id)
    
    return valid_ids


def random_generation() -> str:
    """Structured generation based on Odd-Zeros FSM with random sampling from possible tokens.

    Args:
    Returns:
        generation (str): A binary string with an odd number of zeros.
    """
    # Define our vocabulary
    vocab = {0: "[EOS]", 1: "0", 2: "1"}
    eos_token_id = 0
    # Init Finite-State Machine
    fsm, state = build_odd_zeros_fsm()

    # List with generate tokens
    tokens: list[int] = []
    # Sample until EOS token
    while True:
        # 1. Get valid tokens
        valid_ids = get_valid_tokens(vocab, eos_token_id, fsm, state)

        # 2. Get next token
        next_token_id = random.choice(valid_ids)

        # 3. End generation or move to next iteration
        if next_token_id == eos_token_id:
            break
        tokens.append(next_token_id)
        state = fsm.move(vocab[next_token_id], state)

    # Convert tokens to string
    return "".join([vocab[it] for it in tokens])


if __name__ == "__main__":
    print(random_generation())
