from dataclasses import dataclass, field
from typing import Optional


@dataclass
class State:
    is_terminal: bool
    transitions: dict[str, "State"] = field(default_factory=dict)

    def add_transition(self, char, state):
        self.transitions[char] = state


class FSM:
    def __init__(self, states: list[State], initial: int):
        self.states = states
        self.initial = initial

    def is_terminal(self, state_id):
        return self.states[state_id].is_terminal

    def move(self, line: str, start: Optional[int] = None) -> Optional[int]:
        """Iterate over the FSM from the given state using symbols from the line.
        If no possible transition is found during iteration, return None.
        If no given state start from initial.

        Args:
            line (str): line to iterate via FSM
            start (optional int): if passed, using as start start
        Returns:
            end (optional int): end state if possible, None otherwise
        """
        # Check start state
        if start is None:
            start = self.initial
        if len(self.states) <= start:
            return None

        # Iterate over FSM
        cur_state_id = start
        for sym in line:
            if sym not in self.states[cur_state_id].transitions:
                return None

            # Will be work only with unique states, otherwise need to change State class and tests
            next_state = self.states[cur_state_id].transitions[sym]
            cur_state_id = self.states.index(next_state)

        return cur_state_id

    def accept(self, candidate: str) -> bool:
        """Check if the candidate is accepted by the FSM.

        Args:
            candidate (str): line to check
        Returns:
            is_accept (bool): result of checking
        """
        end_state_id = self.move(line=candidate)

        return (end_state_id is not None) and self.is_terminal(end_state_id)

    def validate_continuation(self, state_id: int, continuation: str) -> bool:
        """Check if the continuation can be achieved from the given state.

        Args:
            state_id (int): state to iterate from
            continuation (str): continuation to check
        Returns:
            is_possible (bool): result of checking
        """
        return self.move(line=continuation, start=state_id) is not None


def build_odd_zeros_fsm() -> tuple[FSM, int]:
    """FSM that accepts binary numbers with odd number of zeros

    For example,
    - correct words: 0, 01, 10, 101010
    - incorrect words: 1, 1010

    Args:
    Returns:
        fsm (FSM): FSM
        start_state (int): index of initial state
    """
    start_state = State(is_terminal=False)
    false_0 = State(is_terminal=False)
    false_1 = State(is_terminal=False)
    true_0 = State(is_terminal=True)
    true_1 = State(is_terminal=True)

    start_state.add_transition("1", false_1)
    start_state.add_transition("0", true_0)

    false_1.add_transition("1", false_1)
    false_1.add_transition("0", true_0)

    true_0.add_transition("1", true_1)
    true_0.add_transition("0", false_0)

    false_0.add_transition("1", false_1)
    false_0.add_transition("0", true_0)

    true_1.add_transition("1", true_1)
    true_1.add_transition("0", false_0)

    return FSM([start_state, false_0, false_1, true_0, true_1], initial=0), 0


if __name__ == "__main__":
    _fsm, _ = build_odd_zeros_fsm()
    print("101010 -- ", _fsm.accept("101010"))
    print("10101 -- ", _fsm.accept("10101"))
