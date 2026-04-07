from dataclasses import dataclass
from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

@dataclass
class PRReviewAction(Action):
    """The AI agent's response."""
    action_type: str            # Must be 'inspect_line', 'request_changes', or 'approve_pr'
    line_number: Optional[int] = None
    issue_type: Optional[str] = None

@dataclass
class PRReviewObservation(Observation):
    """What the AI agent sees at each step."""
    code_diff: List[str]
    step_count: int
    max_steps: int
    feedback: str               # The environment's response to the last action
    done: bool
    reward: float

@dataclass
class PRReviewState(State):
    """Internal state for debugging and validation."""
    episode_id: str
    step_count: int
    current_task_level: str
    target_line: int            # The hidden correct line
    target_issue: str           # The hidden correct issue