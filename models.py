from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class PRReviewAction(BaseModel):
    model_config = ConfigDict(extra='ignore')
    action_type: str  # Now accepts: "scroll_down", "scroll_up", "search", "inspect_line", "request_changes"
    line_number: Optional[int] = None
    issue_type: Optional[str] = None
    search_query: Optional[str] = None # Added for the search action

class PRReviewObservation(BaseModel):
    model_config = ConfigDict(extra='ignore')
    task_name: Optional[str] = None  # Which named task is active
    code_diff: List[str]
    step_count: int
    max_steps: int
    feedback: str
    done: bool
    reward: float
    error: Optional[str] = None  # The logger looks for this

class PRReviewState(BaseModel):
    model_config = ConfigDict(extra='ignore')
    episode_id: str
    step_count: int
    current_task_level: str
    target_line: int
    target_issue: str