import uuid
from typing import Optional
from openenv.core.env_server import Environment
from models import PRReviewAction, PRReviewObservation, PRReviewState


def clamp_score(score: float) -> float:
    """Clamp score to strictly (0.01, 0.99) — never exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, score))


# Registry of independently addressable tasks, each with its own grader config.
# The Phase 2 evaluator selects tasks by name via reset(task_name=...).
TASK_REGISTRY = {
    "pr_review_easy": {
        "level": "easy",
        "diff": [
            "def connect_to_database():",
            "    api_key = '12345'",
            "    db = Database(api_key)",
            "    return db",
        ],
        "target_line": 2,
        "target_issue": "hardcoded_secret",
        "description": "Find the hardcoded API key in the database connector.",
    },
    "pr_review_medium": {
        "level": "medium",
        "diff": [
            "def process_matrix(matrix):",
            "    n = len(matrix)",
            "    for i in range(n):",
            "        for j in range(n):",
            "            matrix[i][j] *= 2",
            "    return matrix",
        ],
        "target_line": 4,
        "target_issue": "inefficient_loop",
        "description": "Spot the O(n^2) nested loop that should be vectorised.",
    },
    "pr_review_hard": {
        "level": "hard",
        "diff": [
            "def get_user_data(user_input):",
            "    query = 'SELECT * FROM users WHERE id=' + user_input",
            "    cursor.execute(query)",
            "    return cursor.fetchall()",
        ],
        "target_line": 2,
        "target_issue": "sql_injection",
        "description": "Detect the unsanitised string concatenation enabling SQL injection.",
    },
}

DEFAULT_TASK_NAME = "pr_review_easy"


class PrReviewerEnvEnvironment(Environment):
    """PR Reviewer environment with 3 named tasks and partial-reward graders."""

    def __init__(self):
        self._episode_id = ""
        self._step_count = 0
        self._current_task_name = DEFAULT_TASK_NAME
        self._max_steps = 10

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> PRReviewObservation:
        """Reset the environment, optionally selecting a named task.

        Args:
            task_name: One of 'pr_review_easy', 'pr_review_medium',
                       'pr_review_hard'. Defaults to 'pr_review_easy'.
        """
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0

        if task_name and task_name in TASK_REGISTRY:
            self._current_task_name = task_name
        else:
            self._current_task_name = DEFAULT_TASK_NAME

        task = TASK_REGISTRY[self._current_task_name]

        return PRReviewObservation(
            task_name=self._current_task_name,
            code_diff=task["diff"],
            step_count=self._step_count,
            max_steps=self._max_steps,
            feedback=(
                f"[{task['level'].upper()}] {task['description']} "
                "Inspect lines or request changes."
            ),
            done=False,
            reward=clamp_score(0.05),  # non-zero starting reward placeholder
        )

    def step(self, action: PRReviewAction, **kwargs) -> PRReviewObservation:
        """Execute one action and return a partially-rewarded observation.

        Reward scale (all values are clamped to [0.01, 0.99]):
        - inspect_line on the target line  → 0.15  (navigation credit)
        - inspect_line on other valid line → 0.08  (exploration credit)
        - inspect_line on invalid line     → 0.05  (no credit, minimal)
        - request_changes: correct issue + correct line → 0.95
        - request_changes: correct issue + wrong line  → 0.45
        - request_changes: wrong issue                 → 0.05
        - approve_pr (wrong decision)                  → 0.02
        - unknown action                               → 0.05
        - max steps exceeded without decision          → 0.05
        """
        self._step_count += 1
        task = TASK_REGISTRY[self._current_task_name]

        raw_reward = 0.05
        done = False
        feedback = ""

        if action.action_type == "inspect_line":
            if action.line_number and 1 <= action.line_number <= len(task["diff"]):
                line_content = task["diff"][action.line_number - 1].strip()
                feedback = f"Line {action.line_number}: '{line_content}'"
                if action.line_number == task["target_line"]:
                    raw_reward = 0.15  # agent found the right line
                else:
                    raw_reward = 0.08  # exploring, not there yet
            else:
                feedback = "Invalid line number."
                raw_reward = 0.05

        elif action.action_type == "request_changes":
            correct_issue = action.issue_type == task["target_issue"]
            correct_line = action.line_number == task["target_line"]

            if correct_issue and correct_line:
                raw_reward = 0.95
                feedback = (
                    f"Correct! Found '{task['target_issue']}' "
                    f"at line {task['target_line']}."
                )
            elif correct_issue and not correct_line:
                raw_reward = 0.45
                feedback = (
                    f"Right issue type ('{task['target_issue']}') "
                    f"but wrong line ({action.line_number})."
                )
            else:
                raw_reward = 0.05
                feedback = (
                    f"Wrong issue type. Got '{action.issue_type}', "
                    f"expected '{task['target_issue']}'."
                )
            done = True

        elif action.action_type == "approve_pr":
            raw_reward = 0.02
            feedback = "Incorrect: approved a PR with a known vulnerability."
            done = True

        else:
            feedback = f"Unknown action type: '{action.action_type}'."
            raw_reward = 0.05

        # Timeout
        if not done and self._step_count >= self._max_steps:
            done = True
            raw_reward = 0.05
            feedback = "Max steps reached without a final decision."

        reward = clamp_score(raw_reward)

        return PRReviewObservation(
            task_name=self._current_task_name,
            code_diff=task["diff"],
            step_count=self._step_count,
            max_steps=self._max_steps,
            feedback=feedback,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> PRReviewState:
        task = TASK_REGISTRY[self._current_task_name]
        return PRReviewState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_task_level=task["level"],
            target_line=task["target_line"],
            target_issue=task["target_issue"],
        )