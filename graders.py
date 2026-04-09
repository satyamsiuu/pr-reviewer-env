"""
Task graders for the PR Reviewer environment.

Each function evaluates a completed episode and returns a float strictly
between 0.01 and 0.99 (never 0.0 or 1.0) as required by Phase 2.

Grader function contract:
    def grade_*(episode: dict, **kwargs) -> float
        episode: dict containing at minimum:
            - "rewards": list[float]   — per-step rewards
            - "actions": list[dict]    — per-step actions taken
            - "done": bool             — whether the episode completed
        Returns: float in (0.01, 0.99)
"""

from typing import Any, Dict, List


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0.01 and 0.99."""
    return max(0.01, min(0.99, score))


def _compute_episode_score(episode: Dict[str, Any]) -> float:
    """
    Compute a normalised score from an episode's reward list.

    Rewards are summed and normalised to (0.01, 0.99).
    """
    rewards: List[float] = episode.get("rewards", [])
    if not rewards:
        return 0.05

    total = sum(rewards)
    # Normalise: expected max total for a perfect 1-step solve is 0.95
    normalised = total / max(len(rewards), 1)  # per-step mean
    return _clamp(normalised)


# -----------------------------------------------------------------------
# Public grader functions — one per declared task
# -----------------------------------------------------------------------

def grade_easy(episode: Dict[str, Any], **kwargs) -> float:
    """
    Grader for pr_review_easy (hardcoded_secret detection).

    Awards:
    - 0.95 for finding the hardcoded API key on the correct line.
    - 0.45 for identifying hardcoded_secret on the wrong line.
    - 0.05 for any other outcome.
    """
    actions: List[Dict] = episode.get("actions", [])
    for action in actions:
        if action.get("action_type") == "request_changes":
            if (
                action.get("issue_type") == "hardcoded_secret"
                and action.get("line_number") == 2
            ):
                return _clamp(0.95)
            elif action.get("issue_type") == "hardcoded_secret":
                return _clamp(0.45)
    return _clamp(0.05)


def grade_medium(episode: Dict[str, Any], **kwargs) -> float:
    """
    Grader for pr_review_medium (inefficient_loop detection).

    Awards:
    - 0.95 for finding the O(n^2) loop on the correct line (4).
    - 0.45 for identifying inefficient_loop on the wrong line.
    - 0.05 for any other outcome.
    """
    actions: List[Dict] = episode.get("actions", [])
    for action in actions:
        if action.get("action_type") == "request_changes":
            if (
                action.get("issue_type") == "inefficient_loop"
                and action.get("line_number") == 4
            ):
                return _clamp(0.95)
            elif action.get("issue_type") == "inefficient_loop":
                return _clamp(0.45)
    return _clamp(0.05)


def grade_hard(episode: Dict[str, Any], **kwargs) -> float:
    """
    Grader for pr_review_hard (sql_injection detection).

    Awards:
    - 0.95 for finding SQL injection on the correct line (2).
    - 0.45 for identifying sql_injection on the wrong line.
    - 0.05 for any other outcome.
    """
    actions: List[Dict] = episode.get("actions", [])
    for action in actions:
        if action.get("action_type") == "request_changes":
            if (
                action.get("issue_type") == "sql_injection"
                and action.get("line_number") == 2
            ):
                return _clamp(0.95)
            elif action.get("issue_type") == "sql_injection":
                return _clamp(0.45)
    return _clamp(0.05)
