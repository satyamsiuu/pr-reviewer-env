from typing import Any, Dict
from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult
from models import PRReviewAction, PRReviewObservation, PRReviewState

class PrReviewerEnvClient(HTTPEnvClient[PRReviewAction, PRReviewObservation]):
    def _step_payload(self, action: PRReviewAction) -> dict:
        """Convert Action into JSON"""
        return {
            "action_type": action.action_type,
            "line_number": action.line_number,
            "issue_type": action.issue_type
        }

    def _parse_result(self, payload: dict) -> StepResult[PRReviewObservation]:
        """Convert JSON back into Observation"""
        obs_data = payload.get("observation", {})
        obs = PRReviewObservation(
            code_diff=obs_data.get("code_diff", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 5),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=float(payload.get("reward", 0.0))
        )
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload: dict) -> PRReviewState:
        """Convert JSON back into State"""
        return PRReviewState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_task_level=payload.get("current_task_level", ""),
            target_line=payload.get("target_line", 0),
            target_issue=payload.get("target_issue", "")
        )