import uuid
from typing import List
from openenv.core.env_server import Environment
from models import PRReviewAction, PRReviewObservation, PRReviewState

class PrReviewerEnvEnvironment(Environment):
    def __init__(self):
        self.tasks = [
            {
                "level": "easy",
                "diff": [
                    "def connect_to_database():",
                    "    api_key = '12345'", 
                    "    db = Database(api_key)",
                    "    return db"
                ],
                "target_line": 2,
                "target_issue": "hardcoded_secret"
            },
            {
                "level": "medium",
                "diff": [
                    "def process_matrix(matrix):",
                    "    n = len(matrix)",
                    "    for i in range(n):",
                    "        for j in range(n):", 
                    "            matrix[i][j] *= 2",
                    "    return matrix"
                ],
                "target_line": 4,
                "target_issue": "inefficient_loop"
            },
            {
                "level": "hard",
                "diff": [
                    "def get_user_data(user_input):",
                    "    query = 'SELECT * FROM users WHERE id=' + user_input", 
                    "    cursor.execute(query)",
                    "    return cursor.fetchall()"
                ],
                "target_line": 2,
                "target_issue": "sql_injection"
            }
        ]
        
        self._episode_id = ""
        self._step_count = 0
        self._current_task_idx = 0
        self._max_steps = 5  

    def reset(self) -> PRReviewObservation:
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._current_task_idx = 0
        
        task = self.tasks[self._current_task_idx]
        
        return PRReviewObservation(
            code_diff=task["diff"],
            step_count=self._step_count,
            max_steps=self._max_steps,
            feedback="New PR loaded. Please inspect or review.",
            done=False,
            reward=0.0
        )

    def step(self, action: PRReviewAction) -> PRReviewObservation:
        self._step_count += 1
        task = self.tasks[self._current_task_idx]
        
        reward = 0.0
        done = False
        feedback = ""

        if action.action_type == "inspect_line":
            if action.line_number and 1 <= action.line_number <= len(task["diff"]):
                line_content = task["diff"][action.line_number - 1].strip()
                feedback = f"Line {action.line_number} reads: '{line_content}'"
            else:
                feedback = "Invalid line number requested."
                reward = -0.1 

        elif action.action_type == "request_changes":
            if action.issue_type == task["target_issue"] and action.line_number == task["target_line"]:
                reward = 1.0  
                feedback = f"Correct! Found {task['target_issue']} at line {task['target_line']}."
                done = True
            elif action.issue_type == task["target_issue"] and action.line_number != task["target_line"]:
                reward = 0.5  
                feedback = f"Partial success. The issue is {task['target_issue']}, but not on line {action.line_number}."
                done = True
            else:
                reward = -0.5 
                feedback = f"Incorrect. The issue is not {action.issue_type} on line {action.line_number}."
                done = True

        elif action.action_type == "approve_pr":
            reward = -1.0 
            feedback = "Catastrophic failure. You approved a PR with a severe vulnerability."
            done = True
            
        else:
            feedback = "Invalid action type."
            reward = -0.1

        if not done and self._step_count >= self._max_steps:
            done = True
            feedback = "Max steps reached without a final decision."
            reward = 0.0

        if done:
            self._current_task_idx += 1
            if self._current_task_idx >= len(self.tasks):
                feedback += " All tasks completed."

        current_task_diff = [] if self._current_task_idx >= len(self.tasks) else self.tasks[min(self._current_task_idx, len(self.tasks)-1)]["diff"]
        
        return PRReviewObservation(
            code_diff=current_task_diff,
            step_count=self._step_count,
            max_steps=self._max_steps,
            feedback=feedback,
            done=done,
            reward=reward
        )

    @property
    def state(self) -> PRReviewState:
        task = self.tasks[min(self._current_task_idx, len(self.tasks)-1)]
        return PRReviewState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_task_level=task["level"],
            target_line=task["target_line"],
            target_issue=task["target_issue"]
        )