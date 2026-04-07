import asyncio
import os
import json
import re
import textwrap
from typing import List, Optional
from openai import OpenAI

from models import PRReviewAction
from client import PrReviewerEnvClient

# --- HACKATHON REQUIRED CONFIGURATION ---
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "pr_reviewer_env:latest") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1") # Change to your LLM provider
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

TASK_NAME = "pr_review"
BENCHMARK = "pr_reviewer_env"
MAX_STEPS = 5
TEMPERATURE = 0.2

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert autonomous code reviewer. Your job is to find vulnerabilities in a code diff.
    You can inspect lines, request changes, or approve the PR.
    
    You MUST respond with a valid JSON object matching this exact schema:
    {
        "action_type": "inspect_line" | "request_changes" | "approve_pr",
        "line_number": <int or null>,
        "issue_type": "<string or null>"
    }
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Flatten the action string to remove newlines for the logger
    action_flat = action.replace("\n", "").replace("\r", "")
    print(f"[STEP] step={step} action={action_flat} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, diff: List[str], feedback: str) -> str:
    diff_text = "\n".join([f"Line {i+1}: {line}" for i, line in enumerate(diff)])
    return textwrap.dedent(
        f"""
        Step: {step}
        Code Diff:
        {diff_text}
        
        Previous Feedback: {feedback}
        
        What is your next action? Return ONLY the JSON object.
        """
    ).strip()

def extract_action_from_llm(text: str) -> PRReviewAction:
    """Safely extracts the JSON action from the LLM's text output."""
    try:
        # Find JSON block even if LLM adds markdown formatting
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return PRReviewAction(
                action_type=data.get("action_type", "inspect_line"),
                line_number=data.get("line_number"),
                issue_type=data.get("issue_type")
            )
    except Exception as e:
        print(f"[DEBUG] JSON Parse Error: {e}")
    # Fallback if LLM fails
    return PRReviewAction(action_type="inspect_line", line_number=1)

async def main() -> None:
    # Note: If testing without a real LLM endpoint, the client will fail here.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment (Can also use base_url="http://localhost:8000" if running locally without Docker yet)
    # Since the hackathon validator expects it to run, we use the local running server logic fallback for testing.
    try:
        env = await PrReviewerEnvClient.from_docker_image(IMAGE_NAME)
    except Exception as e:
        print(f"[DEBUG] Docker image not found or failed, falling back to local UV server. Error: {e}")
        env = PrReviewerEnvClient(base_url="http://localhost:8000")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, obs.code_diff, obs.feedback)
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                )
                llm_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"[DEBUG] Model Request Failed (Check API Keys): {e}")
                llm_text = '{"action_type": "inspect_line", "line_number": 1}'

            action = extract_action_from_llm(llm_text)
            
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=str(action), reward=reward, done=result.done, error=obs.error)

            if result.done:
                break

        # Calculate score (0.0 to 1.0). Our tasks give max 1.0 reward on success.
        score = max(0.0, min(1.0, sum(rewards)))
        success = score > 0.5

    finally:
        try:
            await env.close()
        except Exception as e:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())