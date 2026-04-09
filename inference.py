import asyncio
import os
import json
import re
import textwrap
from typing import List, Optional
from openai import OpenAI
from models import PRReviewAction
from client import PrReviewerEnvClient
from dotenv import load_dotenv

load_dotenv(override=True)

# --- HACKATHON REQUIRED CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = HF_TOKEN or os.getenv("API_KEY") or "dummy_key"

# All 3 tasks must be run so the Phase 2 validator sees ≥3 tasks with graders.
TASK_NAMES = ["pr_review_easy", "pr_review_medium", "pr_review_hard"]
BENCHMARK = "pr_reviewer_env"
MAX_STEPS = 10
TEMPERATURE = 0.2

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert autonomous code reviewer navigating a large codebase.
    You only see 15 lines of code at a time.
    
    You MUST respond with a valid JSON object matching this exact schema:
    {
        "action_type": "scroll_down" | "scroll_up" | "search" | "inspect_line" | "request_changes",
        "line_number": <int or null>,
        "issue_type": "hardcoded_secret" | "inefficient_loop" | "sql_injection" | null,
        "search_query": "<string or null>"
    }
    
    Tools:
    - "search": Provide a "search_query" (e.g., "SELECT") to find suspicious keywords.
    - "scroll_down" / "scroll_up": Move your viewport through the file.
    - "request_changes": Use this ONLY when you are looking at the exact line of the vulnerability.
    """
).strip()


# ---------------------------------------------------------------------------
# STRICT STDOUT FORMATTING  – one [START]/[STEP]*/[END] block per task
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    action_flat = action.replace("\n", "").replace("\r", "")
    print(f"[STEP] step={step} action={action_flat} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Prompt / action helpers
# ---------------------------------------------------------------------------

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
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return PRReviewAction(
                action_type=data.get("action_type", "inspect_line"),
                line_number=data.get("line_number"),
                issue_type=data.get("issue_type"),
                search_query=data.get("search_query")
            )
    except Exception:
        pass
    return PRReviewAction(action_type="scroll_down")


def clamp_score(score: float) -> float:
    """Ensure the final score is strictly between 0.01 and 0.99."""
    return max(0.01, min(0.99, score))


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

async def run_task(task_name: str, env, client: OpenAI) -> None:
    """Execute one full episode for *task_name* and log [START]/[STEP]*/[END]."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
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
            except Exception:
                llm_text = '{"action_type": "inspect_line", "line_number": 1}'

            action = extract_action_from_llm(llm_text)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action.model_dump_json(),
                reward=reward,
                done=result.done,
                error=obs.error,
            )

            if result.done:
                break

        score = clamp_score(sum(rewards) if rewards else 0.05)
        success = score > 0.5

    except Exception as exc:
        # Graceful degradation – still emit a valid [END] so the grader
        # doesn't abort due to a missing block.
        score = clamp_score(0.05)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main – iterate over all 3 tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        env = await PrReviewerEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception:
        env = PrReviewerEnvClient(
            base_url=f"http://localhost:{os.getenv('ENV_PORT', '8000')}"
        )

    try:
        for task_name in TASK_NAMES:
            await run_task(task_name, env, llm_client)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())