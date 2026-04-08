import os
import json
import urllib.request
from typing import List, Optional
from openai import OpenAI

# Configuration — set these as environment variables before running
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
BENCHMARK = "cyber-ops-v1"

# OpenAI-compatible client pointing to HuggingFace router
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# ── HTTP helpers using urllib (no requests dependency) ────────────────────────

def http_post(url: str, data: dict = None) -> dict:
    body = json.dumps(data).encode("utf-8") if data else b""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Logging helpers (must match official format exactly) ──────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ──────────────────────────────────────────────────────────────────

def ask_llm(task_name: str, obs_text: str) -> str:
    """Call the LLM and return a single bash command."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Cybersecurity Expert operating a Linux terminal. "
                    "Your job is to complete the given security task step by step.\n\n"
                    "Rules:\n"
                    "- Output ONLY a single raw bash command. No explanation, no markdown, no backticks.\n"
                    "- To investigate: use 'ps aux', 'ls -la /etc', 'cat /var/log/auth.log', 'grep', etc.\n"
                    "- To fix log-analysis: once you see an attacker IP, output it as: echo '192.168.x.x'\n"
                    "- To fix process-hunt: once you see the PID, run: kill -9 <PID>\n"
                    "- To fix perm-fix: once you see a bad file, run: chmod 644 <filepath>\n"
                    "- Never repeat a command that already showed you results. Move to the fix step."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task_name}\n"
                    f"Last terminal output: {obs_text}\n\n"
                    "What is your next bash command?"
                )
            }
        ],
        temperature=0.0,
        max_tokens=64
    )
    cmd = response.choices[0].message.content.strip()
    cmd = cmd.replace("`", "").replace("```bash", "").replace("```", "").strip()
    return cmd


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation():
    tasks = ["log-analysis", "process-hunt", "perm-fix"]
    all_results = {}

    for t_name in tasks:
        log_start(task=t_name, env=BENCHMARK, model=MODEL_NAME)

        resp = http_post(f"{ENV_BASE_URL}/reset?task={t_name}")
        obs_text = resp["terminal_output"]

        done = False
        step_count = 0
        rewards: List[float] = []
        error = None

        try:
            while not done and step_count < 5:
                step_count += 1

                action_cmd = ask_llm(t_name, obs_text)

                data = http_post(f"{ENV_BASE_URL}/step", {"command": action_cmd})

                reward = data["reward"]
                done = data["done"]
                obs_text = data["observation"]["terminal_output"]
                error = data["observation"].get("last_action_error", None)

                rewards.append(reward)
                log_step(step=step_count, action=action_cmd, reward=reward, done=done, error=error)

        except Exception as e:
            error = str(e)

        finally:
            success = any(r >= 1.0 for r in rewards)
            score = min(max(max(rewards) if rewards else 0.0, 0.0), 1.0)
            log_end(success=success, steps=step_count, score=score, rewards=rewards)
            all_results[t_name] = {"success": success, "steps": step_count, "score": score}

    # Final summary
    tasks_passed = sum(1 for r in all_results.values() if r["success"])
    total_tasks = len(tasks)
    print("", flush=True)
    print("=" * 42, flush=True)
    print("           FINAL RESULTS SUMMARY", flush=True)
    print("=" * 42, flush=True)
    for t_name, r in all_results.items():
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  {t_name:<16} {status}  ({r['steps']} steps, score={r['score']:.2f})", flush=True)
    print("-" * 42, flush=True)
    print(f"  Tasks Passed : {tasks_passed}/{total_tasks}", flush=True)
    print(f"  Score        : {(tasks_passed / total_tasks) * 100:.0f}%", flush=True)
    print("=" * 42, flush=True)


if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: Missing HF_TOKEN environment variable.")
        print("  Windows:   $env:HF_TOKEN = 'hf_...'")
        print("  Linux/Mac: export HF_TOKEN='hf_...'")
    else:
        run_evaluation()
