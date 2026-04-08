import os
import json
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
BENCHMARK = "cyber-ops-v1"

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Helper for built-in HTTP requests ────────────────────────────────────────

def make_request(url: str, method: str = "GET", data: dict = None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        json_data = json.dumps(data).encode('utf-8')
    else:
        json_data = None
    
    with urllib.request.urlopen(req, data=json_data) as response:
        return json.loads(response.read().decode('utf-8'))

# ── Logging helpers (Format preserved) ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── LLM call ──────────────────────────────────────────────────────────────────

def ask_llm(task_name: str, obs_text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a Cybersecurity Expert. Output ONLY a single raw bash command. No markdown. No backticks. To fix log-analysis: echo '192.168.1.105'. To fix process-hunt: kill -9 <PID>. To fix perm-fix: chmod 644 <path>."},
            {"role": "user", "content": f"Task: {task_name}\nOutput: {obs_text}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip().replace("`", "")

# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation():
    tasks = ["log-analysis", "process-hunt", "perm-fix"]
    all_results = {}

    for t_name in tasks:
        log_start(task=t_name, env=BENCHMARK, model=MODEL_NAME)
        rewards = []
        step_count = 0
        done = False

        try:
            # Replaced requests.post with make_request
            data = make_request(f"{ENV_BASE_URL}/reset?task={t_name}", method="POST")
            obs_text = data["terminal_output"]

            while not done and step_count < 5:
                step_count += 1
                action_cmd = ask_llm(t_name, obs_text)

                # Replaced requests.post with make_request
                step_data = make_request(f"{ENV_BASE_URL}/step", method="POST", data={"command": action_cmd})
                
                reward = step_data["reward"]
                done = step_data["done"]
                obs_text = step_data["observation"]["terminal_output"]
                error = step_data["observation"].get("last_action_error")

                rewards.append(reward)
                log_step(step_count, action_cmd, reward, done, error)

        except Exception as e:
            print(f"Error during task {t_name}: {e}")

        finally:
            success = any(r >= 1.0 for r in rewards)
            score = max(rewards) if rewards else 0.0
            log_end(success, step_count, score, rewards)
            all_results[t_name] = {"success": success, "steps": step_count, "score": score}

    # Summary prints remain the same...
    print("\n" + "="*42 + "\n           FINAL RESULTS SUMMARY\n" + "="*42)
    for t_name, r in all_results.items():
        print(f"  {t_name:<16} {'SUCCESS' if r['success'] else 'FAILED'}  ({r['steps']} steps, score={r['score']:.2f})")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: Missing OPENAI_API_KEY environment variable.")
    else:
        run_evaluation()
