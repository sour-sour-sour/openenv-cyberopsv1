import os
import json
import urllib.request
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load .env for local testing; judges will inject these variables directly
load_dotenv()

# MANDATORY CONFIGURATION 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# ENV_BASE_URL points to your FastAPI/main.py server
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BENCHMARK = "cyber-ops-v1"

# MANDATORY: Use the OpenAI client library as the interface 
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

# ── Helper for HTTP requests (Avoids 'requests' library dependency) ──────────

def env_request(url: str, method: str = "POST", data: dict = None):
    req = urllib.request.Request(url, method=method)
    json_data = json.dumps(data).encode('utf-8') if data else None
    if json_data:
        req.add_header('Content-Type', 'application/json')
    
    with urllib.request.urlopen(req, data=json_data) as response:
        return json.loads(response.read().decode('utf-8'))

# ── Logging helpers (STDOUT FORMAT MANDATORY) [cite: 3, 4, 5] ──────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── LLM call using the required OpenAI Client [cite: 13, 17] ───────────────────

def ask_llm(task_name: str, obs_text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a Cyber Expert. Output ONLY a single raw bash command. No markdown."},
            {"role": "user", "content": f"Task: {task_name}\nTerminal: {obs_text}"}
        ],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip().replace("`", "")

# ── Evaluation Loop ───────────────────────────────────────────────────────────

def run_evaluation():
    tasks = ["log-analysis", "process-hunt", "perm-fix"]

    for t_name in tasks:
        log_start(task=t_name, env=BENCHMARK, model=MODEL_NAME)
        rewards = []
        step_count = 0
        done = False

        try:
            # Reset the environment for the current task [cite: 18]
            reset_data = env_request(f"{ENV_BASE_URL}/reset?task={t_name}")
            obs_text = reset_data["terminal_output"]

            while not done and step_count < 8: # MAX_STEPS 8 
                step_count += 1
                action_cmd = ask_llm(t_name, obs_text)

                # Send command to environment [cite: 19]
                step_data = env_request(f"{ENV_BASE_URL}/step", method="POST", data={"command": action_cmd})
                
                # GRADER LOGIC: Map intermediate rewards to 0.49
                raw_reward = float(step_data["reward"])
                reward = 0.49 if 0.0 < raw_reward < 1.0 else raw_reward

                done = step_data["done"]
                obs_text = step_data["observation"]["terminal_output"]
                error = step_data.get("error", None)

                rewards.append(reward)
                log_step(step_count, action_cmd, reward, done, error) # [cite: 21]

        except Exception as e:
            print(f"[DEBUG] Error: {e}", flush=True)
        finally:
            success = any(r >= 1.0 for r in rewards)
            score = 1.0 if success else (max(rewards) if rewards else 0.0) # [cite: 22]
            log_end(success, step_count, score, rewards) # [cite: 15]

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY/HF_TOKEN not found in environment or .env file.")
    else:
        run_evaluation()
