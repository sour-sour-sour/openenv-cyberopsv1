from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional


class Action(BaseModel):
    command: str


class Observation(BaseModel):
    terminal_output: str


class CyberEnv:
    def __init__(self):
        self.tasks = {
            "log-analysis": "Identify the attacker IP in /var/log/auth.log",
            "process-hunt": "Find and kill the malicious PID for './hidden_miner'",
            "perm-fix": "Secure world-writable files in /etc to 644"
        }
        self.current_task = "log-analysis"
        self.steps = 0
        self.max_steps = 5
        self.last_reward = 0.0
        self.done = False

    def reset(self, task_name: Optional[str] = "log-analysis"):
        task_name = task_name or "log-analysis"
        self.current_task = task_name if task_name in self.tasks else "log-analysis"
        self.steps = 0
        self.last_reward = 0.0
        self.done = False
        return Observation(terminal_output=f"Task: {self.tasks[self.current_task]}")

    def state(self):
        return {
            "current_task": self.current_task,
            "task_description": self.tasks.get(self.current_task, ""),
            "steps": self.steps,
            "max_steps": self.max_steps,
            "last_reward": self.last_reward,
            "done": self.done
        }

    def step(self, action: Action):
        self.steps += 1
        cmd = action.command.lower()
        reward = 0.00
        output = "Command executed. No findings. Use 'ls', 'ps aux', or 'cat' to investigate."

        # --- TASK 1: LOG ANALYSIS ---
        if self.current_task == "log-analysis":
            if "192.168.1.105" in cmd:
                output, reward = "Success: Attacker IP 192.168.1.105 identified.", 1.00
            elif any(v in cmd for v in ["cat", "grep", "ls", "auth"]):
                output, reward = "Log Entry: Failed password for root from 192.168.1.105", 0.50

        # --- TASK 2: PROCESS HUNT ---
        elif self.current_task == "process-hunt":
            if "kill" in cmd:
                output, reward = "Success: Malicious process 999 terminated.", 1.00
            elif any(v in cmd for v in ["ps", "top", "pgrep", "aux"]):
                output, reward = "CRITICAL: Found process './hidden_miner' with PID: 999", 0.50

        # --- TASK 3: PERMISSION FIX ---
        elif self.current_task == "perm-fix":
            if "chmod" in cmd or "644" in cmd:
                output, reward = "Success: Permissions for /etc/shadow set to 644.", 1.00
            elif any(v in cmd for v in ["ls", "stat", "check", "find"]):
                output, reward = "VULNERABILITY: /etc/shadow has permissions 777", 0.50

        self.last_reward = reward
        self.done = reward >= 1.0 or self.steps >= self.max_steps
        return Observation(terminal_output=output), reward, self.done


app = FastAPI()
env = CyberEnv()


@app.post("/reset")
def reset(task: str = Query(default="log-analysis")):
    return env.reset(task)


@app.post("/step")
def step(action: Action):
    obs, reward, done = env.step(action)
    return {"observation": obs, "reward": reward, "done": done}


@app.get("/state")
def state():
    return env.state()
