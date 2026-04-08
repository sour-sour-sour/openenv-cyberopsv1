import os

class CyberTask:
    def __init__(self, name, difficulty, goal, solution_keyword):
        self.name = name
        self.difficulty = difficulty
        self.goal = goal
        self.solution_keyword = solution_keyword

TASKS = {
    "log-analysis": CyberTask("log-analysis", "easy", "Analyze /var/log/auth.log and identify the attacker IP.", "192.168.1.105"),
    "process-hunt": CyberTask("process-hunt", "medium", "Find and kill the malicious PID for './hidden_miner'.", "kill -9"),
    "perm-fix": CyberTask("perm-fix", "hard", "Locate world-writable files in /etc and change permissions to 644.", "chmod 644")
}

def get_reward(task_name, action_cmd):
    task = TASKS[task_name]
    #tiny reward
    if "ls" in action_cmd or "grep" in action_cmd:
        return 0.20
    #final reward
    if task.solution_keyword in action_cmd:
        return 1.00
    #mistake
    if "rm -rf" in action_cmd:
        return -0.50
    return 0.00
