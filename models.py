from pydantic import BaseModel
from typing import Optional


class Action(BaseModel):
    command: str


class Observation(BaseModel):
    terminal_output: str
    last_action_error: Optional[str] = None


class Reward(BaseModel):
    value: float
    done: bool
