from pydantic import BaseModel
from typing import Optional, List

class Message(BaseModel):
    sender: str
    type: str
    text: str
    timestamp: Optional[str] = None
    emotion: Optional[str] = None
    distress: Optional[float] = None


class DistressRequest(BaseModel):
    text: str
    conversationHistory: Optional[List[Message]] = None