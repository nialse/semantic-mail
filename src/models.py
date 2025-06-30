from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Email(BaseModel):
    id: str = Field(..., description="Unique email ID from Gmail")
    message_id: str = Field(..., description="Message-ID header value")
    thread_id: str = Field(..., description="Gmail thread ID")
    subject: str = Field(..., description="Email subject")
    sender: str = Field(..., description="Email sender")
    recipients: List[str] = Field(default_factory=list, description="Email recipients")
    date: datetime = Field(..., description="Email date")
    body: str = Field(..., description="Email body content")
    labels: List[str] = Field(default_factory=list, description="Gmail labels")
    snippet: str = Field(default="", description="Email snippet")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Attachment metadata")
    
    @property
    def content_for_embedding(self) -> str:
        return f"Subject: {self.subject}\nFrom: {self.sender}\nTo: {', '.join(self.recipients)}\n\n{self.body}"


class SearchResult(BaseModel):
    email: Email
    score: float = Field(..., description="Similarity score")
    distance: float = Field(..., description="Vector distance")


class SyncStatus(BaseModel):
    total_emails: int = 0
    synced_emails: int = 0
    embedded_emails: int = 0
    last_sync: Optional[datetime] = None
    status: str = "idle" 