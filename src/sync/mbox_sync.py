import mailbox
import email
from email.utils import parsedate_to_datetime, getaddresses
from datetime import datetime
from typing import List, Optional, Dict, Any
from rich.console import Console

from ..models import Email

console = Console()


class MboxSyncer:
    """Load emails from a local mbox file"""

    def __init__(self, mbox_path: str):
        self.mbox_path = mbox_path

    def _strip_html(self, html: str) -> str:
        import re
        html = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL)
        html = re.sub(r"<[^>]+>", " ", html)
        html = re.sub(r"\s+", " ", html)
        return html.strip()

    def _get_body(self, msg: email.message.Message) -> str:
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                if part.get_filename():
                    continue
                content_type = part.get_content_type()
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                text = payload.decode(errors="ignore")
                if content_type == "text/plain":
                    parts.append(text)
                elif content_type == "text/html" and not parts:
                    parts.append(self._strip_html(text))
            return "\n".join(parts)
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode(errors="ignore")
        return str(payload)

    def _parse_email(self, msg: email.message.Message, index: int) -> Optional[Email]:
        try:
            message_id = msg.get("Message-ID") or f"mbox-{index}"
            date_header = msg.get("Date")
            date = datetime.now()
            if date_header:
                try:
                    date = parsedate_to_datetime(date_header)
                except Exception:
                    pass

            sender = email.utils.parseaddr(msg.get("From", ""))[1]
            recipients = []
            for field in ["To", "Cc"]:
                if msg.get(field):
                    recipients.extend(addr for name, addr in getaddresses([msg.get(field)]))

            body = self._get_body(msg)
            snippet = body[:100].replace("\n", " ").strip()

            attachments: List[Dict[str, Any]] = []
            if msg.is_multipart():
                for part in msg.walk():
                    filename = part.get_filename()
                    if filename:
                        attachments.append({
                            "filename": filename,
                            "mime_type": part.get_content_type(),
                            "size": len(part.get_payload(decode=True) or b""),
                        })

            return Email(
                id=message_id,
                message_id=message_id,
                thread_id=message_id,
                subject=msg.get("Subject", "(No Subject)"),
                sender=sender,
                recipients=recipients,
                date=date,
                body=body,
                labels=[],
                snippet=snippet,
                attachments=attachments,
            )
        except Exception as e:
            console.print(f"[red]Error parsing mbox email: {e}[/red]")
            return None

    def sync_emails(self) -> List[Email]:
        console.print(f"[bold blue]Loading mbox file {self.mbox_path}...[/bold blue]")
        mbox = mailbox.mbox(self.mbox_path)
        emails: List[Email] = []
        for idx, msg in enumerate(mbox):
            email_obj = self._parse_email(msg, idx)
            if email_obj:
                emails.append(email_obj)
        mbox.close()
        console.print(f"[bold green]Loaded {len(emails)} emails from mbox[/bold green]")
        return emails
