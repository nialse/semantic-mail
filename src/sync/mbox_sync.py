import mailbox
import email
from email.utils import parsedate_to_datetime, getaddresses
from email.header import decode_header
from datetime import datetime
from typing import List, Optional, Dict, Any
from rich.console import Console
from tqdm import tqdm

from ..models import Email

console = Console()

import mmap, re, mailbox, os
from tqdm import tqdm

class FastMbox(mailbox.mbox):
    _pat = re.compile(br'\nFrom ')
    def _generate_toc(self):
        size = os.path.getsize(self._path)
        with open(self._path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            bar = tqdm(total=size, unit='B', unit_scale=True, desc='Indexing mbox')
            starts, stops, last = [0], [], 0
            for m in self._pat.finditer(mm):
                pos = m.start() + 1
                stops.append(pos)
                starts.append(pos)
                if bar: bar.update(pos - last)
                last = pos
            stops.append(len(mm))
            if bar:
                bar.update(len(mm) - last)
                bar.close()
            self._toc = dict(enumerate(zip(starts, stops)))
            self._next_key = len(self._toc)
            self._file_length = len(mm)




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

    def _decode_header(self, value: str) -> str:
        """Decode RFC2047 header values to a readable string."""
        if not value:
            return ""
        decoded = decode_header(value)
        parts = []
        for fragment, charset in decoded:
            if isinstance(fragment, bytes):
                charset = charset or "utf-8"
                try:
                    parts.append(fragment.decode(charset, errors="ignore"))
                except LookupError:
                    parts.append(fragment.decode("utf-8", errors="ignore"))
            else:
                parts.append(fragment)
        return "".join(parts)

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
                charset = part.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="ignore")
                except LookupError:
                    text = payload.decode("utf-8", errors="ignore")
                if content_type == "text/plain":
                    parts.append(text)
                elif content_type == "text/html" and not parts:
                    parts.append(self._strip_html(text))
            return "\n".join(parts)
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="ignore")
            except LookupError:
                return payload.decode("utf-8", errors="ignore")
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

            sender_header = self._decode_header(msg.get("From", ""))
            sender = email.utils.parseaddr(sender_header)[1]

            recipients = []
            for field in ["To", "Cc"]:
                header_value = msg.get(field)
                if header_value:
                    decoded = self._decode_header(header_value)
                    recipients.extend(addr for name, addr in getaddresses([decoded]))

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

            subject_header = msg.get("Subject", "(No Subject)")
            subject = self._decode_header(subject_header) or "(No Subject)"

            return Email(
                id=message_id,
                message_id=message_id,
                thread_id=message_id,
                subject=subject,
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
        mbox = FastMbox(self.mbox_path)
        emails: List[Email] = []
        total_messages = len(mbox)
        with tqdm(total=total_messages, desc="Parsing mbox") as pbar:
            for idx, msg in enumerate(mbox):
                email_obj = self._parse_email(msg, idx)
                if email_obj:
                    emails.append(email_obj)
                pbar.update(1)
        mbox.close()
        console.print(f"[bold green]Loaded {len(emails)} emails from mbox[/bold green]")
        return emails
