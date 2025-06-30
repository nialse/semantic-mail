import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from rich.console import Console
from googleapiclient.errors import HttpError

from ..models import Email
from ..auth.gmail_auth import GmailAuthenticator
from ..config import get_settings


console = Console()


class GmailSyncer:
    def __init__(self, authenticator: GmailAuthenticator):
        self.auth = authenticator
        self.settings = get_settings()
        self.service = None
        
    def _get_email_content(self, msg_data: Dict[str, Any]) -> str:
        parts = msg_data.get('payload', {}).get('parts', [])
        
        if not parts:
            body_data = msg_data.get('payload', {}).get('body', {}).get('data', '')
            if body_data:
                return base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
        
        body_text = []
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                data = part.get('body', {}).get('data', '')
                if data:
                    body_text.append(base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore'))
            elif part.get('mimeType') == 'text/html' and not body_text:
                data = part.get('body', {}).get('data', '')
                if data:
                    html_content = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    body_text.append(self._strip_html(html_content))
        
        return '\n'.join(body_text)
    
    def _strip_html(self, html: str) -> str:
        import re
        html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL)
        html = re.sub(r'<[^>]+>', ' ', html)
        html = re.sub(r'\s+', ' ', html)
        return html.strip()
    
    def _parse_email_headers(self, headers: List[Dict[str, str]]) -> Dict[str, str]:
        header_dict = {}
        for header in headers:
            header_dict[header['name'].lower()] = header['value']
        return header_dict
    
    def _extract_email_address(self, email_string: str) -> str:
        import re
        match = re.search(r'<([^>]+)>', email_string)
        if match:
            return match.group(1)
        return email_string
    
    def _parse_email(self, msg_data: Dict[str, Any]) -> Optional[Email]:
        try:
            headers = self._parse_email_headers(msg_data.get('payload', {}).get('headers', []))
            
            # Try to parse date from headers
            date_str = headers.get('date', '')
            date = None
            
            if date_str:
                # Try various date formats
                date_formats = [
                    '%a, %d %b %Y %H:%M:%S %z',  # Standard RFC format with timezone
                    '%a, %d %b %Y %H:%M:%S %Z',  # With timezone name
                    '%d %b %Y %H:%M:%S %z',       # Without day name
                    '%a, %d %b %Y %H:%M:%S',      # Without timezone
                    '%d %b %Y %H:%M:%S',          # Minimal format
                ]
                
                for fmt in date_formats:
                    try:
                        date = datetime.strptime(date_str.strip(), fmt)
                        break
                    except ValueError:
                        continue
                
                # If still no luck, try parsing with timezone offset like +0000 or -0800
                if not date and date_str:
                    import re
                    # Remove parenthetical timezone names like (PST) or (UTC)
                    date_str_clean = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)
                    # Try parsing again
                    try:
                        date = datetime.strptime(date_str_clean.strip(), '%a, %d %b %Y %H:%M:%S %z')
                    except ValueError:
                        pass
            
            # If we still don't have a date, use Gmail's internal timestamp
            if not date:
                # Gmail provides internal timestamp in milliseconds
                internal_date = msg_data.get('internalDate')
                if internal_date:
                    try:
                        # Convert milliseconds to seconds
                        timestamp = int(internal_date) / 1000
                        date = datetime.fromtimestamp(timestamp)
                    except (ValueError, TypeError):
                        pass
            
            # Last resort: use current time (but warn about it)
            if not date:
                console.print(f"[yellow]Warning: Could not parse date for email {msg_data.get('id', 'unknown')}, using current time[/yellow]")
                date = datetime.now()
            
            sender = self._extract_email_address(headers.get('from', ''))
            
            recipients = []
            for field in ['to', 'cc']:
                if field in headers:
                    for email in headers[field].split(','):
                        recipients.append(self._extract_email_address(email.strip()))
            
            message_id = headers.get('message-id', msg_data['id'])

            attachments = []
            for part in msg_data.get('payload', {}).get('parts', []):
                if part.get('filename'):
                    attachments.append({
                        'filename': part['filename'],
                        'mime_type': part.get('mimeType', ''),
                        'size': part.get('body', {}).get('size', 0)
                    })
            
            return Email(
                id=msg_data['id'],
                message_id=message_id,
                thread_id=msg_data['threadId'],
                subject=headers.get('subject', '(No Subject)'),
                sender=sender,
                recipients=recipients,
                date=date,
                body=self._get_email_content(msg_data),
                labels=msg_data.get('labelIds', []),
                snippet=msg_data.get('snippet', ''),
                attachments=attachments
            )
            
        except Exception as e:
            console.print(f"[red]Error parsing email {msg_data.get('id', 'unknown')}: {e}[/red]")
            return None
    
    def get_all_messages(self, query: str = "", max_results: Optional[int] = None) -> List[str]:
        try:
            service = self.auth.get_service()
            message_ids = []
            next_page_token = None
            
            with console.status("[bold green]Fetching email list...") as status:
                while True:
                    results = service.users().messages().list(
                        userId='me',
                        q=query,
                        pageToken=next_page_token,
                        maxResults=min(500, max_results) if max_results else 500
                    ).execute()
                    
                    messages = results.get('messages', [])
                    message_ids.extend([msg['id'] for msg in messages])
                    
                    status.update(f"[bold green]Fetched {len(message_ids)} email IDs...")
                    
                    if max_results and len(message_ids) >= max_results:
                        message_ids = message_ids[:max_results]
                        break
                    
                    next_page_token = results.get('nextPageToken')
                    if not next_page_token:
                        break
            
            console.print(f"[green]Found {len(message_ids)} emails[/green]")
            return message_ids
            
        except HttpError as error:
            console.print(f"[red]An error occurred: {error}[/red]")
            return []
    
    def fetch_emails(self, message_ids: List[str]) -> List[Email]:
        try:
            service = self.auth.get_service()
            emails = []
            
            with tqdm(total=len(message_ids), desc="Fetching emails") as pbar:
                for msg_id in message_ids:
                    try:
                        msg = service.users().messages().get(
                            userId='me',
                            id=msg_id,
                            format='full'
                        ).execute()
                        
                        email = self._parse_email(msg)
                        if email:
                            emails.append(email)
                        
                    except HttpError as error:
                        console.print(f"[red]Error fetching email {msg_id}: {error}[/red]")
                    
                    pbar.update(1)
            
            console.print(f"[green]Successfully fetched {len(emails)} emails[/green]")
            return emails
            
        except Exception as e:
            console.print(f"[red]Failed to fetch emails: {e}[/red]")
            return []
    
    def sync_emails(self, query: str = "", max_results: Optional[int] = None) -> List[Email]:
        console.print("[bold blue]Starting email sync...[/bold blue]")
        
        message_ids = self.get_all_messages(query, max_results)
        
        if not message_ids:
            console.print("[yellow]No emails found matching the criteria[/yellow]")
            return []
        
        emails = self.fetch_emails(message_ids)
        
        console.print(f"[bold green]Email sync complete! Fetched {len(emails)} emails[/bold green]")
        return emails 