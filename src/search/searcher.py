import json
from typing import List
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..models import Email, SearchResult
from ..embedding.ollama_embedder import OllamaEmbedder
from .vector_store import EmailVectorStore
from ..config import get_settings


console = Console()


class EmailSearcher:
    def __init__(self, embedder: OllamaEmbedder, vector_store: EmailVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = get_settings()

    def search(self, query: str, n_results: int = 10) -> List[SearchResult]:
        console.print(f"[bold blue]Searching for: '{query}'[/bold blue]")

        query_embedding = self.embedder.generate_embedding(query)

        if query_embedding is None:
            console.print("[red]Failed to generate query embedding[/red]")
            return []

        total_count = self.vector_store.collection.count()
        fetch_count = min(n_results * 2, total_count)
        accumulated = []

        while True:
            search_results = self.vector_store.search(query_embedding, fetch_count)

            if not search_results:
                break

            accumulated.extend(search_results)

            dedup = {}
            for email_id, distance, metadata in accumulated:
                message_id = metadata.get("message_id", email_id)
                if message_id not in dedup or distance < dedup[message_id][1]:
                    dedup[message_id] = (email_id, distance, metadata)

            if len(dedup) >= n_results or fetch_count >= total_count:
                break

            fetch_count = min(fetch_count * 2, total_count)

        if not accumulated:
            console.print("[yellow]No results found[/yellow]")
            return []

        dedup = {}
        for email_id, distance, metadata in accumulated:
            message_id = metadata.get("message_id", email_id)
            if message_id not in dedup or distance < dedup[message_id][1]:
                dedup[message_id] = (email_id, distance, metadata)

        sorted_results = sorted(dedup.values(), key=lambda x: x[1])[:n_results]

        results = []
        for email_id, distance, metadata in sorted_results:
            email_data = self.vector_store.get_email_by_id(email_id)

            if email_data:
                email = Email(
                    id=email_id,
                    message_id=metadata.get("message_id", email_id),
                    thread_id=metadata["thread_id"],
                    subject=metadata["subject"],
                    sender=metadata["sender"],
                    recipients=[],
                    date=datetime.fromisoformat(metadata["date"]),
                    body=email_data["document"],
                    labels=json.loads(metadata.get("labels", "[]")),
                    snippet=metadata["snippet"],
                    attachments=[],
                )

                score = 1.0 - distance

                results.append(
                    SearchResult(email=email, score=score, distance=distance)
                )

        return results

    def display_results(self, results: List[SearchResult], detailed: bool = False):
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return

        table = Table(
            title="Search Results", show_header=True, header_style="bold magenta"
        )
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Date", style="green", width=12)
        table.add_column("From", style="blue", width=25)
        table.add_column("Subject", style="yellow", width=40)

        if detailed:
            table.add_column("Snippet", style="dim", width=50)

        for result in results:
            score_str = f"{result.score:.3f}"
            date_str = result.email.date.strftime("%Y-%m-%d")

            row = [
                score_str,
                date_str,
                result.email.sender[:25],
                result.email.subject[:40],
            ]

            if detailed:
                snippet = (
                    result.email.snippet[:50] + "..."
                    if len(result.email.snippet) > 50
                    else result.email.snippet
                )
                row.append(snippet)

            table.add_row(*row)

        console.print(table)

    def display_email_detail(self, email: Email):
        panel_content = f"""[bold cyan]Subject:[/bold cyan] {email.subject}
[bold cyan]From:[/bold cyan] {email.sender}
[bold cyan]Date:[/bold cyan] {email.date.strftime("%Y-%m-%d %H:%M:%S")}
[bold cyan]Thread ID:[/bold cyan] {email.thread_id}
[bold cyan]Labels:[/bold cyan] {", ".join(email.labels)}

[bold cyan]Body:[/bold cyan]
{email.body}
"""

        panel = Panel(panel_content, title=f"Email: {email.id}", expand=False)
        console.print(panel)
