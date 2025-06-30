import click
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

from .auth.gmail_auth import GmailAuthenticator
from .sync.gmail_sync import GmailSyncer
from .sync.mbox_sync import MboxSyncer
from .embedding.embedder_factory import (
    get_embedder,
    get_smart_embedder,
    list_available_models,
)
from .search.vector_store import EmailVectorStore
from .search.searcher import EmailSearcher
from .answering.llm_factory import get_llm, list_available_llm_models
from .config import get_settings


console = Console()


@click.group()
def cli():
    """Semantic Email Search - Local semantic search for your Gmail"""
    pass


@cli.command()
def setup():
    """Set up Gmail OAuth credentials"""
    console.print("[bold blue]Gmail OAuth Setup[/bold blue]\n")

    console.print("To use this application, you need to:")
    console.print("1. Go to https://console.cloud.google.com/")
    console.print("2. Create a new project or select existing one")
    console.print("3. Enable Gmail API")
    console.print("4. Create OAuth 2.0 credentials (Desktop application)")
    console.print("5. Download the credentials and note the Client ID and Secret\n")

    client_id = Prompt.ask("Enter your Gmail OAuth Client ID")
    client_secret = Prompt.ask("Enter your Gmail OAuth Client Secret", password=True)

    console.print("\n[bold blue]Embedding Provider Setup[/bold blue]")
    console.print("Choose your default embedding provider:")
    console.print("1. Ollama (local, free)")
    console.print("2. OpenAI (cloud, requires API key)")

    provider_choice = Prompt.ask("Enter choice", choices=["1", "2"], default="1")
    provider = "ollama" if provider_choice == "1" else "openai"

    openai_key = ""
    if provider == "openai" or Confirm.ask(
        "\nWould you like to configure OpenAI as well?", default=False
    ):
        openai_key = Prompt.ask(
            "Enter your OpenAI API key (optional)", password=True, default=""
        )

    env_path = Path(".env")
    env_content = f"""GMAIL_CLIENT_ID={client_id}
GMAIL_CLIENT_SECRET={client_secret}

# Embedding provider: 'ollama' or 'openai'
EMBEDDING_PROVIDER={provider}

# Ollama settings
OLLAMA_HOST=http://localhost:11434

# OpenAI settings
OPENAI_API_KEY={openai_key}
OPENAI_MODEL=text-embedding-3-small

# Storage settings
CHROMA_PERSIST_DIRECTORY=data/chroma
"""

    with open(env_path, "w") as f:
        f.write(env_content)

    console.print("[green]✓ Configuration saved to .env file[/green]")
    if provider == "ollama":
        console.print("\n[yellow]Make sure Ollama is running locally![/yellow]")
        console.print("Run: ollama serve")


@cli.command()
def models():
    """List available embedding models"""
    list_available_models()

    console.print("\n[bold cyan]Current Collections:[/bold cyan]")
    collections = EmailVectorStore.list_collections()
    if collections:
        for name, metadata in collections:
            model_id = metadata.get("model_id", "unknown")
            dimension = metadata.get("embedding_dimension", "unknown")
            console.print(f"  • {name} (model: {model_id}, dimension: {dimension})")
    else:
        console.print("  [dim]No collections found[/dim]")

    console.print("\n")
    list_available_llm_models()


@cli.command()
@click.option(
    "--query", "-q", default="", help='Gmail search query (e.g., "after:2024/1/1")'
)
@click.option("--limit", "-l", type=int, help="Maximum number of emails to sync")
@click.option("--clear", is_flag=True, help="Clear existing email database before sync")
@click.option(
    "--incremental", "-i", is_flag=True, help="Only sync emails since last sync"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "openai"]),
    help="Embedding provider",
)
@click.option("--model", "-m", help="Specific model to use for embeddings")
def sync(query, limit, clear, incremental, provider, model):
    """Sync emails from Gmail and create embeddings"""
    try:
        auth = GmailAuthenticator()
        if not auth.test_connection():
            console.print("[red]Failed to connect to Gmail. Run 'setup' first.[/red]")
            return

        try:
            embedder = get_embedder(provider=provider, model=model)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Run 'models' command to see available options")
            return

        if not embedder.test_connection():
            console.print("[red]Failed to connect to embedding service.[/red]")
            return

        vector_store = EmailVectorStore(embedder)

        stats = vector_store.get_stats()
        if stats["total_emails"] > 0:
            console.print(f"\n[cyan]Current collection:[/cyan]")
            console.print(f"  Collection: {stats['collection_name']}")
            console.print(f"  Model: {stats['model_id']}")
            console.print(f"  Existing emails: {stats['total_emails']}")
            if stats.get("last_sync_date"):
                console.print(f"  Last sync: {stats['last_sync_date']}")

            if clear:
                if Confirm.ask(
                    "\n[bold red]Delete all existing emails in this collection?[/bold red]",
                    default=False,
                ):
                    vector_store.clear_collection()
                    console.print("[green]Collection cleared[/green]\n")
                else:
                    console.print("[yellow]Keeping existing emails[/yellow]\n")
            else:
                console.print("\n[dim]Adding new emails to existing collection[/dim]\n")

        # Handle incremental sync
        if incremental:
            last_sync_date = vector_store.get_last_sync_date()
            if last_sync_date:
                # Format date for Gmail query
                date_str = last_sync_date.strftime("%Y/%m/%d")
                incremental_query = f"after:{date_str}"

                if query:
                    # Combine with existing query
                    query = f"{query} {incremental_query}"
                else:
                    query = incremental_query

                console.print(
                    f"[cyan]Incremental sync: fetching emails after {date_str}[/cyan]"
                )
            else:
                console.print(
                    "[yellow]No previous sync date found, performing full sync[/yellow]"
                )

        syncer = GmailSyncer(auth)
        emails = syncer.sync_emails(query, limit)

        if emails:
            console.print(
                f"\n[cyan]Processing {len(emails)} emails from Gmail...[/cyan]"
            )

            email_ids = [email.id for email in emails]
            existing, new = vector_store.check_emails_exist(email_ids)

            if existing:
                console.print(f"[dim]Skipping {len(existing)} duplicate emails[/dim]")

            emails_with_embeddings = embedder.embed_emails(emails)
            vector_store.add_emails(emails_with_embeddings)

            stats = vector_store.get_stats()
            console.print("\n[bold green]✓ Sync complete![/bold green]")
            console.print(f"[green]Collection: {stats['collection_name']}[/green]")
            console.print(f"[green]Model: {stats['model_id']}[/green]")
            console.print(f"[green]Total emails: {stats['total_emails']}[/green]")
            if stats.get("last_sync_date"):
                console.print(f"[green]Last sync: {stats['last_sync_date']}[/green]")
        else:
            console.print("\n[yellow]No emails found to sync[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during sync: {e}[/red]")


@cli.command("sync-mbox")
@click.argument("mbox_file", type=click.Path(exists=True))
@click.option("--clear", is_flag=True, help="Clear existing email database before import")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "openai"]),
    help="Embedding provider",
)
@click.option("--model", "-m", help="Specific model to use for embeddings")
def sync_mbox(mbox_file, clear, provider, model):
    """Import emails from a local mbox file"""
    try:
        try:
            embedder = get_embedder(provider=provider, model=model)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Run 'models' command to see available options")
            return

        if not embedder.test_connection():
            console.print("[red]Failed to connect to embedding service.[/red]")
            return

        vector_store = EmailVectorStore(embedder)

        stats = vector_store.get_stats()
        if stats["total_emails"] > 0:
            console.print(f"\n[cyan]Current collection:[/cyan]")
            console.print(f"  Collection: {stats['collection_name']}")
            console.print(f"  Model: {stats['model_id']}")
            console.print(f"  Existing emails: {stats['total_emails']}")
            if stats.get("last_sync_date"):
                console.print(f"  Last sync: {stats['last_sync_date']}")

            if clear:
                if Confirm.ask(
                    "\n[bold red]Delete all existing emails in this collection?[/bold red]",
                    default=False,
                ):
                    vector_store.clear_collection()
                    console.print("[green]Collection cleared[/green]\n")
                else:
                    console.print("[yellow]Keeping existing emails[/yellow]\n")
            else:
                console.print("\n[dim]Adding new emails to existing collection[/dim]\n")

        syncer = MboxSyncer(mbox_file)
        emails = syncer.sync_emails()

        if emails:
            console.print(f"\n[cyan]Processing {len(emails)} emails from mbox...[/cyan]")

            email_ids = [email.id for email in emails]
            existing, new = vector_store.check_emails_exist(email_ids)

            if existing:
                console.print(f"[dim]Skipping {len(existing)} duplicate emails[/dim]")

            emails_with_embeddings = embedder.embed_emails(emails)
            vector_store.add_emails(emails_with_embeddings)

            stats = vector_store.get_stats()
            console.print("\n[bold green]✓ Import complete![/bold green]")
            console.print(f"[green]Collection: {stats['collection_name']}[/green]")
            console.print(f"[green]Model: {stats['model_id']}[/green]")
            console.print(f"[green]Total emails: {stats['total_emails']}[/green]")
            if stats.get("last_sync_date"):
                console.print(f"[green]Last sync: {stats['last_sync_date']}[/green]")
        else:
            console.print("\n[yellow]No emails found in mbox file[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during mbox import: {e}[/red]")


@cli.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, type=int, help="Number of results to return")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed results")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "openai"]),
    help="Embedding provider",
)
@click.option("--model", "-m", help="Specific model to use (must match indexed model)")
def search(query, limit, detailed, provider, model):
    """Search emails using semantic search"""
    try:
        try:
            embedder, collection_info = get_smart_embedder(
                provider=provider, model=model
            )
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Run 'models' command to see available options")
            return

        vector_store = EmailVectorStore(embedder)
        searcher = EmailSearcher(embedder, vector_store)

        stats = vector_store.get_stats()
        if stats["total_emails"] == 0:
            console.print("[yellow]No emails found in any collection.[/yellow]")
            console.print(f"[yellow]Run 'sync' to index emails first.[/yellow]")
            return

        console.print(f"[blue]Using collection: {collection_info}[/blue]")
        console.print(f"[blue]Model: {stats['model_id']}[/blue]\n")

        results = searcher.search(query, limit)
        searcher.display_results(results, detailed)

        if results and Confirm.ask("\nView full email details?", default=True):
            for i, result in enumerate(results):
                console.print(f"\n[bold]Result {i + 1}/{len(results)}[/bold]")
                searcher.display_email_detail(result.email)

                if i < len(results) - 1:
                    if not Confirm.ask("Continue to next email?", default=True):
                        break

    except Exception as e:
        console.print(f"[red]Error during search: {e}[/red]")


@cli.command()
@click.argument("question")
@click.option(
    "--search-limit",
    "-sl",
    default=5,
    type=int,
    help="Number of emails to search for context",
)
@click.option(
    "--provider", "-p", type=click.Choice(["ollama", "openai"]), help="LLM provider"
)
@click.option("--model", "-m", help="Specific LLM model to use")
@click.option(
    "--embedding-provider",
    "-ep",
    type=click.Choice(["ollama", "openai"]),
    help="Embedding provider for search",
)
@click.option("--embedding-model", "-em", help="Embedding model for search")
@click.option(
    "--max-tokens", "-t", default=500, type=int, help="Maximum tokens for response"
)
def ask(
    question,
    search_limit,
    provider,
    model,
    embedding_provider,
    embedding_model,
    max_tokens,
):
    """Ask a question about your emails using AI"""
    try:
        try:
            embedder, collection_info = get_smart_embedder(
                provider=embedding_provider, model=embedding_model
            )
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Run 'models' command to see available options")
            return

        vector_store = EmailVectorStore(embedder)
        searcher = EmailSearcher(embedder, vector_store)

        stats = vector_store.get_stats()
        if stats["total_emails"] == 0:
            console.print("[yellow]No emails found in any collection.[/yellow]")
            console.print("[yellow]Run 'sync' to index emails first.[/yellow]")
            return

        console.print(f"[blue]Searching in collection: {collection_info}[/blue]")
        console.print(f"[blue]Embedding model: {stats['model_id']}[/blue]")

        results = searcher.search(question, search_limit)

        if not results:
            console.print(
                "[yellow]No relevant emails found for your question.[/yellow]"
            )
            return

        console.print(f"[green]Found {len(results)} relevant emails[/green]\n")

        emails = [result.email for result in results]

        try:
            llm = get_llm(provider=provider, model=model)
            llm_model_id = llm.get_model_id()
            console.print(f"[blue]Using LLM: {llm_model_id}[/blue]")

            if not llm.test_connection():
                console.print("[red]Failed to connect to LLM service.[/red]")
                return

            console.print("\n[cyan]Generating answer...[/cyan]")
            answer = llm.answer_question(question, emails, max_tokens)

            console.print("\n[bold green]Answer:[/bold green]")
            console.print(answer)

            if Confirm.ask(
                "\n\nWould you like to see the source emails?", default=False
            ):
                console.print("\n[bold cyan]Source Emails:[/bold cyan]")
                for i, result in enumerate(results, 1):
                    console.print(
                        f"\n[dim]--- Email {i} (Score: {result.score:.3f}) ---[/dim]"
                    )
                    console.print(f"[yellow]From:[/yellow] {result.email.sender}")
                    console.print(f"[yellow]Date:[/yellow] {result.email.date}")
                    console.print(f"[yellow]Subject:[/yellow] {result.email.subject}")
                    if Confirm.ask(f"\nView full content of email {i}?", default=False):
                        searcher.display_email_detail(result.email)

        except ValueError as e:
            console.print(f"[red]Error initializing LLM: {e}[/red]")
            console.print("Make sure the LLM provider is properly configured")
            return

    except Exception as e:
        console.print(f"[red]Error during ask: {e}[/red]")


@cli.command()
def stats():
    """Show database statistics"""
    try:
        console.print("[bold blue]Email Database Statistics[/bold blue]\n")

        collections = EmailVectorStore.list_collections()
        if not collections:
            console.print(
                "[yellow]No collections found. Run 'sync' to create your first collection.[/yellow]"
            )
            return

        total_emails = 0
        for name, metadata in collections:
            model_id = metadata.get("model_id", "unknown")
            dimension = metadata.get("embedding_dimension", "unknown")
            last_sync = metadata.get("last_sync_date", "Never")

            try:
                if model_id.startswith("ollama_"):
                    model_name = model_id.replace("ollama_", "").replace("_", ":")
                    embedder = get_embedder("ollama", model_name)
                elif model_id.startswith("openai_"):
                    model_name = model_id.replace("openai_", "").replace("_", "-")
                    embedder = get_embedder("openai", model_name)
                else:
                    embedder = get_embedder()

                vector_store = EmailVectorStore(embedder)
                stats = vector_store.get_stats()
                count = stats["total_emails"]
                total_emails += count

                console.print(f"[cyan]Collection:[/cyan] {name}")
                console.print(f"  • Model: {model_id}")
                console.print(f"  • Dimension: {dimension}")
                console.print(f"  • Emails: {count}")
                console.print(f"  • Last sync: {stats.get('last_sync_date', 'Never')}")
                console.print()
            except:
                console.print(f"[cyan]Collection:[/cyan] {name}")
                console.print(f"  • Model: {model_id}")
                console.print(f"  • Dimension: {dimension}")
                console.print(f"  • Last sync: {last_sync}")
                console.print(f"  • [red]Unable to load stats[/red]")
                console.print()

        settings = get_settings()
        console.print(
            f"[bold]Total emails across all collections:[/bold] {total_emails}"
        )
        console.print(
            f"[bold]Storage location:[/bold] {settings.chroma_persist_directory}"
        )

    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")


@cli.command()
def test():
    """Test connections to Gmail and embedding services"""
    console.print("[bold blue]Testing connections...[/bold blue]\n")

    try:
        settings = get_settings()
        console.print("[green]✓ Configuration loaded[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load configuration: {e}[/red]")
        console.print("Run 'setup' command first")
        return

    console.print("\n[cyan]Testing Gmail connection...[/cyan]")
    try:
        auth = GmailAuthenticator()
        if auth.test_connection():
            console.print("[green]✓ Gmail connection successful[/green]")
        else:
            console.print("[red]✗ Gmail connection failed[/red]")
    except Exception as e:
        console.print(f"[red]✗ Gmail error: {e}[/red]")

    console.print("\n[cyan]Testing Ollama connection...[/cyan]")
    try:
        embedder = get_embedder("ollama")
        if embedder.test_connection():
            console.print("[green]✓ Ollama connection successful[/green]")
        else:
            console.print("[red]✗ Ollama connection failed[/red]")
    except Exception as e:
        console.print(f"[red]✗ Ollama error: {e}[/red]")

    if settings.openai_api_key:
        console.print("\n[cyan]Testing OpenAI connection...[/cyan]")
        try:
            embedder = get_embedder("openai")
            if embedder.test_connection():
                console.print("[green]✓ OpenAI connection successful[/green]")
            else:
                console.print("[red]✗ OpenAI connection failed[/red]")
        except Exception as e:
            console.print(f"[red]✗ OpenAI error: {e}[/red]")
    else:
        console.print("\n[yellow]OpenAI API key not configured[/yellow]")


if __name__ == "__main__":
    cli()
