import json
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from rich.console import Console

from ..models import Email
from ..config import get_settings
from ..embedding.base_embedder import BaseEmbedder
from .chroma_client import get_chroma_client


console = Console()


class EmailVectorStore:
    def __init__(self, embedder: BaseEmbedder):
        self.settings = get_settings()
        self.embedder = embedder
        self.model_id = embedder.get_model_id()
        self.collection_name = f"emails_{self.model_id}"

        self.client = get_chroma_client(self.settings.chroma_persist_directory)

        self._init_collection()

    def _init_collection(self):
        try:
            self.collection = self.client.get_collection(self.collection_name)
            console.print(
                f"[green]Loaded existing collection '{self.collection_name}'[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error initializing collection: {e}[/red]")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "model_id": self.model_id,
                    "embedding_dimension": self.embedder.get_embedding_dimension(),
                },
            )
            console.print(
                f"[green]Created new collection '{self.collection_name}'[/green]"
            )
            console.print(f"[blue]Using embedder: {self.model_id}[/blue]")

    def _get_sync_metadata_path(self) -> Path:
        """Get path to sync metadata file"""
        metadata_dir = Path(self.settings.chroma_persist_directory) / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        return metadata_dir / f"{self.collection_name}_sync.json"

    def update_last_sync_date(self):
        """Update the last sync date in a separate metadata file"""
        try:
            metadata_path = self._get_sync_metadata_path()
            sync_data = {
                "last_sync_date": datetime.now().isoformat(),
                "collection_name": self.collection_name,
                "model_id": self.model_id,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(sync_data, f)
        except Exception as e:
            console.print(f"[red]Error updating last sync date: {e}[/red]")

    def get_last_sync_date(self) -> Optional[datetime]:
        """Get the last sync date from metadata file"""
        try:
            metadata_path = self._get_sync_metadata_path()
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    sync_data = json.load(f)
                    last_sync_str = sync_data.get("last_sync_date")
                    if last_sync_str:
                        return datetime.fromisoformat(last_sync_str)
            return None
        except Exception as e:
            console.print(f"[red]Error getting last sync date: {e}[/red]")
            return None

    def add_emails(
        self, emails_with_embeddings: List[Tuple[Email, Optional[List[float]]]]
    ):
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        skipped_no_embedding = 0
        for email, embedding in emails_with_embeddings:
            if embedding is None:
                skipped_no_embedding += 1
                continue

            documents.append(email.content_for_embedding)
            embeddings.append(embedding)
            ids.append(email.id)

            metadata = {
                "subject": email.subject,
                "sender": email.sender,
                "date": email.date.isoformat(),
                "message_id": email.message_id,
                "thread_id": email.thread_id,
                "snippet": email.snippet[:500],
                "labels": json.dumps(email.labels),
                "has_attachments": len(email.attachments) > 0,
            }
            metadatas.append(metadata)

        if skipped_no_embedding > 0:
            console.print(
                f"[yellow]Skipped {skipped_no_embedding} emails without embeddings[/yellow]"
            )

        if documents:
            try:
                # Batch check for existing emails to avoid loading all IDs for large collections
                batch_size = 100
                new_documents = []
                new_embeddings = []
                new_metadatas = []
                new_ids = []
                duplicates = 0

                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i : i + batch_size]

                    # Check which IDs already exist in this batch
                    try:
                        existing = self.collection.get(ids=batch_ids, include=[])
                        existing_set = set(existing["ids"])
                    except Exception:
                        # If get fails, assume none exist
                        existing_set = set()

                    # Add only non-existing emails from this batch
                    for j, email_id in enumerate(batch_ids):
                        idx = i + j
                        if email_id not in existing_set:
                            new_documents.append(documents[idx])
                            new_embeddings.append(embeddings[idx])
                            new_metadatas.append(metadatas[idx])
                            new_ids.append(ids[idx])
                        else:
                            duplicates += 1

                if new_documents:
                    # Add in batches to avoid potential memory issues
                    for i in range(0, len(new_ids), batch_size):
                        end_idx = min(i + batch_size, len(new_ids))
                        self.collection.add(
                            documents=new_documents[i:end_idx],
                            embeddings=new_embeddings[i:end_idx],
                            metadatas=new_metadatas[i:end_idx],
                            ids=new_ids[i:end_idx],
                        )

                    console.print(
                        f"[green]✓ Added {len(new_documents)} new emails to collection '{self.collection_name}'[/green]"
                    )
                    if duplicates > 0:
                        console.print(
                            f"[dim]  Skipped {duplicates} duplicate emails (already in collection)[/dim]"
                        )

                    # Update last sync date after successful add
                    self.update_last_sync_date()
                else:
                    console.print(
                        f"[yellow]All {duplicates} emails already exist in collection '{self.collection_name}'[/yellow]"
                    )

            except Exception as e:
                console.print(f"[red]Error adding emails to vector store: {e}[/red]")
                raise

    def search(
        self, query_embedding: List[float], n_results: int = 10
    ) -> List[Tuple[str, float, dict]]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            search_results = []
            for i in range(len(results["ids"][0])):
                email_id = results["ids"][0][i]
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]

                search_results.append((email_id, distance, metadata))

            return search_results

        except Exception as e:
            console.print(f"[red]Error searching vector store: {e}[/red]")
            return []

    def get_email_by_id(self, email_id: str) -> Optional[dict]:
        try:
            results = self.collection.get(
                ids=[email_id], include=["documents", "metadatas"]
            )

            if results["ids"]:
                return {
                    "id": email_id,
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0],
                }
            return None

        except Exception as e:
            console.print(f"[red]Error retrieving email: {e}[/red]")
            return None

    def check_emails_exist(self, email_ids: List[str]) -> Tuple[set, set]:
        """Check which emails already exist in the collection.
        Returns (existing_ids, new_ids)"""
        try:
            existing = self.collection.get(ids=email_ids, include=[])
            existing_set = set(existing["ids"])
            new_set = set(email_ids) - existing_set
            return existing_set, new_set
        except Exception:
            # If check fails, assume all are new
            return set(), set(email_ids)

    def get_stats(self) -> dict:
        try:
            count = self.collection.count()
            metadata = self.collection.metadata

            # Get last sync date from file
            last_sync_date = None
            last_sync_dt = self.get_last_sync_date()
            if last_sync_dt:
                last_sync_date = last_sync_dt.isoformat()

            return {
                "total_emails": count,
                "collection_name": self.collection_name,
                "model_id": metadata.get("model_id", "unknown"),
                "embedding_dimension": metadata.get("embedding_dimension", "unknown"),
                "persist_directory": str(self.settings.chroma_persist_directory),
                "last_sync_date": last_sync_date,
            }
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")
            return {"total_emails": 0}

    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            console.print("[green]Collection cleared successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error clearing collection: {e}[/red]")
            raise

    @classmethod
    def list_collections(cls):
        settings = get_settings()
        client = get_chroma_client(settings.chroma_persist_directory)

        collections = client.list_collections()
        collection_data = []

        # Add sync metadata from files
        metadata_dir = Path(settings.chroma_persist_directory) / "metadata"

        for c in collections:
            metadata = dict(c.metadata)

            # Try to load sync metadata
            sync_file = metadata_dir / f"{c.name}_sync.json"
            if sync_file.exists():
                try:
                    with open(sync_file, "r", encoding="utf-8") as f:
                        sync_data = json.load(f)
                        metadata["last_sync_date"] = sync_data.get("last_sync_date")
                except Exception:
                    pass

            collection_data.append((c.name, metadata))

        return collection_data

    @classmethod
    def find_matching_collections(
        cls, provider: Optional[str] = None, model: Optional[str] = None
    ):
        """Find collections matching the given provider and/or model criteria"""
        all_collections = cls.list_collections()
        matches = []

        for name, metadata in all_collections:
            model_id = metadata.get("model_id", "")

            # If specific model is requested, match exactly
            if model:
                if (
                    provider == "ollama"
                    and model_id == f"ollama_{model.replace(':', '_')}"
                ):
                    matches.append((name, metadata))
                elif (
                    provider == "openai"
                    and model_id == f"openai_{model.replace('-', '_')}"
                ):
                    matches.append((name, metadata))
                elif not provider:
                    # Try to match any provider
                    if (
                        model_id == f"ollama_{model.replace(':', '_')}"
                        or model_id == f"openai_{model.replace('-', '_')}"
                    ):
                        matches.append((name, metadata))
            # If only provider is specified, match all collections from that provider
            elif provider:
                if model_id.startswith(f"{provider}_"):
                    matches.append((name, metadata))
            # If neither is specified, return all collections
            else:
                matches.append((name, metadata))

        return matches

    @classmethod
    def get_collection_email_count(cls, collection_name: str):
        """Get the number of emails in a specific collection"""
        settings = get_settings()
        client = get_chroma_client(settings.chroma_persist_directory)

        try:
            collection = client.get_collection(collection_name)
            return collection.count()
        except Exception as e:
            console.print(f"[red]Error getting collection email count: {e}[/red]")
            return 0
