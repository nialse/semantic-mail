import json
from pathlib import Path
from typing import Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config import get_settings


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailAuthenticator:
    def __init__(self):
        self.settings = get_settings()
        if not self.settings.gmail_client_id or not self.settings.gmail_client_secret:
            raise ValueError(
                "Gmail credentials not configured. Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET or run 'smail setup'."
            )
        self.creds: Optional[Credentials] = None
        self.service = None

    def _get_credentials_file_path(self) -> Path:
        return Path("credentials/client_secret.json")

    def _create_credentials_file(self):
        creds_data = {
            "installed": {
                "client_id": self.settings.gmail_client_id,
                "client_secret": self.settings.gmail_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
            }
        }

        creds_path = self._get_credentials_file_path()
        creds_path.parent.mkdir(exist_ok=True)

        with open(creds_path, "w") as f:
            json.dump(creds_data, f, indent=2)

    def authenticate(self) -> bool:
        try:
            if not self._get_credentials_file_path().exists():
                self._create_credentials_file()

            if self.settings.credentials_path.exists():
                self.creds = Credentials.from_authorized_user_file(
                    str(self.settings.credentials_path), SCOPES
                )

            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self._get_credentials_file_path()), SCOPES
                    )
                    self.creds = flow.run_local_server(port=0)

                self.settings.credentials_path.parent.mkdir(exist_ok=True)
                with open(self.settings.credentials_path, "w") as token:
                    token.write(self.creds.to_json())

            self.service = build("gmail", "v1", credentials=self.creds)
            return True

        except Exception as e:
            print(f"Authentication failed: {e}")
            return False

    def get_service(self):
        if not self.service:
            if not self.authenticate():
                raise Exception("Failed to authenticate with Gmail")
        return self.service

    def test_connection(self) -> bool:
        try:
            service = self.get_service()
            results = service.users().getProfile(userId="me").execute()
            print(
                f"Connected to Gmail account: {results.get('emailAddress', 'Unknown')}"
            )
            return True
        except HttpError as error:
            print(f"An error occurred: {error}")
            return False
