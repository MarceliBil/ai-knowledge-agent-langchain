import os
import json
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader

load_dotenv()


def load_docs():
    creds_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    loader = GoogleDriveLoader(
        folder_id=folder_id,
        credentials_info=creds,
        recursive=True
    )

    docs = loader.load()
    return docs
