import os
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain_google_community import GoogleDriveLoader

load_dotenv()

creds_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

creds = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

loader = GoogleDriveLoader(
    folder_id=folder_id,
    credentials=creds,
    recursive=True
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content[:200])
