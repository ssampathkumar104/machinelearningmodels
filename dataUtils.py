import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO

def load_data(filename='input.txt'):
    """
    Loads key-value pairs from a text file where each line is formatted as key=value.
    """
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            config[key] = value
    return config

def access_data_from_drive(file_id):
    """
    Accesses a file from Google Drive using its file ID and returns its content as a DataFrame.
    """
    # Define scope for accessing Google Drive
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # Authenticate using OAuth2 credentials
    flow = InstalledAppFlow.from_client_secrets_file('./credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)

    # Build the Google Drive service
    service = build('drive', 'v3', credentials=creds)

    # Request file metadata
    file_metadata = service.files().get(fileId=file_id).execute()
    print("File Metadata:", file_metadata)

    # Read the file content into memory
    file_content = BytesIO()

    # Request the media content from Google Drive
    request = service.files().get_media(fileId=file_id)
    media_downloader = MediaIoBaseDownload(file_content, request)

    # Download the file content into the buffer
    done = False
    while not done:
        status, done = media_downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")

    # Move the buffer's position back to the start
    file_content.seek(0)

    # Process the file content as a CSV
    df = pd.read_csv(file_content)
    print(df)
    return df
