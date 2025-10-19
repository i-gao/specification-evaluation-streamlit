import streamlit as st
from typing import Literal
import os
import json
import io
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account


class LocalConnection:
    def __init__(self, path: str):
        self.root = path

    def ls(self, path: str):
        if not os.path.exists(os.path.join(self.root, path)):
            raise FileNotFoundError(f"Path {path} does not exist")
        return os.listdir(os.path.join(self.root, path))

    def read(self, path: str):
        print(f"Reading {os.path.join(self.root, path)}")
        if not os.path.exists(os.path.join(self.root, path)):
            raise FileNotFoundError(f"Path {path} does not exist")
        with open(os.path.join(self.root, path), "r") as f:
            contents = f.read()
        if path.endswith(".json"):
            return json.loads(contents)
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in contents.splitlines()]
        if path.endswith(".csv"):
            return pd.read_csv(io.StringIO(contents))
        return contents

    def write(self, path: str, content: str):
        print(f"Writing {os.path.join(self.root, path)}")
        os.makedirs(os.path.dirname(os.path.join(self.root, path)), exist_ok=True)
        with open(os.path.join(self.root, path), "w") as f:
            f.write(content)


class GcloudConnection:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        # Get credentials from Streamlit secrets
        try:
            gcs_secrets = st.secrets["connections"]["gcs"]

            # Create credentials from the secrets
            credentials_info = {
                "type": gcs_secrets["type"],
                "project_id": gcs_secrets["project_id"],
                "private_key_id": gcs_secrets["private_key_id"],
                "private_key": gcs_secrets["private_key"],
                "client_email": gcs_secrets["client_email"],
                "client_id": gcs_secrets["client_id"],
                "auth_uri": gcs_secrets["auth_uri"],
                "token_uri": gcs_secrets["token_uri"],
                "auth_provider_x509_cert_url": gcs_secrets[
                    "auth_provider_x509_cert_url"
                ],
                "client_x509_cert_url": gcs_secrets["client_x509_cert_url"],
            }

            credentials = service_account.Credentials.from_service_account_info(
                credentials_info
            )
            self.client = storage.Client(
                credentials=credentials, project=gcs_secrets["project_id"]
            )

        except KeyError as e:
            raise ValueError(
                f"Missing required GCS credentials in Streamlit secrets: {e}"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize GCS client: {e}")

        self.bucket = self.client.bucket(bucket_name)

    def ls(self, path: str):
        """List files/blobs in the given path (prefix)"""
        blobs = self.bucket.list_blobs(prefix=path)
        # Extract just the filenames, removing the path prefix
        files = []
        for blob in blobs:
            # Remove the path prefix and get just the filename
            relative_path = (
                blob.name[len(path) :] if blob.name.startswith(path) else blob.name
            )
            if relative_path and not relative_path.endswith("/"):
                files.append(relative_path)
        return files

    def read(self, path: str):
        """Read file content from GCS"""
        print(f"Reading gs://{self.bucket_name}/{path}")
        blob = self.bucket.blob(path)

        if not blob.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        contents = blob.download_as_text()

        if path.endswith(".json"):
            return json.loads(contents)
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in contents.splitlines()]
        if path.endswith(".csv"):
            return pd.read_csv(io.StringIO(contents))
        return contents

    def write(self, path: str, content: str):
        """Write content to GCS"""
        print(f"Writing gs://{self.bucket_name}/{path}")
        blob = self.bucket.blob(path)
        blob.upload_from_string(content)


class ConnectionWrapper:
    def __init__(self, root: str, client):
        self.root = root
        self.client = client

    def ls(self, path: str):
        return self.client.ls(os.path.join(self.root, path))

    def read(self, path: str):
        return self.client.read(os.path.join(self.root, path))

    def write(self, path: str, content: str):
        return self.client.write(os.path.join(self.root, path), content)


def setup_connection(connection_type: Literal["gcs", "local"] = None, path: str = None):
    """
    Setup the connection to the file system, stored in the session state as 
    st.session_state.connection.
    Args:
        connection_type: The type of connection to use, either "gcs" or "local"
        path: The path to the root of the file system
    Returns:
        The connection object
    """
    if connection_type is None:
        connection_type = st.secrets["file_connection"]["type"]
    if path is None:
        path = st.secrets["file_connection"]["path"]
    if "connection" not in st.session_state:
        if connection_type == "gcs":
            st.session_state.connection = GcloudConnection(path)
        elif connection_type == "local":
            st.session_state.connection = LocalConnection(path)
        else:
            raise ValueError(f"Invalid connection type: {connection_type}")
        print("Connection successfully set up")
    return st.session_state.connection
