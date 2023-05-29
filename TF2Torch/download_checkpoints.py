import os

import gdown


def download_from_GDrive(file_id: str, path_local: str):
    """Downloads a file from Google Drive."""
    os.makedirs(os.path.dirname(path_local), exist_ok=True)
    url = f"https://drive.google.com/uc?&id={file_id}&confirm=t"
    gdown.download(url, output=path_local, quiet=False)


if __name__ == "__main__":
    download_from_GDrive(file_id="1nqv9R4YnpCgsjeEXcq4t9fVBZcW4n-Ea", path_local="checkpoints/classify_character.h5")
