"""Download the attributed UCI Wine Quality dataset without committing raw data."""

import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip"
EXPECTED = {
    "winequality-red.csv": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
    "winequality-white.csv": "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
}


def download(destination):
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED.items():
        contents = archive.read(name)
        actual = hashlib.sha256(contents).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Dataset checksum changed for {name}; verify source before updating the expected hash"
            )
        (destination / name).write_bytes(contents)
        print(f"Saved {destination / name} (SHA-256 {actual})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    download(parser.parse_args().output)
