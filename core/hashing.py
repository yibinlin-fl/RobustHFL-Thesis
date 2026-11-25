"""Streaming checksums for data files and run artifacts."""
import hashlib
from pathlib import Path


def stream_hash(handle, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as handle:
        return stream_hash(handle, algorithm)
