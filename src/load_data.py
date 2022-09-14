import gzip
from importlib.resources import path
import struct
from pathlib import Path

import numpy as np


ALLOWED_TYPES = {
    "UNSIGNED_BYTE": b"\x08",
    "SIGNED_BYTE": b"\x09",
    "SHORT": b"\x0B",
    "INT": b"\x0C",
    "SINGLE": b"\x0D",
    "DOUBLE": b"\x0E",
}


def load_image_data(file_path:Path)-> np.ndarray:
    with gzip.open(file_path, "rb") as fp:
        _ = struct.unpack(">H", fp.read(2))  # dump padding bytes

        (data_type,) = struct.unpack(">c", fp.read(1))
        assert data_type == ALLOWED_TYPES["UNSIGNED_BYTE"]

        number_of_dimensions = ord(struct.unpack(">c", fp.read(1))[0])
        assert number_of_dimensions == 3

        (num_images,) = struct.unpack(">I", fp.read(4))
        (num_rows,) = struct.unpack(">I", fp.read(4))
        (num_cols,) = struct.unpack(">I", fp.read(4))
        assert num_rows == num_cols == 28

        raw = fp.read()
        assert len(raw) == num_images * num_rows * num_cols

    data:np.ndarray = np.frombuffer(raw, dtype=np.dtype(np.uint8).newbyteorder(">"))
    data = data.reshape((num_images, num_rows, num_cols))
    return data



def load_label_data(file_path:Path)-> np.ndarray:
    with gzip.open(file_path) as fp:
        _ = struct.unpack(">H", fp.read(2))  # dump padding bytes

        (data_type,) = struct.unpack(">c", fp.read(1))
        assert data_type == ALLOWED_TYPES["UNSIGNED_BYTE"]

        number_of_dimensions = ord(struct.unpack(">c", fp.read(1))[0])
        assert number_of_dimensions == 1

        (num_images,) = struct.unpack(">I", fp.read(4))
        
        raw = fp.read()
        assert len(raw) == num_images

    data = np.frombuffer(raw, dtype=np.dtype(np.uint8).newbyteorder(">"))
    return data


def load_train_labels():
    with gzip.open(DATA_DIR / "train-labels-idx1-ubyte.gz", "rb") as fp:
        _ = struct.unpack(">H", fp.read(2))  # dump padding bytes

        (data_type,) = struct.unpack(">c", fp.read(1))
        assert data_type == ALLOWED_TYPES["UNSIGNED_BYTE"]

        number_of_dimensions = ord(struct.unpack(">c", fp.read(1))[0])
        assert number_of_dimensions == 1

        (num_images,) = struct.unpack(">I", fp.read(4))
        assert num_images == 60_000

        raw = fp.read()
        assert len(raw) == num_images

    data = np.frombuffer(raw, dtype=np.dtype(np.uint8).newbyteorder(">"))
    return data
