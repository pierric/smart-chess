import glob

import numpy as np
from ffcv.loader import Loader
from ffcv.fields.decoders import NDArrayDecoder,  IntDecoder
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField


FIELDS = {
    "board": NDArrayField(shape=(8, 8, 119), dtype=np.dtype("int")),
    "move": NDArrayField(shape=(4672,), dtype=np.dtype("float32")),
    "outcome": IntField(),
}


def write_beton(path, dataset):
    writer = DatasetWriter(path, FIELDS, num_workers=16)
    writer.from_indexed_dataset(dataset)


def join_betons(path_pattern, outfile):
    files = glob.glob(path_pattern, recursive=False)
    dataset = []
    for inp in files:
        print(f"Loading {inp} ...")
        loader = Loader(
            inp,
            batch_size=1,
            pipelines={
                "board": [NDArrayDecoder()],
                "move": [NDArrayDecoder()],
                "outcome": [IntDecoder()]
            }
        )
        for item in loader:
            b = item[0].reshape((8, 8, 119))
            m = item[1].reshape((4672,))
            r = item[2]
            dataset.append((b, m, r))

    writer = DatasetWriter(outfile, FIELDS, num_workers=16)
    writer.from_indexed_dataset(dataset)

