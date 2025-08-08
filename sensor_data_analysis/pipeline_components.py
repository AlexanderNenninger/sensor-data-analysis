from __future__ import annotations

from pathlib import Path
from typing import IO, Callable, Iterator, TypeVar, Any
from zipfile import ZipFile
from itertools import islice

import polars as pl
import ray
from cloudpathlib import CloudPath

# 'primitive' types we might abstract later.
DataItemT = TypeVar("DataItemT", covariant=True)

SendableLocation = Path | CloudPath
Location = SendableLocation | IO[bytes]  # Allow both Path and file-like objects


# filtering files
FilePredicateT = Callable[[Location], bool]

# reading files
DataReaderT = Callable[[Location], Iterator[DataItemT]]

# Scanner type
ScannerT = Callable[
    [Location, DataReaderT[DataItemT], FilePredicateT], Iterator[DataItemT]
]


def scan_zip_dataset(
    location: Location,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    with ZipFile(location, "r") as zip_ref:
        # Find files that match the predicate
        for file_info in zip_ref.filelist:
            # Open each file and read its data
            with zip_ref.open(file_info) as file_handle:
                if not file_predicate(file_handle):
                    continue
                yield from data_reader(file_handle)


def scan_directory_ray(
    location: SendableLocation,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
    worker_cfg: dict[str, Any] = {},
) -> Iterator[DataItemT]:
    # Iterate through all files in the directory
    assert isinstance(location, Path), "Location must be a Path object"
    timeout = worker_cfg.pop("timeout", None)
    ignore_errors = worker_cfg.pop("ignore_errors", False)
    num_items = worker_cfg.pop("num_items", None)
    file_paths = islice(
        (file_path for file_path in location.glob("**/*") if file_predicate(file_path)),
        num_items,
    )

    @ray.remote(**worker_cfg)
    def task_wrapper(location: Location) -> list[DataItemT]:
        return list(data_reader(location))

    tasks = [task_wrapper.remote(file_path) for file_path in file_paths]
    for obj_ref in tasks:
        try:
            result = ray.get(obj_ref, timeout=timeout)
            yield from result
        except Exception as _e:
            if not ignore_errors:
                raise
            else:
                yield from []


def scan_directory_local(
    location: Location,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    # Iterate through all files in the directory
    assert isinstance(location, Path), "Location must be a Path object"
    file_paths = (
        file_path for file_path in location.glob("**/*") if file_predicate(file_path)
    )

    for file_path in file_paths:
        yield from data_reader(file_path)


def scan_directory(
    location: SendableLocation,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
    worker_cfg: dict[str, Any] = {},
) -> Iterator[DataItemT]:
    if worker_cfg:
        yield from scan_directory_ray(location, data_reader, file_predicate, worker_cfg)
    else:
        yield from scan_directory_local(location, data_reader, file_predicate)


def scan_csv_file(location: Location) -> Iterator[pl.DataFrame]:
    sensor = location.name
    sensor = sensor.rstrip(".csv") if sensor else None
    yield (
        pl.scan_csv(str(location))
        .with_columns(
            pl.lit(sensor).alias("sensor"),
        )
        .collect()
    )


def is_measurement(location: Location) -> bool:
    return location.name.endswith(".csv") and ("meta" not in location.name)


def is_zip_file(
    location: Location,
) -> bool:
    return location.name.endswith(".zip")


def zip_scanner(location: Location) -> Iterator[pl.DataFrame]:
    yield from scan_zip_dataset(
        location, data_reader=scan_csv_file, file_predicate=is_measurement
    )


def scan_dataset(
    location: SendableLocation,
) -> Iterator[pl.DataFrame]:
    yield from scan_directory(
        location,
        data_reader=zip_scanner,
        file_predicate=is_zip_file,
    )
