from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Callable, Iterator, TypeVar
from zipfile import ZipFile

import polars as pl

# 'primitive' types we might abstract later.
ContextT = dict[str, Any]
DataItemT = TypeVar("DataItemT", covariant=True)
Location = IO[bytes] | Path

# filtering files
FilePredicateT = Callable[[Location, ContextT], bool]

# reading files
DataReaderT = Callable[[Location, ContextT], Iterator[DataItemT]]


def scan_zip_dataset(
    zip_location: Location,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    with ZipFile(zip_location, "r") as zip_ref:
        context: ContextT = {"root": zip_location, "file": zip_ref.filename}
        # Find files that match the predicate
        for file_info in zip_ref.filelist:
            # Open each file and read its data
            with zip_ref.open(file_info) as file_handle:
                if not file_predicate(file_handle, context):
                    continue
                local_context: ContextT = {"file_info": file_info, **context}
                yield from data_reader(file_handle, local_context)


def scan_csv_file(location: Location, context: ContextT) -> Iterator[pl.LazyFrame]:
    sensor = location.name
    sensor = sensor.rstrip(".csv") if sensor else None
    print(f"Scanning CSV file: {location}")
    yield pl.scan_csv(location).with_columns(
        pl.lit(sensor).alias("sensor"),
        pl.lit(context["root"].name).alias("measurement"),
    )


def is_csv_file(location: Location, context: ContextT) -> bool:
    return location.name.endswith(".csv") and ("meta" not in location.name)


def scan_directory(
    location: Location,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    # Iterate through all files in the directory
    assert isinstance(location, Path), "Location must be a Path object"
    context = {"root": location}
    for file_path in location.rglob("*"):
        if file_predicate(file_path, context):
            with open(file_path, "rb") as file_handle:
                local_context = {"file_info": file_path, **context}
                yield from data_reader(file_handle, local_context)


def is_zip_file(location: Location, context: ContextT) -> bool:
    return location.name.endswith(".zip")


def zip_scanner(location: Location, context: ContextT) -> Iterator[pl.LazyFrame]:
    print(f"Scanning ZIP file: {location}")
    yield from scan_zip_dataset(
        location, data_reader=scan_csv_file, file_predicate=is_csv_file
    )


def scan_dataset(location: Location, context: ContextT) -> Iterator[pl.LazyFrame]:
    yield from scan_directory(
        location,
        data_reader=zip_scanner,
        file_predicate=is_zip_file,
    )


if __name__ == "__main__":
    container_path = Path("./data")
    index_columns = ["measurement", "sensor"]
    data = (
        (
            pl.concat(
                scan_dataset(container_path, {}), how="diagonal", rechunk=True
            ).sort("measurement", "Time (s)")
        )
        .group_by_dynamic(
            index_column=(pl.col("Time (s)") * 1e6).cast(pl.Datetime("us")),
            every="10ms",
            closed="both",
            group_by="measurement",
        )
        .agg(pl.selectors.numeric().mean())
    ).collect()
    print(data)
