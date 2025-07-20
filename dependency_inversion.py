from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Callable, Iterator, TypeVar
from zipfile import ZipFile, ZipInfo

import polars as pl

# 'primitive' types we might abstract later.
ContextT = dict[str, Any]
FileInfoT = ZipInfo | IO | Path
DataItemT = TypeVar("DataItemT", covariant=True)
Location = IO[bytes] | Path

# filtering files
FilePredicateT = Callable[[FileInfoT, ContextT], bool]

# reading files
DataReaderT = Callable[[Location, ContextT], Iterator[DataItemT]]


def scan_zip_dataset(
    zip_location: Location,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    # Open the zip file
    with ZipFile(zip_location, "r") as zip_ref:
        context = {"root": zip_location, "file": zip_ref.filename}
        # Find files that match the predicate
        data_files = (f for f in zip_ref.filelist if file_predicate(f, context))
        for file_info in data_files:
            # Open each file and read its data
            with zip_ref.open(file_info) as file_handle:
                local_context = {"file_info": file_info, **context}
                yield from data_reader(file_handle, local_context)


DefaultT = TypeVar("DefaultT")


def get_name(finfo: FileInfoT, default: DefaultT = None) -> str | DefaultT:
    return getattr(finfo, "filename", getattr(finfo, "name", default))


def csv_scanner(location: Location, context) -> Iterator[pl.LazyFrame]:
    sensor = get_name(context.get("file_info"))
    sensor = sensor.rstrip(".csv") if sensor else None
    yield pl.scan_csv(location).with_columns(
        pl.lit(sensor).alias("sensor"),
        pl.lit(get_name(context["root"])).alias("measurement"),
    )


def csv_file_predicate(f, context) -> bool:
    name = get_name(f, "")
    return name.endswith(".csv") and ("meta" not in name)


def scan_container(
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


def zip_file_predicate(f, context) -> bool:
    return (get_name(f) or "").endswith(".zip")


def zip_scanner(location, context) -> Iterator[pl.LazyFrame]:
    yield from scan_zip_dataset(
        location, data_reader=csv_scanner, file_predicate=csv_file_predicate
    )


def scan_dataset(location: Location, context: ContextT = {}) -> Iterator[pl.LazyFrame]:
    yield from scan_container(
        location,
        data_reader=zip_scanner,
        file_predicate=zip_file_predicate,
    )


if __name__ == "__main__":
    container_path = Path("./data")
    index_columns = ["measurement", "sensor"]
    data = (
        (
            pl.concat(scan_dataset(container_path), how="diagonal", rechunk=True).sort(
                "measurement", "Time (s)"
            )
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
