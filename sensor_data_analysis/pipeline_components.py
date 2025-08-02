from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Callable, Iterator, TypeVar
from zipfile import ZipFile

import polars as pl
import ray

# 'primitive' types we might abstract later.
DataItemT = TypeVar("DataItemT", covariant=True)

SendableLocation = Path
Location = SendableLocation | IO[bytes]  # Allow both Path and file-like objects


class Context(dict[str, Any]):
    """Context for the pipeline, a hierarchical structure of Mappings."""

    def __init__(self, name: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.children: dict[str | None, Context] = {}
        self._parent = None
        self.name = name

    def with_context(self, context: dict[str, Any]) -> Context:
        """Create a new context with additional items."""
        new_context = Context(self.name, **self)
        new_context.update(context)
        return new_context

    def with_child(self, child: dict[str, Any]) -> Context:
        child = Context(**child)
        child._parent = self
        new_context = Context(self.name, **self)
        new_context.children[child.name] = child
        return new_context

    def pop_child(self, child: Context) -> Context | None:
        child._parent = None
        return self.children.pop(child.name, None)

    def parent(self) -> Context:
        """Return the parent context."""
        if self._parent is None:
            ctx = Context()
            ctx.children = {self.name: self}
            return ctx
        return self._parent


# filtering files
FilePredicateT = Callable[[Location, Context], bool]

# reading files
DataReaderT = Callable[[Location, Context], Iterator[DataItemT]]


def scan_zip_dataset(
    location: Location,
    context: Context,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    local_context = Context("zip_scanner", filename=location.name)
    context = context.with_child(local_context) if context else local_context
    with ZipFile(location, "r") as zip_ref:
        # Find files that match the predicate
        for file_info in zip_ref.filelist:
            # Open each file and read its data
            with zip_ref.open(file_info) as file_handle:
                if not file_predicate(file_handle, context):
                    continue
                yield from data_reader(file_handle, context)


def scan_directory(
    location: SendableLocation,
    context: Context,
    data_reader: DataReaderT[DataItemT],
    file_predicate: FilePredicateT,
) -> Iterator[DataItemT]:
    # Iterate through all files in the directory
    assert isinstance(location, Path), "Location must be a Path object"
    context = context.with_child({"name": location.name})
    file_paths = (
        file_path
        for file_path in location.glob("**/*")
        if file_predicate(file_path, context)
    )

    def task_wrapper(file_path: Location, context: Context) -> list[DataItemT]:
        return list(data_reader(file_path, context))

    tasks = [
        ray.remote(task_wrapper).remote(file_path, context) for file_path in file_paths
    ]
    for task in ray.get(tasks):
        yield from task


def scan_csv_file(location: Location, context: Context) -> Iterator[pl.DataFrame]:
    sensor = location.name
    sensor = sensor.rstrip(".csv") if sensor else None
    name = context.get("filename", location)
    yield pl.scan_csv(location).with_columns(
        pl.lit(sensor).alias("sensor"),
        pl.lit(name).alias("measurement"),
    ).collect()


def is_measurement(location: Location, context: Context) -> bool:
    return location.name.endswith(".csv") and ("meta" not in location.name)


def is_zip_file(location: Location, context: Context) -> bool:
    return location.name.endswith(".zip")


def zip_scanner(location: Location, context: Context) -> Iterator[pl.DataFrame]:
    yield from scan_zip_dataset(
        location, context, data_reader=scan_csv_file, file_predicate=is_measurement
    )


def scan_dataset(
    location: SendableLocation, context: Context = Context()
) -> Iterator[pl.DataFrame]:
    yield from scan_directory(
        location,
        context,
        data_reader=zip_scanner,
        file_predicate=is_zip_file,
    )
