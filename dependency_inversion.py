from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Callable, Iterator, TypeVar
from zipfile import ZipFile, ZipInfo

# 'primitive' types we might abstract later.
ContextT = dict[str, Any]
FileInfoT = ZipInfo
DataItemT = TypeVar("DataItemT", covariant=True)
Location = Path | IO | str

# filtering files
FilePredicateT = Callable[[FileInfoT, ContextT], bool]

# reading files
DataReaderT = Callable[[Location, ContextT], DataItemT]


def scan_zip_dataset(
    zip_location: Location, data_reader: DataReaderT, file_predicate: FilePredicateT
) -> Iterator[DataItemT]:
    # Open the zip file
    with ZipFile(zip_location, "r") as zip_ref:
        context = {"root": zip_location, "zip_ref": zip_ref}
        # Find files that match the predicate
        data_files = (f for f in zip_ref.filelist if file_predicate(f, context))
        for file_info in data_files:
            # Open each file and read its data
            with zip_ref.open(file_info) as file_handle:
                local_context = {"file_info": file_info, **context}
                yield data_reader(file_handle, local_context)
