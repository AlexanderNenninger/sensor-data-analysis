import zipfile
import polars as pl
import pytest
from pathlib import Path
import tempfile
from typing import Iterator

import sensor_data_analysis.pipeline_components as pc
import ray


@pytest.fixture
def tmp_file() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def csv_file(tmp_file: Path) -> Path:
    file = tmp_file / "sensor1.csv"
    file.write_text("a,b\n1,2\n3,4\n")
    return file


@pytest.fixture
def meta_csv_file(tmp_file: Path) -> Path:
    file = tmp_file / "meta_sensor1.csv"
    file.write_text("a,b\n5,6\n")
    return file


@pytest.fixture
def zip_file_with_csv(tmp_file: Path, csv_file: Path, meta_csv_file: Path) -> Path:
    zip_path = tmp_file / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_file, arcname=csv_file.name)
        zf.write(meta_csv_file, arcname=meta_csv_file.name)
    return zip_path


def test_is_measurement(csv_file: Path, meta_csv_file: Path):
    assert pc.is_measurement(csv_file)
    assert not pc.is_measurement(meta_csv_file)


def test_is_zip_file(zip_file_with_csv: Path, csv_file: Path):
    assert pc.is_zip_file(zip_file_with_csv)
    assert not pc.is_zip_file(csv_file)


def test_scan_csv_file(csv_file: Path):
    dfs = list(pc.scan_csv_file(csv_file))
    assert len(dfs) == 1
    df = dfs[0]
    assert isinstance(df, pl.DataFrame)
    assert "sensor" in df.columns
    assert df["sensor"][0] == "sensor1"


def test_scan_zip_dataset(zip_file_with_csv: Path):
    def dummy_reader(loc: pc.Location):
        # Should only be called for sensor1.csv
        return pc.scan_csv_file(loc)

    def pred(loc: pc.Location):
        return pc.is_measurement(loc)

    dfs = list(pc.scan_zip_dataset(zip_file_with_csv, dummy_reader, pred))
    assert len(dfs) == 1
    assert isinstance(dfs[0], pl.DataFrame)
    assert dfs[0]["sensor"][0] == "sensor1"


def test_zip_scanner(zip_file_with_csv: Path):
    dfs = list(pc.zip_scanner(zip_file_with_csv))
    assert len(dfs) == 1
    assert isinstance(dfs[0], pl.DataFrame)
    assert dfs[0]["sensor"][0] == "sensor1"


def test_scan_directory(tmp_file: Path, zip_file_with_csv: Path):
    # Place the zip file in a directory and scan it
    dfs = list(
        pc.scan_directory_ray(
            tmp_file,
            data_reader=pc.zip_scanner,
            file_predicate=pc.is_zip_file,
        )
    )
    assert len(dfs) == 1
    assert isinstance(dfs[0], pl.DataFrame)
    assert dfs[0]["sensor"][0] == "sensor1"


def test_scan_directory_multiple_dirs_ray(tmp_path: Path):
    import ray

    ray.init()  # type: ignore
    # Create two subdirectories with .csv files
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    file1 = dir1 / "sensorA.csv"
    file2 = dir2 / "sensorB.csv"
    file1.write_text("a,b\n1,2\n3,4\n")
    file2.write_text("a,b\n5,6\n7,8\n")

    def file_predicate(loc: pc.Location) -> bool:
        return str(loc).endswith(".csv")

    dfs = list(
        pc.scan_directory(
            tmp_path,
            data_reader=pc.scan_csv_file,
            file_predicate=file_predicate,
            worker_cfg={"num_cpus": 1, "memory": 1024, "timeout": 10},
        )
    )
    sensors = sorted(df["sensor"][0] for df in dfs)
    assert sensors == ["sensorA", "sensorB"]
    ray.shutdown()  # type: ignore


def test_scan_directory_ray_with_worker_crash(tmp_path: Path):
    ray.init(ignore_reinit_error=True)  # type: ignore
    # Create files, one of which will cause a crash
    file1 = tmp_path / "sensorA.csv"
    file2 = tmp_path / "sensorB.csv"
    file3 = tmp_path / "sensorC.csv"
    file1.write_text("a,b\n1,2\n3,4\n")
    file2.write_text("a,b\n5,6\n7,8\n")
    file3.write_text("a,b\n9,10\n11,12\n")

    # Custom data_reader that crashes on a specific file
    def crashing_reader(loc: pc.Location):
        if "sensorB" in str(loc):
            # OOM Error
            _data = bytearray(10**20)  # Simulate large data
        return pc.scan_csv_file(loc)

    def file_predicate(loc: pc.Location) -> bool:
        return str(loc).endswith(".csv")

    # scan_directory_ray should propagate the error
    res = list(
        pc.scan_directory_ray(
            tmp_path,
            data_reader=crashing_reader,
            file_predicate=file_predicate,
            worker_cfg={
                "num_cpus": 1,
                "memory": 1024,
                "timeout": 10,
                "ignore_errors": True,
            },
        )
    )
    assert len(res) == 2  # sensorA and sensorC should be processed
    ray.shutdown()  # type: ignore


def test_scan_dataset(tmp_file: Path, zip_file_with_csv: Path):
    dfs = list(pc.scan_dataset(tmp_file))
    assert len(dfs) == 1
    assert isinstance(dfs[0], pl.DataFrame)
    assert dfs[0]["sensor"][0] == "sensor1"
