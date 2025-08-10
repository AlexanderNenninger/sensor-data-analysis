import os
from dotenv import load_dotenv
from cloudpathlib import S3Client, S3Path
import pytest
from sensor_data_analysis.pipeline_components import (
    scan_directory,
    Location,
    is_zip_file,
    scan_zip_dataset,
)
from typing import Iterator
import ray
from sensor_data_analysis.utils import init_ray, get_pip_dependencies


PYPI_DEPS = [
    "boto3>=1.40.6",
    "cloudpathlib>=0.21.1",
    "matplotlib>=3.10.5",
    "polars>=1.31.0",
    "pytest>=8.4.1",
    "python-dotenv>=1.1.1",
    "ray[client]>=2.48.0",
]

load_dotenv()

STORAGE_OPTIONS = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "aws_region": os.getenv("AWS_REGION"),
    "aws_endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
}

BUCKET_NAME = "measurement-data"
S3_BASE_URI = f"s3://{BUCKET_NAME}/"
RAY_CLUSTER_ADDRESS = os.getenv("RAY_CLUSTER_ADDRESS")


@pytest.fixture()
def s3_client():
    client = S3Client(
        endpoint_url=STORAGE_OPTIONS["aws_endpoint_url"],
        aws_access_key_id=STORAGE_OPTIONS["aws_access_key_id"],
        aws_secret_access_key=STORAGE_OPTIONS["aws_secret_access_key"],
    )
    return client


def test_cloudpathlib(s3_client: S3Client):
    # Use the s3_client fixture to interact with your S3 bucket
    base_path = S3Path(S3_BASE_URI, client=s3_client)
    assert base_path.exists()


def test_scan_dataset(s3_client: S3Client):
    # Use the s3_client fixture to interact with your S3 bucket
    base_path = S3Path(S3_BASE_URI, client=s3_client)

    def get_file_name(loc: Location) -> Iterator[str]:
        yield loc.name

    def zip_reader(loc: Location):
        return scan_zip_dataset(
            loc, data_reader=get_file_name, file_predicate=lambda loc: True
        )

    files = list(
        scan_directory(base_path, data_reader=zip_reader, file_predicate=is_zip_file)
    )
    assert len(files) > 0


@pytest.fixture(scope="session", autouse=True)
def with_ray():
    # Ensure remote workers install from PyPI deps only, not local editable paths
    runtime_env = {"pip": PYPI_DEPS}
    with init_ray(address=RAY_CLUSTER_ADDRESS, runtime_env=runtime_env):
        yield


def test_get_pip_dependencies():
    deps = get_pip_dependencies()
    assert isinstance(deps, list)


def test_run_ray_task():
    import socket

    @ray.remote
    def where_am_i():
        return socket.gethostname()

    result = ray.get(where_am_i.remote())
    assert result != socket.gethostname(), "Ray task should run on a different node"


@pytest.mark.skipif(
    bool(os.getenv("RAY_CLUSTER_ADDRESS")),
    reason="Local package is not distributed to Ray workers without working_dir; skip on remote",
)
def test_ray_scan_dataset(s3_client: S3Client):
    base_path = S3Path(S3_BASE_URI, client=s3_client)

    def get_file_name(loc: Location) -> Iterator[str]:
        yield loc.name

    def zip_reader(loc: Location) -> Iterator[str]:
        return scan_zip_dataset(
            loc, data_reader=get_file_name, file_predicate=lambda loc: True
        )

    files = list(
        scan_directory(
            base_path,
            data_reader=zip_reader,
            file_predicate=is_zip_file,
            worker_cfg={"num_cpus": 2, "num_gpus": 0},
        )
    )
    assert len(files) > 0
