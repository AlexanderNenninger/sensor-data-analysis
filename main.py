from pathlib import Path

import polars as pl

from sensor_data_analysis.pipeline_components import scan_dataset


def main():
    container_path = Path("./data")
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
    )
    print(data)


if __name__ == "__main__":
    main()
