# Dependency Inversion Pattern for IMU Data Processing

A flexible framework for processing IMU sensor data from ZIP archives using the dependency inversion principle.

## Features

- Processes ZIP files containing CSV sensor data
- Type-safe, extensible architecture
- Time-series data aggregation with Polars

## Usage

```bash
pip install polars
python dependency_inversion.py
```

The script scans `./data` for ZIP files, extracts CSV sensor data, and aggregates it by 10ms time windows.

## Data Structure

```text
data/
├── test/
│   └── IMU Data 2025-07-19 09-57-36.zip
└── train/
    └── IMU Data 2025-07-19 09-55-48.zip
```

## License

MIT
