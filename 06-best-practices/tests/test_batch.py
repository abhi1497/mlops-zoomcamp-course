from datetime import datetime

import pandas as pd
from pandas.testing import assert_frame_equal

from batch import (
    get_input_path,
    get_output_path,
    get_s3_storage_options,
    prepare_data,
    save_data,
)


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
    ]
    df = pd.DataFrame(data, columns=columns)
    categorical = ['PULocationID', 'DOLocationID']

    actual = prepare_data(df, categorical)

    expected = pd.DataFrame(
        [
            ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
            ('1', '1', dt(1, 2), dt(1, 10), 8.0),
        ],
        columns=columns + ['duration'],
    )

    assert_frame_equal(actual, expected)


def test_get_input_path_uses_environment_pattern(monkeypatch):
    monkeypatch.setenv(
        'INPUT_FILE_PATTERN',
        's3://nyc-duration/in/{year:04d}-{month:02d}.parquet',
    )

    assert get_input_path(2023, 3) == 's3://nyc-duration/in/2023-03.parquet'


def test_get_output_path_uses_environment_pattern(monkeypatch):
    monkeypatch.setenv(
        'OUTPUT_FILE_PATTERN',
        's3://nyc-duration/out/{year:04d}-{month:02d}.parquet',
    )

    assert get_output_path(2023, 3) == 's3://nyc-duration/out/2023-03.parquet'


def test_get_s3_storage_options_uses_endpoint(monkeypatch):
    monkeypatch.setenv('S3_ENDPOINT_URL', 'http://localhost:4566')

    assert get_s3_storage_options() == {
        'client_kwargs': {'endpoint_url': 'http://localhost:4566'},
        'config_kwargs': {'response_checksum_validation': 'when_required'},
    }


def test_save_data_uses_s3_storage_options(monkeypatch):
    calls = {}

    def to_parquet(path, **kwargs):
        calls['path'] = path
        calls['kwargs'] = kwargs

    monkeypatch.setenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    df = pd.DataFrame({'ride_id': ['2023/01_0']})
    monkeypatch.setattr(df, 'to_parquet', to_parquet)

    save_data(df, 's3://nyc-duration/out/2023-01.parquet')

    assert calls == {
        'path': 's3://nyc-duration/out/2023-01.parquet',
        'kwargs': {
            'engine': 'pyarrow',
            'index': False,
            'storage_options': {
                'client_kwargs': {'endpoint_url': 'http://localhost:4566'},
                'config_kwargs': {'response_checksum_validation': 'when_required'},
            },
        },
    }
