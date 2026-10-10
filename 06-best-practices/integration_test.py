import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


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
df_input = pd.DataFrame(data, columns=columns)

endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
input_file = os.getenv(
    'INPUT_FILE_PATTERN',
    's3://nyc-duration/in/{year:04d}-{month:02d}.parquet',
).format(year=2023, month=1)
output_file = os.getenv(
    'OUTPUT_FILE_PATTERN',
    's3://nyc-duration/out/{year:04d}-{month:02d}.parquet',
).format(year=2023, month=1)
os.environ['S3_ENDPOINT_URL'] = endpoint_url
os.environ['INPUT_FILE_PATTERN'] = input_file
os.environ['OUTPUT_FILE_PATTERN'] = output_file
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'test')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'test')
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
options = {
    'client_kwargs': {'endpoint_url': endpoint_url},
    'config_kwargs': {'response_checksum_validation': 'when_required'},
}

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

script_dir = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, 'batch.py', '2023', '1'],
    cwd=script_dir,
    env=os.environ.copy(),
    check=True,
)

df_result = pd.read_parquet(output_file, storage_options=options)
assert df_result['ride_id'].tolist() == ['2023/01_0', '2023/01_1']
assert len(df_result) == 2
print(df_result)
print('sum of predicted durations:', df_result['predicted_duration'].sum())
