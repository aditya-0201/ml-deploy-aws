# Title: Untitled 2026-03-23 18:51:03

# Set environment variables for sagemaker_studio imports

import os
os.environ['DataZoneProjectId'] = '5fpu8ad0qsbcox'
os.environ['DataZoneDomainId'] = 'dzd-bn7czrrcg3megx'
os.environ['DataZoneEnvironmentId'] = 'cwn4pi400pjog1'
os.environ['DataZoneDomainRegion'] = 'ap-south-1'

# create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "5fpu8ad0qsbcox",
                "DataZoneDomainId": "dzd-bn7czrrcg3megx",
                "DataZoneEnvironmentId": "cwn4pi400pjog1",
                "DataZoneDomainRegion": "ap-south-1",
            }
        }
    return _resource_metadata
metadata = _get_resource_metadata()

"""
Logging Configuration

Purpose:
--------
This sets up the logging framework for code executed in the user namespace.
"""

from typing import Optional


def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    # fallback to /tmp dir on access, helpful for local dev setup
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Rotating file handler
    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")


_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

def _reset_os_path():
    """
    Reset the process's working directory to handle mount timing issues.
    
    This function resolves a race condition where the Python process starts
    before the filesystem mount is complete, causing the process to reference
    old mount paths and inodes. By explicitly changing to the mounted directory
    (/home/sagemaker-user), we ensure the process uses the correct, up-to-date
    mount point.
    
    The function logs stat information (device ID and inode) before and after
    the directory change to verify that the working directory is properly
    updated to reference the new mount.
    
    Note:
        This is executed at module import time to ensure the fix is applied
        as early as possible in the kernel initialization process.
    """
    try:
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)

        os.chdir("/home/sagemaker-user")

        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

import pandas as pd
import numpy as np
import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput

np.random.seed(42)
n = 200

df = pd.DataFrame({
    'sqft':      np.random.randint(800, 5000, n),
    'bedrooms':  np.random.randint(1, 6, n),
    'bathrooms': np.random.randint(1, 4, n),
    'age':       np.random.randint(1, 50, n),
})

df['price'] = (
    df['sqft'] * 150 +
    df['bedrooms'] * 10000 +
    df['bathrooms'] * 8000 -
    df['age'] * 500 +
    np.random.randint(-20000, 20000, n)
)

# price column MUST be first for SageMaker Linear Learner
df = df[['price', 'sqft', 'bedrooms', 'bathrooms', 'age']]
df.to_csv('train.csv', index=False, header=False)

print(f"Dataset shape: {df.shape}")
print(df.head())
print("train.csv created!")

df = pd.read_csv('train.csv', header=None)

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

# 3. Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# 4. Remove outliers on price column (col 0)
Q1 = df[0].quantile(0.25)
Q3 = df[0].quantile(0.75)
IQR = Q3 - Q1
df = df[(df[0] >= Q1 - 1.5*IQR) & (df[0] <= Q3 + 1.5*IQR)]

# 5. Reset index
df.reset_index(drop=True, inplace=True)

# 6. Save cleaned file
df.to_csv('train.csv', index=False, header=False)

print(f"Shape after cleaning: {df.shape}")
print("Cleaned train.csv saved!")

s3 = boto3.client('s3')

s3.upload_file(
    Filename='train.csv',
    Bucket='sagemaker-aditya006',
    Key='data/train.csv'
)

print("Upload successful!")
print("S3 path: s3://sagemaker-aditya006/data/train.csv")

session = sagemaker.Session()
region = boto3.Session().region_name
role = sagemaker.get_execution_role()

bucket = "sagemaker-aditya006"
prefix = "house-price-prediction"

print(f"Region: {region}")
print(f"Role: {role}")
print(f"Bucket: {bucket}")

s3_train_path = f's3://{bucket}/data/train.csv'
container = image_uris.retrieve('linear-learner', region)

print(f"Training data: {s3_train_path}")
print(f"Container URI: {container}")

ll = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=session
)
print("Estimator created.")

ll.set_hyperparameters(
    feature_dim=4,
    predictor_type='regressor',
    mini_batch_size=20
)
print("Hyperparameters set.")

train_input = TrainingInput(
    s3_data=s3_train_path,
    content_type='text/csv'
)
print(f"Training input configured: {s3_train_path}")

print("Starting training... (takes 3-5 mins)")
ll.fit({'train': train_input})
print("Training complete!")

print("Deploying endpoint... (takes 3-5 mins)")
predictor = ll.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=sagemaker.serializers.CSVSerializer()
)
print(f"Endpoint deployed: {predictor.endpoint_name}")

import boto3
import json

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name=region)

# Input: [sqft, bedrooms, bathrooms, age]
test_data = [[2500, 3, 2, 15]]

# Convert to CSV string
payload = '\n'.join([','.join(map(str, row)) for row in test_data])
print(f"Payload sent: {payload}")

# Call endpoint via API
response = runtime.invoke_endpoint(
    EndpointName=predictor.endpoint_name,
    ContentType='text/csv',
    Body=payload
)

# Read and parse result
result = response['Body'].read().decode('utf-8')
result_json = json.loads(result)

print("Raw response:", result)
print("Predicted price: $", result_json['predictions'][0]['score'])