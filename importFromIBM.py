# %% markdown
# Importing video data from ibm COS

# %%
import json
import os
import pickle
import random
import sys
import time
import argparse

import cv2
import ibm_boto3
import numpy as np
import pandas as pd
from ibm_botocore.client import Config
from tqdm import tqdm

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to save vids at')
args = parser.parse_args()

# %%
cred_path = "/root/creds/w251-credentials.json"
with open(cred_path, "r") as f:
    creds = json.load(f)

auth_endpoint = 'https://iam.bluemix.net/oidc/token'
service_endpoint = 'https://s3.us.cloud-object-storage.appdomain.cloud'

# Store relevant details for interacting with IBM COS store and uploading data
cos = ibm_boto3.resource('s3',
                         ibm_api_key_id=creds['apikey'],
                         ibm_service_instance_id=creds['resource_instance_id'],
                         ibm_auth_endpoint=auth_endpoint,
                         config=Config(signature_version='oauth'),
                         endpoint_url=service_endpoint)
# %%
# FIXME: need to figure out how to install aspera for massive download speed
# boost. Surprisingly, the docs suck
#
# from ibm_s3transfer.aspera.manager import AsperaTransferManager
# cos = ibm_boto3.client('s3',
#                          ibm_api_key_id=creds['apikey'],
#                          ibm_service_instance_id=creds['resource_instance_id'],
#                          ibm_auth_endpoint=auth_endpoint,
#                          config=Config(signature_version='oauth'),
#                          endpoint_url=service_endpoint)
# transfer_manager = AsperaTransferManager(cos)
# %%
bucket = cos.Bucket('w251-fp-bucket')
# %%
files = list(bucket.objects.all())
# %%
savepath = args.path
os.makedirs(savepath, exist_ok=True)
filelist = os.listdir(savepath)
for file in tqdm(files):
    ext = file.key.split('.')[-1]
    if 'mp4' not in ext and 'avi' not in ext:
        continue
    filename = file.key.split('/')[-1]
    if filename in filelist:
        continue
    else:
        filename = os.path.join(savepath, filename)
        bucket.download_file(file.key, filename)
