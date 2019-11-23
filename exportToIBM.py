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
parser.add_argument('--cred_path',
                    default='/home/alex/Documents/MIDS/w251/creds/',
                    help='path to bucket credentials')
parser.add_argument('--image_dir',
                    default='/data/combined_data/',
                    help='path to images')
parser.add_argument('--delete_first',
                    action='store_true',
                    default=False,
                    help='whether or not to delete the bucket first')
args = parser.parse_args()

# %%
# cred_path = "/root/creds/w251-credentials.json"
cred_path = os.path.join(args.cred_path, 'w251-credentials.json')
with open(cred_path, "r") as f:
    creds = json.load(f)

auth_endpoint = 'https://iam.bluemix.net/oidc/token'
service_endpoint = 'https://s3.us-east.cloud-object-storage.appdomain.cloud'

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
# delete all objects in bucket
if args.delete_first:
    break_flag = False
    while not break_flag:
        files = list(bucket.objects.all())
        if len(files) < 1000:
            batch = files
            break_flag = True
        else:
            batch = files[:1000]
        objects = []
        for file in batch:
            objects.append({'Key': file.key})
        response = bucket.delete_objects(
            Delete={'Objects': objects, 'Quiet': False})
        print(response)
# %%
# get test images
keys_in_bucket = [f.key for f in bucket.objects.all()]
image_dir = args.image_dir
for dirname, subdirs, files in os.walk(image_dir):
    for file in tqdm(files):
        ext = file.split('.')[-1]
        if 'jpg' not in ext:
            continue
        filepath = os.path.join(dirname, file)
        key = '/'.join([dirname.split('/')[-1], file])
        if key in keys_in_bucket:
            print('file already in bucket')
            continue
        img = cv2.imread(filepath)
        _, img = cv2.imencode('.jpg', img)
        bucket.put_object(Key=key, Body=img.tobytes())
