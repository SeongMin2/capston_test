import os
from google.cloud import storage
from ABS_PATH import ABS_PATH

KEY_PATH = ABS_PATH + "/data/mlops-348504-1d12c4fc9b7d.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

client = storage.Client(project="mlops-348504")
bucket = client.bucket("sm_mlops_data")
