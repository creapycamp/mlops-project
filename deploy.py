import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ["HF_TOKEN"]

files_to_upload = ["app.py", "model.pkl"]

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="R00man/imdb-sentiment",
        repo_type="space",
        token=token
    )
    print(f"Uploaded {file}")

print("Deployed successfully to Hugging Face!")