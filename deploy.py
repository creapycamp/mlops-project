import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ.get("HF_TOKEN")

repo_id = "R00man/imdb-sentiment"

# Upload app.py
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="space",
    token=token
)
print("Uploaded app.py")

# Upload model.pkl
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=repo_id,
    repo_type="space",
    token=token
)
print("Uploaded model.pkl")

# Upload requirements for the Space
space_requirements = "scikit-learn\ngradio\n"
with open("space_requirements.txt", "w") as f:
    f.write(space_requirements)

api.upload_file(
    path_or_fileobj="space_requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=repo_id,
    repo_type="space",
    token=token
)
print("Uploaded requirements.txt")

print("Deployed successfully!")