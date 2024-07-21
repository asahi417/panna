"""https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces"""
import argparse
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description='Upload space to HuggingFace hub')
parser.add_argument('-t', '--token-hf', required=True, type=str)
parser.add_argument('-r', '--repo-id', required=True, type=str)
parser.add_argument('-s', '--src', required=True, type=str)
args = parser.parse_args()

api = HfApi()
api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
api.add_space_secret(repo_id=args.repo_id, key="HF_TOKEN", value=args.token_hf)
api.upload_folder(repo_id=args.repo_id, repo_type="space", folder_path=args.src)
