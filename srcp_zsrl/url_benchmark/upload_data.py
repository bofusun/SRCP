from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/data/sjb/exorl-main/datasets",
    repo_id="bofusun/HILP_offline",
    repo_type="dataset",
)