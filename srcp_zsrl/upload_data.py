# # from huggingface_hub import HfApi
# # import os 

# # api = HfApi(token=os.getenv("HF_TOKEN"))
# # api.upload_folder(
# #     folder_path="/data/sjb/exorl-main/datasets_new",
# #     repo_id="bofusun/HILP_offline",
# #     repo_type="dataset",
# # )

# from huggingface_hub import HfApi
# import os 
# from tqdm import tqdm

# api = HfApi(token=os.getenv("HF_TOKEN"))

# # 获取文件夹中所有文件
# folder_path = "/data/sjb/exorl-main/datasets"
# all_files = []
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         all_files.append(os.path.join(root, file))

# # 上传文件并显示进度
# with tqdm(total=len(all_files), desc="Uploading files") as pbar:
#     for file in all_files:
#         relative_path = os.path.relpath(file, folder_path)
#         api.upload_file(
#             path_or_fileobj=file,
#             path_in_repo=relative_path,
#             repo_id="bofusun/HILP_offline",
#             repo_type="dataset",
#         )
#         pbar.update(1)
                
from huggingface_hub import HfApi
from huggingface_hub import login
login() 
api = HfApi()
api.upload_file(
    path_or_fileobj="/data/sjb/exorl-main/datasets/walker/aps/replay_pixel64.pt",
    path_in_repo="/datasets/walker/aps/replay_pixel64.pt",
    repo_id="bofusun/HILP_offline1",
    repo_type="dataset",
)
        
# huggingface-cli upload bofusun/HILP_offline1 /data/sjb/exorl-main/datasets/walker/aps/replay_pixel64.pt walker/aps/replay_pixel64.pt
huggingface-cli upload bofusun/HILP_offline1 \
    /data/sjb/exorl-main/datasets/walker/aps/replay_pixel64.pt \
    "walker/aps/replay_pixel64.pt" \
    --token="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" \
    --repo-type=dataset
        
"""
from huggingface_hub import HfApi
api = HfApi()

"""