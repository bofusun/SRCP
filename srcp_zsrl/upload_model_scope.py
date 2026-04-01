from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '3d351fe8-3ebd-4cf3-872c-705193c41abc'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

api.upload_file(
    repo_id=f"sunbofu1/HILP_offline_data",
    path_or_fileobj='/data/sjb/exorl-main/datasets/walker/aps/replay_pixel64.pt',
    path_in_repo='walker/aps/replay_pixel64.pt',
    commit_message='upload walker rnd',
)