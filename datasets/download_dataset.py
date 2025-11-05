from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # Patterns: BASE DFS RD MRP CHF
    snapshot_download(repo_id="hardware-fab/Chameleon", repo_type="dataset", local_dir=".", allow_patterns="DFS/*")