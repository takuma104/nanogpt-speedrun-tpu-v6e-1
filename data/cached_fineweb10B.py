# from KellerJordan/modded-nanogpt: GPT-2 tokens of FineWeb10B (saves ~1 hour vs re-tokenizing)
import os
import sys

from huggingface_hub import hf_hub_download


def get(fname: str) -> None:
    local_dir = os.path.join(os.path.dirname(__file__), "fineweb10B")
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname, repo_type="dataset", local_dir=local_dir)


def main() -> None:
    get("fineweb_val_%06d.bin" % 0)
    num_chunks = int(sys.argv[1]) if len(sys.argv) >= 2 else 103  # each chunk is 100M tokens
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)


if __name__ == "__main__":
    main()
