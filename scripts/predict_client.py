"""Call a running API with image files.

    python scripts/predict_client.py demo_images/*.png          (PowerShell may need explicit names)
    python scripts/predict_client.py --url http://localhost:8000 demo_images/cat.png
"""

import argparse
import mimetypes
from pathlib import Path

import httpx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    with httpx.Client(timeout=30) as client:
        print("model info:", client.get(f"{args.url}/model-info").json())
        for path in map(Path, args.images):
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            r = client.post(f"{args.url}/predict", params={"top_k": args.top_k},
                            files={"file": (path.name, path.read_bytes(), ctype)})
            if r.status_code != 200:
                print(f"{path.name}: HTTP {r.status_code} {r.text}")
                continue
            body = r.json()
            print(f"{path.name:22s} -> {body['predicted_class']:11s} conf={body['confidence']:.3f} "
                  f"low_conf={body['low_confidence']} ({body['inference_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
