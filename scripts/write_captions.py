#!/usr/bin/env python3
"""Write captions.vtt from yt-dlp auto-subs for a video.

Usage:
  python3 write_captions.py --video-id kv_3S6zyfxE --out ./out
"""
import argparse
import os
import subprocess
import sys

YT_DLP_JS_RUNTIME_ARGS = ["--js-runtimes", "deno"]
YT_DLP_CLIENT_ARGS = ["--extractor-args", "youtube:player_client=web"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={args.video_id}"
    cmd = [
        "yt-dlp", "--skip-download", "--write-auto-subs",
        "--sub-langs", "en", "--sub-format", "vtt",
        *YT_DLP_JS_RUNTIME_ARGS, *YT_DLP_CLIENT_ARGS,
        "-o", os.path.join(args.out, "%(id)s.%(ext)s"), url,
    ]
    result = subprocess.run(cmd, check=False)
    src = os.path.join(args.out, f"{args.video_id}.en.vtt")
    dst = os.path.join(args.out, "captions.vtt")
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"Wrote {dst}")
    else:
        print(f"No captions produced (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
