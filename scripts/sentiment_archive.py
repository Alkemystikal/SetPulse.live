#!/usr/bin/env python3
"""
SetPulse sentiment archiving pipeline.

Meant to run unattended on a schedule (GitHub Actions), with no manual
terminal step from the producer. For a given show, it:

  1. Finds the video (auto-detects the latest completed livestream on the
     channel, or takes an explicit --video-id for testing/backfill).
  2. Downloads the live chat replay via yt-dlp.
  3. Fetches real performer on/off-stage windows from the SetPulse Apps
     Script `archive_windows` endpoint for the show's date (no PIN needed -
     see SetPulse_Producer_MC_v4.gs).
  4. Buckets every chat message into whichever comic's window it falls in
     (messages outside every window are bucketed as "between_sets").
  5. Scores each message with a small built-in lexicon sentiment scorer -
     deliberately NOT an external NLP dependency, so this can't be broken
     by a restrictive pip mirror on whatever host it ends up running on.
  6. Writes chat.csv (every message, its assigned comic, its score) and
     stats.json (per-comic aggregates + show-level totals) to --out.

A separate step (not in this script) commits those two files to the
Alkemystikal/SetPulse.live repo - see the sample GitHub Actions workflow
shipped alongside this file.

Usage:
  # Real run - auto-detect the latest completed stream on the channel:
  python3 sentiment_archive.py --auto \
      --channel-id UC_47Pv7ji6eRoRggwyc25ZQ \
      --archive-base "https://script.google.com/macros/s/AKfycbz.../exec" \
      --out ./out

  # Real run - a specific past video, for a backfill/test:
  python3 sentiment_archive.py --video-id dQw4w9WgXcQ --date 2026-09-14 \
      --archive-base "https://script.google.com/macros/s/AKfycbz.../exec" \
      --out ./out

  # Dry run with bundled sample fixtures (no network, no yt-dlp) - proves
  # out the windowing + sentiment + output logic on its own:
  python3 sentiment_archive.py --demo --out ./out
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

# --------------------------------------------------------------------------
# Lightweight built-in sentiment scorer.
#
# Live comedy-show chat is mostly emotes, laughter, and short reactions, not
# well-formed sentences - a general NLP sentiment model is overkill and (as
# this sandbox demonstrated) pulling one in from an external index is a
# needless point of failure. This scores -1..+1 from small hand-tuned word/
# emote lists tailored to what this chat actually looks like.
# --------------------------------------------------------------------------
# Word-level tokens: matched against the whole lowercased, punctuation-
# stripped word only (never as a substring or a bare character), so a
# short slangy token here can't accidentally match inside an unrelated
# word.
POSITIVE_WORDS = {
    "lol", "lmao", "lmfao", "haha", "hahaha", "hilarious", "funny", "fire",
    "goated", "goat", "king", "queen", "killed", "killer", "amazing",
    "incredible", "love", "loved", "banger", "insane", "crushed", "legend",
    "based", "yesss", "yes", "poggers", "pog", "clip", "clipthat",
}
NEGATIVE_WORDS = {
    "boo", "trash", "mid", "cringe", "bomb", "bombed", "yikes", "bad",
    "worst", "weak", "booo", "lame", "awful", "terrible", "sucked",
}
# Emoji tokens: matched only against actual emoji characters found in the
# raw (un-stripped) token, never against plain letters.
POSITIVE_EMOJI = {
    "\U0001F602", "\U0001F923", "\U0001F525", "\U00002764", "\U0001F44F",
    "\U0001F44D", "\U0001F4AF", "\U0001F929", "\U0001F60D", "\U0001F389",
}
NEGATIVE_EMOJI = {
    "\U0001F44E", "\U0001F4A4", "\U0001F480", "\U0001F62C",
}
NEGATION_TOKENS = {"not", "no", "never", "isnt", "isn't", "wasnt", "wasn't"}

# Slang gets stretched a lot in live chat ("lmaooo", "hahahaha", "nooo") -
# exact word matches above miss those, so anything starting with one of
# these roots also counts, on top of the exact-match lists.
POSITIVE_PREFIXES = ("lmao", "lmfao", "haha", "lol", "goat", "fire", "king", "queen")
NEGATIVE_PREFIXES = ("bomb", "yikes", "cring", "trash")


def _emoji_hits(raw, emoji_set):
    return sum(1 for ch in raw if ch in emoji_set)


def score_sentiment(text):
    """Return a compound-style score in [-1, 1] for one chat message."""
    if not text:
        return 0.0
    words = text.strip().split()
    pos = neg = 0
    negate_next = False
    for raw in words:
        w = raw.strip(".,!?:;\"'").lower()
        if not w:
            continue
        if w in NEGATION_TOKENS:
            negate_next = True
            continue
        is_pos = (w in POSITIVE_WORDS or w.startswith(POSITIVE_PREFIXES)
                  or _emoji_hits(raw, POSITIVE_EMOJI) > 0)
        is_neg = (w in NEGATIVE_WORDS or w.startswith(NEGATIVE_PREFIXES)
                  or _emoji_hits(raw, NEGATIVE_EMOJI) > 0)
        if is_pos and not is_neg:
            if negate_next:
                neg += 1
            else:
                pos += 1
        elif is_neg and not is_pos:
            if negate_next:
                pos += 1
            else:
                neg += 1
        negate_next = False
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


# --------------------------------------------------------------------------
# Step 1: find the video
# --------------------------------------------------------------------------
def find_latest_completed_video(channel_id, api_key):
    """Latest completed (not live, not upcoming) video on the channel."""
    q = urllib.parse.urlencode({
        "key": api_key, "channelId": channel_id, "part": "snippet",
        "order": "date", "type": "video", "maxResults": 10,
    })
    with urllib.request.urlopen(f"{YOUTUBE_API}/search?{q}") as r:
        data = json.load(r)
    for item in data.get("items", []):
        vid = item["id"]["videoId"]
        details = get_video_details(vid, api_key)
        if details and details.get("liveBroadcastContent") == "none" \
                and details.get("actualStartTime"):
            return vid, details
    raise SystemExit("No completed livestream found in the last 10 uploads.")


def get_video_details(video_id, api_key):
    q = urllib.parse.urlencode({
        "key": api_key, "id": video_id, "part": "liveStreamingDetails,snippet",
    })
    with urllib.request.urlopen(f"{YOUTUBE_API}/videos?{q}") as r:
        data = json.load(r)
    items = data.get("items", [])
    if not items:
        return None
    item = items[0]
    live = item.get("liveStreamingDetails", {})
    return {
        "title": item["snippet"]["title"],
        "publishedAt": item["snippet"]["publishedAt"],
        "actualStartTime": live.get("actualStartTime"),
        "actualEndTime": live.get("actualEndTime"),
        "liveBroadcastContent": item["snippet"].get("liveBroadcastContent", "none"),
    }


# --------------------------------------------------------------------------
# Step 2: pull the chat replay
# --------------------------------------------------------------------------
def download_live_chat(video_id, workdir):
    """
    Shells out to yt-dlp to grab the live-chat-replay track for a completed
    stream. yt-dlp writes it as <id>.live_chat.json (JSON Lines - one
    youtubei "action" per line, each carrying its own relative timestamp).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp", "--skip-download", "--write-subs",
        "--sub-langs", "live_chat", "--sub-format", "json",
        "-o", os.path.join(workdir, "%(id)s.%(ext)s"), url,
    ]
    subprocess.run(cmd, check=True)
    path = os.path.join(workdir, f"{video_id}.live_chat.json")
    if not os.path.exists(path):
        raise SystemExit(f"yt-dlp did not produce {path}")
    return path


def parse_live_chat(path):
    """
    Yields {offset_seconds, author, message} for each renderable chat
    message in a yt-dlp live_chat.json file.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                action = json.loads(line)
            except json.JSONDecodeError:
                continue
            replay = action.get("replayChatItemAction", {})
            offset_ms = replay.get("videoOffsetTimeMsec")
            actions = replay.get("actions", [])
            for a in actions:
                item = a.get("addChatItemAction", {}).get("item", {})
                renderer = item.get("liveChatTextMessageRenderer")
                if not renderer:
                    continue
                author = renderer.get("authorName", {}).get("simpleText", "")
                runs = renderer.get("message", {}).get("runs", [])
                text = "".join(r.get("text", "") for r in runs if "text" in r)
                if offset_ms is None:
                    continue
                out.append({
                    "offset_seconds": int(offset_ms) / 1000.0,
                    "author": author,
                    "message": text,
                })
    return out


# --------------------------------------------------------------------------
# Step 3: real performer windows
# --------------------------------------------------------------------------
def fetch_archive_windows(archive_base, date_str):
    q = urllib.parse.urlencode({"format": "archive_windows", "date": date_str})
    with urllib.request.urlopen(f"{archive_base}?{q}", timeout=20) as r:
        data = json.load(r)
    if not data.get("ok"):
        raise SystemExit(f"archive_windows returned an error: {data}")
    return data.get("windows", [])


def fetch_archive_windows_with_lag_search(archive_base, date_str, lookahead_days=3):
    """
    The real-world 'Archived At' date can lag the actual show date by a few
    days (finish_show is sometimes pressed well after the fact). Try the
    given date first, then a few days after it, and use the first date that
    actually has windows.
    """
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    for offset in range(0, lookahead_days + 1):
        d = (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")
        windows = fetch_archive_windows(archive_base, d)
        if windows:
            return d, windows
    return date_str, []


# --------------------------------------------------------------------------
# Step 4/5: bucket + score
# --------------------------------------------------------------------------
def build_report(chat_messages, windows, stream_start_iso):
    """
    chat_messages: [{offset_seconds, author, message}] - offsets relative
        to stream start.
    windows: [{name, on_stage_at (ISO), done_at (ISO)}] - absolute times.
    stream_start_iso: ISO8601 actualStartTime of the stream, used to convert
        chat offsets to absolute wall-clock time so they line up with the
        Sheet's on_stage_at/done_at timestamps.
    """
    stream_start = datetime.fromisoformat(stream_start_iso.replace("Z", "+00:00"))
    parsed_windows = []
    for w in windows:
        on = datetime.fromisoformat(w["on_stage_at"].replace("Z", "+00:00"))
        done = datetime.fromisoformat(w["done_at"].replace("Z", "+00:00"))
        parsed_windows.append({"name": w["name"], "on": on, "done": done})

    rows = []
    per_comic = {}
    for msg in chat_messages:
        abs_time = stream_start + timedelta(seconds=msg["offset_seconds"])
        comic = "between_sets"
        for w in parsed_windows:
            if w["on"] <= abs_time <= w["done"]:
                comic = w["name"]
                break
        score = score_sentiment(msg["message"])
        rows.append({
            "offset_seconds": round(msg["offset_seconds"], 1),
            "timestamp_utc": abs_time.isoformat(),
            "author": msg["author"],
            "message": msg["message"],
            "comic": comic,
            "sentiment": score,
        })
        bucket = per_comic.setdefault(comic, {"messages": 0, "sentiment_sum": 0.0,
                                               "positive": 0, "negative": 0, "neutral": 0})
        bucket["messages"] += 1
        bucket["sentiment_sum"] += score
        if score > 0.15:
            bucket["positive"] += 1
        elif score < -0.15:
            bucket["negative"] += 1
        else:
            bucket["neutral"] += 1

    stats = {"generated_at": datetime.now(timezone.utc).isoformat(), "comics": {}}
    for name, b in per_comic.items():
        avg = round(b["sentiment_sum"] / b["messages"], 4) if b["messages"] else 0.0
        stats["comics"][name] = {
            "messages": b["messages"],
            "avg_sentiment": avg,
            "positive": b["positive"],
            "negative": b["negative"],
            "neutral": b["neutral"],
        }
    stats["totals"] = {
        "messages": len(rows),
        "comics_with_chat": len([n for n in per_comic if n != "between_sets"]),
    }
    return rows, stats


def write_outputs(rows, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "chat.csv")
    json_path = os.path.join(out_dir, "stats.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "offset_seconds", "timestamp_utc", "author", "message", "comic", "sentiment",
        ])
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return csv_path, json_path


# --------------------------------------------------------------------------
# Demo fixtures - proves the windowing/sentiment/output logic without
# needing yt-dlp or network access (both restricted in some sandboxes).
# --------------------------------------------------------------------------
def load_demo_inputs(fixtures_dir):
    with open(os.path.join(fixtures_dir, "sample_live_chat.json")) as f:
        chat = json.load(f)
    with open(os.path.join(fixtures_dir, "sample_archive_windows.json")) as f:
        windows_payload = json.load(f)
    return chat, windows_payload["windows"], windows_payload["stream_start"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", action="store_true", help="Auto-detect latest completed stream on --channel-id")
    ap.add_argument("--channel-id", default=os.environ.get("SETPULSE_CHANNEL_ID", ""))
    ap.add_argument("--video-id", help="Explicit video ID (backfill/test)")
    ap.add_argument("--date", help="Show date YYYY-MM-DD (for --video-id runs, or overrides auto-detect date)")
    ap.add_argument("--archive-base", default=os.environ.get("SETPULSE_ARCHIVE_BASE", ""),
                     help="Apps Script /exec base URL")
    ap.add_argument("--youtube-api-key", default=os.environ.get("YOUTUBE_API_KEY", ""))
    ap.add_argument("--out", default="./out")
    ap.add_argument("--demo", action="store_true", help="Run entirely on bundled sample fixtures, no network/yt-dlp")
    args = ap.parse_args()

    if args.demo:
        fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        chat, windows, stream_start = load_demo_inputs(fixtures_dir)
        print(f"[demo] loaded {len(chat)} sample chat messages and {len(windows)} performer windows")
        rows, stats = build_report(chat, windows, stream_start)
        csv_path, json_path = write_outputs(rows, stats, args.out)
        print(f"[demo] wrote {csv_path}")
        print(f"[demo] wrote {json_path}")
        print(json.dumps(stats, indent=2))
        return

    workdir = os.path.join(args.out, "_work")
    os.makedirs(workdir, exist_ok=True)

    if args.auto:
        if not args.channel_id or not args.youtube_api_key:
            raise SystemExit("--auto requires --channel-id and --youtube-api-key")
        video_id, details = find_latest_completed_video(args.channel_id, args.youtube_api_key)
        stream_start_iso = details["actualStartTime"]
        show_date = args.date or stream_start_iso[:10]
        print(f"Auto-detected video {video_id}: {details['title']} ({stream_start_iso})")
    else:
        if not args.video_id:
            raise SystemExit("Provide --video-id, or use --auto, or use --demo")
        video_id = args.video_id
        if args.youtube_api_key:
            details = get_video_details(video_id, args.youtube_api_key)
            stream_start_iso = details["actualStartTime"]
        elif args.date:
            # No API key available - assume the stream started at local
            # midnight of --date; this only matters for lining chat offsets
            # up with archive_windows, so pass --youtube-api-key when possible.
            stream_start_iso = args.date + "T00:00:00Z"
        else:
            raise SystemExit("Provide --date or --youtube-api-key so the stream start time is known")
        show_date = args.date or stream_start_iso[:10]

    if not args.archive_base:
        raise SystemExit("--archive-base is required (the SetPulse Apps Script /exec URL)")

    resolved_date, windows = fetch_archive_windows_with_lag_search(args.archive_base, show_date)
    if not windows:
        raise SystemExit(f"No archive_windows found for {show_date} (or the next few days after it).")
    print(f"Using archive_windows for {resolved_date}: {len(windows)} performer(s)")

    chat_path = download_live_chat(video_id, workdir)
    chat_messages = parse_live_chat(chat_path)
    print(f"Parsed {len(chat_messages)} chat messages")

    rows, stats = build_report(chat_messages, windows, stream_start_iso)
    csv_path, json_path = write_outputs(rows, stats, args.out)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
