#!/usr/bin/env python3
"""
SetPulse sentiment archiving pipeline.

Meant to run unattended on a schedule (GitHub Actions), with no manual
terminal step from the producer. For a given show, it:

  1. Finds the video (selects the latest completed livestream on the
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
import bisect
import csv
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

# YouTube now often requires solving a small JS challenge before it'll
# serve a video's data (yt-dlp's own warning: "No supported JavaScript
# runtime could be found... YouTube extraction without a JS runtime has
# been deprecated"). The GitHub Actions workflow installs deno and runs
# `pip install -U yt-dlp` specifically so this flag is both available and
# actually used. If you're running this outside that workflow (e.g. a
# local test) on an older yt-dlp, either upgrade yt-dlp or clear this list.
YT_DLP_JS_RUNTIME_ARGS = ["--js-runtimes", "deno"]

# By default yt-dlp tries several "client" variants (web, some mobile/TV/
# app-internal ones) and picks whichever answers first. Some of those
# variants (e.g. "visionos") aren't ones a real signed-in browser session
# ever presents as, so cookies exported from an actual browser don't work
# for them and they get blocked by YouTube's bot check even with a valid
# cookies file. Restricting to "web" - the ordinary youtube.com website,
# not a different device or app - matches what the exported cookies
# actually are, so they get used instead of being wasted on a client they
# were never valid for.
YT_DLP_CLIENT_ARGS = ["--extractor-args", "youtube:player_client=web"]

# On top of the JS challenge, YouTube also wants proof the request is
# coming from a signed-in browser session ("Sign in to confirm you're not
# a bot"), which the JS runtime alone doesn't satisfy. The fix is a cookies
# file from a real signed-in YouTube session (the workflow writes the
# YTDLP_COOKIES secret to disk and points this at it).
#
# That cookie export can come from any browser/device - including an iOS
# Safari cookie-export app, which typically exports JSON, not the
# Netscape-format cookies.txt yt-dlp expects. Rather than ask whoever's
# exporting to get the format exactly right, _normalize_cookies_file below
# accepts either shape and converts JSON to Netscape format automatically.
YT_DLP_COOKIES_FILE = os.environ.get("YT_DLP_COOKIES_FILE", "")

# Cookies alone haven't been reliably enough - YouTube's bot check also
# weighs the requesting IP, and GitHub Actions runners share a small pool of
# well-known datacenter IPs that are common bot-check targets regardless of
# how valid the cookies are. Routing the yt-dlp request through a residential
# proxy (an ordinary home-internet IP, rented from a proxy provider) instead
# of the runner's own IP addresses that specific problem. Optional: if
# YT_DLP_PROXY isn't set, yt-dlp just uses the runner's IP directly as
# before, so this has no effect until that secret exists.
YT_DLP_PROXY = os.environ.get("YT_DLP_PROXY", "")


def _normalize_cookies_file(raw_path):
    """
    Accepts whatever a cookie-export tool produced at raw_path and returns
    the path to a Netscape-format cookies.txt yt-dlp can actually use.
    - Already Netscape format (starts with the standard header, or is
      tab-separated with 7 fields per line) -> used as-is.
    - A JSON array of cookie objects (the common shape most browser
      extensions - including iOS Safari cookie apps - export, with fields
      like domain/name/value/path/expirationDate/secure) -> converted.
    """
    with open(raw_path, "r", encoding="utf-8") as f:
        content = f.read()
    stripped = content.strip()
    if not stripped:
        return raw_path
    if stripped.startswith("#") or "\t" in stripped.splitlines()[0]:
        return raw_path  # already Netscape format
    if stripped[0] not in "[{":
        return raw_path  # not JSON, not obviously ours to fix - pass through

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return raw_path
    if isinstance(data, dict):
        data = data.get("cookies", [data])

    lines = ["# Netscape HTTP Cookie File", "# Auto-converted from JSON export"]
    for c in data:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        domain = c.get("domain", "") or ""
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/") or "/"
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = c.get("expirationDate") or c.get("expiry") or c.get("expires") or 0
        try:
            expiry = int(float(expiry))
        except (TypeError, ValueError):
            expiry = 0
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append("\t".join([domain, include_subdomains, path, secure, str(expiry), name, value]))

    normalized_path = raw_path + ".normalized.txt"
    with open(normalized_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return normalized_path


def _yt_dlp_proxy_args():
    if YT_DLP_PROXY:
        return ["--proxy", YT_DLP_PROXY]
    return []


def _yt_dlp_auth_args():
    if YT_DLP_COOKIES_FILE and os.path.exists(YT_DLP_COOKIES_FILE):
        usable_path = _normalize_cookies_file(YT_DLP_COOKIES_FILE)
        return ["--cookies", usable_path]
    return []

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
    """
    Latest livestream on the channel that has actually started - either
    still running ("live") or finished ("none" broadcast status) - never
    one that's merely scheduled/upcoming. Chat pulled from a still-running
    stream will only cover whatever's happened up to the moment this runs,
    since yt-dlp's clean "grab the whole replay" behavior only applies once
    a stream has ended.
    """
    q = urllib.parse.urlencode({
        "key": api_key, "channelId": channel_id, "part": "snippet",
        "order": "date", "type": "video", "maxResults": 10,
    })
    with urllib.request.urlopen(f"{YOUTUBE_API}/search?{q}") as r:
        data = json.load(r)
    for item in data.get("items", []):
        vid = item["id"]["videoId"]
        details = get_video_details(vid, api_key)
        if details and details.get("liveBroadcastContent") in ("none", "live") \
                and details.get("actualStartTime"):
            return vid, details
    raise SystemExit("No started livestream (running or completed) found in the last 10 uploads.")


def get_video_details(video_id, api_key):
    q = urllib.parse.urlencode({
        "key": api_key, "id": video_id,
        "part": "liveStreamingDetails,snippet,contentDetails",
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
        "activeLiveChatId": live.get("activeLiveChatId"),
        "liveBroadcastContent": item["snippet"].get("liveBroadcastContent", "none"),
        "duration_seconds": parse_iso8601_duration(item.get("contentDetails", {}).get("duration")),
    }


def parse_iso8601_duration(duration):
    """'PT1H2M3S' -> 3723. Returns None if duration is missing/unparseable."""
    if not duration:
        return None
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", duration)
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


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
        "yt-dlp", "--skip-download", "--ignore-no-formats-error", "--write-subs",
        "--sub-langs", "live_chat", "--sub-format", "json",
        *YT_DLP_JS_RUNTIME_ARGS, *YT_DLP_CLIENT_ARGS, *_yt_dlp_proxy_args(), *_yt_dlp_auth_args(),
        "-o", os.path.join(workdir, "%(id)s.%(ext)s"), url,
    ]
    # check=False, not check=True: --ignore-no-formats-error means yt-dlp can
    # still exit nonzero even when the subtitle track downloaded fine (e.g.
    # a video with no playable video/audio format at all, live chat replay
    # only). Whether the expected output file actually landed on disk is the
    # real signal of success, not the exit code.
    result = subprocess.run(cmd, check=False)
    path = os.path.join(workdir, f"{video_id}.live_chat.json")
    if not os.path.exists(path):
        raise SystemExit(f"yt-dlp did not produce {path} (exit code {result.returncode})")
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
# Step 2b: caption-cue fallback - for shows with no archive_windows data
# (older shows, or any night the archive step didn't happen to capture
# clean on/off-stage timestamps). Detects Chino's spoken introduction cue
# ("our next comic is ___, give it up for ___") from YouTube's own
# auto-generated captions and uses each cue as a window boundary. Closer
# callbacks ("once again / one more time, give it up for ___") are
# recognized and skipped, so they don't get mistaken for a new comic.
#
# This is a best-effort fallback, not a replacement for real Sheet data:
# auto-captions are reliable at flagging THAT a transition happened, much
# less reliable at spelling a comic's actual stage name correctly. Treat
# the detected names as a starting guess a human should skim, not ground
# truth to publish blindly.
# --------------------------------------------------------------------------
# Unambiguous "this is a NEW comic" phrases - always trusted as a
# transition on their own.
TRANSITION_PATTERNS = [
    re.compile(r"please welcome\s+([a-z][a-z' -]{1,24})", re.I),
    re.compile(r"our next comic is\s+([a-z][a-z' -]{1,24})", re.I),
    re.compile(r"next up (?:we have|is)\s+([a-z][a-z' -]{1,24})", re.I),
]
# "give it up for ___" is ambiguous on its own - it's used both for a
# genuine new introduction (right after "our next comic is ___") AND for a
# closer's callback to someone who already went ("once again/one more
# time, give it up for ___"). Chino's actual callback phrasing always
# leads with one of these - if the ~40 characters right before the match
# contain one, it's a repeat mention, not a new comic, and gets skipped.
GIVE_IT_UP_PATTERN = re.compile(r"give it up for\s+([a-z][a-z' -]{1,24})", re.I)
CALLBACK_LEAD_IN = re.compile(r"(?:once again|one more time|again)\W*$", re.I)
# Chino's spoken cue that kicks off the actual show, as opposed to any
# pre-show waiting-room banter beforehand. Used as a fallback show-start
# marker (see find_show_start_offset) when YouTube's API doesn't report a
# live-stream start time for a video.
BEGIN_SIMULATION_PATTERN = re.compile(r"begin simulation", re.I)
# Cues within this many seconds of each other collapse into one - repeated
# or split captions ("give it up for... give it up for Dove!") shouldn't
# create a second, spurious comic boundary.
CUE_DEDUPE_SECONDS = 45
# A stray "give it up for" mid-set (a callback bit, a crowd shoutout) is
# common - stop candidate name at the next few filler/stop words so a
# runaway match doesn't eat the rest of the sentence.
NAME_STOPWORDS = {"everybody", "everyone", "guys", "tonight", "coming",
                   "the", "next", "our", "on", "stage", "please", "up",
                   "for", "to", "come", "and", "give", "it", "one"}


def download_auto_captions(video_id, workdir):
    """
    Pulls YouTube's auto-generated English captions (speech-to-text, with
    timestamps) via yt-dlp - a separate track from the live chat replay.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp", "--skip-download", "--ignore-no-formats-error", "--write-auto-subs",
        "--sub-langs", "en", "--sub-format", "json3",
        *YT_DLP_JS_RUNTIME_ARGS, *YT_DLP_CLIENT_ARGS, *_yt_dlp_proxy_args(), *_yt_dlp_auth_args(),
        "-o", os.path.join(workdir, "%(id)s.%(ext)s"), url,
    ]
    # check=False, not check=True: --ignore-no-formats-error means yt-dlp can
    # still exit nonzero even when the caption track downloaded fine (e.g. a
    # video yt-dlp has no playable video/audio format for at all). Whether
    # the expected output file actually landed on disk is the real signal of
    # success, not the exit code.
    result = subprocess.run(cmd, check=False)
    path = os.path.join(workdir, f"{video_id}.en.json3")
    if not os.path.exists(path):
        raise SystemExit(f"yt-dlp did not produce auto-caption file {path} "
                          f"(exit code {result.returncode}) - the video may not have "
                          f"auto-captions available")
    return path


def parse_caption_events(path):
    """[(start_seconds, text), ...] from a yt-dlp json3 caption file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for ev in data.get("events", []):
        start_ms = ev.get("tStartMs")
        segs = ev.get("segs") or []
        text = "".join(s.get("utf8", "") for s in segs)
        if start_ms is None or not text.strip():
            continue
        out.append((start_ms / 1000.0, text))
    return out


def _clean_name(raw):
    """Trim a regex-captured name down to just the plausible name tokens."""
    words = raw.strip().split()
    kept = []
    for w in words:
        bare = w.strip(".,!?:;\"'").lower()
        if bare in NAME_STOPWORDS or not bare:
            break
        kept.append(w.strip(".,!?:;\"'"))
        if len(kept) >= 3:  # names are short - stop runaway matches
            break
    return " ".join(kept).strip() or "unknown"


def find_cue_transitions(events):
    """
    events: [(start_seconds, text), ...] from parse_caption_events.
    Returns a de-duplicated, time-sorted list of
    {"offset_seconds", "name", "raw_context"} - one per detected comic
    introduction.
    """
    # Build one long lowercase transcript plus a parallel char-offset ->
    # timestamp index, so a regex match anywhere (even split across two
    # caption events) can still be traced back to a real timestamp.
    full_text_parts = []
    offsets = []  # char offset at the START of each event's text
    cursor = 0
    for start_sec, text in events:
        offsets.append((cursor, start_sec))
        full_text_parts.append(text)
        cursor += len(text) + 1  # +1 for the joining space below
    full_text = " ".join(full_text_parts)
    offset_positions = [o for o, _ in offsets]
    offset_times = [t for _, t in offsets]

    def time_at(char_pos):
        i = bisect.bisect_right(offset_positions, char_pos) - 1
        i = max(0, min(i, len(offset_times) - 1))
        return offset_times[i]

    raw_hits = []
    for pattern in TRANSITION_PATTERNS:
        for m in pattern.finditer(full_text):
            ts = time_at(m.start())
            name = _clean_name(m.group(1))
            context = full_text[max(0, m.start() - 20): m.end() + 20]
            raw_hits.append({"offset_seconds": ts, "name": name, "raw_context": context})

    for m in GIVE_IT_UP_PATTERN.finditer(full_text):
        lead_in = full_text[max(0, m.start() - 40): m.start()]
        if CALLBACK_LEAD_IN.search(lead_in):
            continue  # "once again / one more time, give it up for ___" - a repeat mention, not a new comic
        ts = time_at(m.start())
        name = _clean_name(m.group(1))
        context = full_text[max(0, m.start() - 20): m.end() + 20]
        raw_hits.append({"offset_seconds": ts, "name": name, "raw_context": context})

    raw_hits.sort(key=lambda h: h["offset_seconds"])
    deduped = []
    for hit in raw_hits:
        if deduped and hit["offset_seconds"] - deduped[-1]["offset_seconds"] < CUE_DEDUPE_SECONDS:
            continue
        deduped.append(hit)
    return deduped


def find_show_start_offset(events):
    """
    Returns the offset (seconds into the video) of the first "begin
    simulation" cue, or None if it's never said. events is the same
    [(start_seconds, text), ...] shape parse_caption_events returns.
    Checks each caption event individually, and adjacent pairs joined
    together, since the phrase can land split across two caption chunks.
    """
    for start_sec, text in events:
        if BEGIN_SIMULATION_PATTERN.search(text):
            return start_sec
    for i in range(len(events) - 1):
        joined = events[i][1] + " " + events[i + 1][1]
        if BEGIN_SIMULATION_PATTERN.search(joined):
            return events[i][0]
    return None


def build_windows_from_cues(cue_transitions, video_duration_seconds, stream_start_iso):
    """
    Turns a list of detected cue transitions into archive_windows-shaped
    {name, on_stage_at, done_at} entries, so they can flow through the
    exact same build_report() pipeline as real Sheet data. Each comic's
    window runs from their own cue to the next cue (or to the end of the
    video for the last one).
    """
    stream_start = datetime.fromisoformat(stream_start_iso.replace("Z", "+00:00"))
    windows = []
    for i, cue in enumerate(cue_transitions):
        start = cue["offset_seconds"]
        end = cue_transitions[i + 1]["offset_seconds"] if i + 1 < len(cue_transitions) \
            else (video_duration_seconds if video_duration_seconds else start + 600)
        windows.append({
            "name": f"{cue['name']} (auto-detected, please verify)",
            "on_stage_at": (stream_start + timedelta(seconds=start)).isoformat(),
            "done_at": (stream_start + timedelta(seconds=end)).isoformat(),
        })
    return windows


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
def build_report(chat_messages, windows, stream_start_iso, window_source="sheet_archive"):
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

    stats = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "window_source": window_source, "comics": {}}
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


# --------------------------------------------------------------------------
# Live polling - a lighter-weight companion to the main --auto/--video-id
# path above, meant to run every ~5 minutes (via its own scheduled workflow)
# while a show is still in progress, so a site can show sentiment updating
# during the night instead of only after it ends.
#
# Comic-level windows (build_report, above) need archive_windows, which the
# Sheet only has once the show has finished and Finish Show has been
# pressed - not available while still live. So this buckets by 5-minute
# wall-clock chunks instead of by comic; the regular post-show run is still
# what produces the definitive per-comic report once the show ends.
#
# Getting chat WHILE a stream is live is also a different mechanism than
# the finished-replay download above (that trick only works once a stream
# has ended) - this uses YouTube's liveChatMessages API instead of yt-dlp.
# Checking "is anything live right now" cheaply matters too: a full search
# query costs 100 quota units, which would blow the daily 10,000-unit quota
# if run every 5 minutes all day. Looking at the channel's single latest
# upload's status instead costs about 2-3 units, so polling all day is fine.
# --------------------------------------------------------------------------
LIVE_STATE_FILENAME = "live_poll_state.json"


def get_uploads_playlist_id(channel_id, api_key):
    """1 quota unit. The channel's "uploads" playlist ID barely ever
    changes, so callers should cache and reuse this across polls."""
    q = urllib.parse.urlencode({"key": api_key, "id": channel_id, "part": "contentDetails"})
    with urllib.request.urlopen(f"{YOUTUBE_API}/channels?{q}") as r:
        data = json.load(r)
    items = data.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_latest_upload_video_id(playlist_id, api_key):
    """1 quota unit."""
    q = urllib.parse.urlencode({
        "key": api_key, "playlistId": playlist_id, "part": "contentDetails", "maxResults": 1,
    })
    with urllib.request.urlopen(f"{YOUTUBE_API}/playlistItems?{q}") as r:
        data = json.load(r)
    items = data.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["videoId"]


def get_live_stream_status(video_id, api_key):
    """1 quota unit - just enough to tell if this one video is live right
    now and, if so, its chat ID."""
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
        "liveBroadcastContent": item["snippet"].get("liveBroadcastContent", "none"),
        "activeLiveChatId": live.get("activeLiveChatId"),
    }


def find_currently_live_video(channel_id, api_key, cached_playlist_id=None):
    """
    Cheap way (~2-3 quota units total, vs 100 for a search query) to check
    whether the channel currently has a live broadcast: look at its single
    most recent upload and check its live status directly, rather than
    searching. Returns (video_id, status_dict, uploads_playlist_id) when
    something's live, or (None, None, uploads_playlist_id) when not -
    callers should cache and pass back uploads_playlist_id to skip the
    lookup on the next poll.
    """
    playlist_id = cached_playlist_id or get_uploads_playlist_id(channel_id, api_key)
    if not playlist_id:
        return None, None, playlist_id
    video_id = get_latest_upload_video_id(playlist_id, api_key)
    if not video_id:
        return None, None, playlist_id
    status = get_live_stream_status(video_id, api_key)
    if status and status.get("liveBroadcastContent") == "live" and status.get("activeLiveChatId"):
        return video_id, status, playlist_id
    return None, None, playlist_id


def load_live_state(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_live_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def poll_live_chat_once(live_chat_id, api_key, page_token=None):
    """One page of new live chat messages since page_token (None = start
    from whatever's currently in the chat, not the whole history)."""
    params = {"key": api_key, "liveChatId": live_chat_id, "part": "snippet,authorDetails"}
    if page_token:
        params["pageToken"] = page_token
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{YOUTUBE_API}/liveChat/messages?{q}") as r:
        return json.load(r)


def append_live_messages(rows_path, items):
    """Scores and appends new live-chat items to a running CSV."""
    if not items:
        return
    file_exists = os.path.exists(rows_path)
    os.makedirs(os.path.dirname(rows_path) or ".", exist_ok=True)
    with open(rows_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "author", "message", "sentiment"])
        if not file_exists:
            writer.writeheader()
        for item in items:
            snippet = item.get("snippet", {})
            text = snippet.get("displayMessage", "")
            published = snippet.get("publishedAt", "")
            author = item.get("authorDetails", {}).get("displayName", "")
            writer.writerow({
                "timestamp_utc": published,
                "author": author,
                "message": text,
                "sentiment": score_sentiment(text),
            })


def update_live_stats(rows_path, stats_path, video_id):
    """
    Rebuilds a rolling stats file from the running CSV, bucketed into
    5-minute wall-clock chunks (comic windows aren't known yet for a show
    that's still live - see the module docstring above).
    """
    buckets = {}
    total_messages = 0
    sentiment_sum = 0.0
    if os.path.exists(rows_path):
        with open(rows_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = row.get("timestamp_utc", "")
                try:
                    score = float(row.get("sentiment") or 0.0)
                except ValueError:
                    score = 0.0
                total_messages += 1
                sentiment_sum += score
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        bucket_key = dt.replace(minute=(dt.minute // 5) * 5, second=0,
                                                 microsecond=0).isoformat()
                    except ValueError:
                        bucket_key = "unknown"
                else:
                    bucket_key = "unknown"
                b = buckets.setdefault(bucket_key, {"messages": 0, "sentiment_sum": 0.0})
                b["messages"] += 1
                b["sentiment_sum"] += score

    bucket_stats = {}
    for key, b in sorted(buckets.items()):
        bucket_stats[key] = {
            "messages": b["messages"],
            "avg_sentiment": round(b["sentiment_sum"] / b["messages"], 4) if b["messages"] else 0.0,
        }

    stats = {
        "video_id": video_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "window_source": "live_poll_5min_buckets",
        "totals": {
            "messages": total_messages,
            "avg_sentiment": round(sentiment_sum / total_messages, 4) if total_messages else 0.0,
        },
        "buckets": bucket_stats,
    }
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def run_live_poll(channel_id, api_key, out_dir):
    """
    One shot, meant to be called every ~5 minutes by a scheduled workflow.
    Fast no-op when nothing's currently live. When a show is live, pulls
    whatever new chat has arrived since the last poll and updates that
    show's rolling chat.csv/stats.json under out_dir.
    """
    state_path = os.path.join(out_dir, LIVE_STATE_FILENAME)
    state = load_live_state(state_path)

    video_id = state.get("video_id")
    live_chat_id = state.get("live_chat_id")
    playlist_id = state.get("uploads_playlist_id")
    still_tracking = False

    if video_id and live_chat_id:
        status = get_live_stream_status(video_id, api_key)
        if status and status.get("liveBroadcastContent") == "live" and status.get("activeLiveChatId"):
            live_chat_id = status["activeLiveChatId"]
            still_tracking = True
        else:
            print(f"Show {video_id} is no longer live - doing a final poll, then clearing state.")
    else:
        new_video_id, status, playlist_id = find_currently_live_video(channel_id, api_key, playlist_id)
        state["uploads_playlist_id"] = playlist_id
        if not new_video_id:
            print("No show currently live - nothing to poll.")
            save_live_state(state_path, state)
            return
        video_id = new_video_id
        live_chat_id = status["activeLiveChatId"]
        state.update({
            "video_id": video_id, "live_chat_id": live_chat_id, "next_page_token": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        still_tracking = True
        print(f"New live show detected: {video_id} ({status.get('title')}) - starting live poll.")

    data = poll_live_chat_once(live_chat_id, api_key, state.get("next_page_token"))
    items = data.get("items", [])
    rows_path = os.path.join(out_dir, f"{video_id}.chat.csv")
    stats_path = os.path.join(out_dir, f"{video_id}.stats.json")
    append_live_messages(rows_path, items)
    update_live_stats(rows_path, stats_path, video_id)
    print(f"Polled {len(items)} new message(s) for {video_id}.")

    state["next_page_token"] = data.get("nextPageToken", state.get("next_page_token"))
    if not still_tracking:
        state.pop("video_id", None)
        state.pop("live_chat_id", None)
        state.pop("next_page_token", None)
        print(f"Show {video_id} ended - the regular post-show pipeline run still produces the "
              f"definitive per-comic report once its data lands in the Sheet.")

    save_live_state(state_path, state)


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


def extract_video_id(raw):
    """
    Accepts a bare video ID OR a pasted YouTube URL of any common shape
    (youtu.be/..., youtube.com/watch?v=..., /live/..., /shorts/...) and
    returns just the ID. Also cleans up stray whitespace/punctuation that
    tends to come along with a manual copy-paste (a trailing period from
    autocorrect, a stray space, share-link tracking params like ?si=...).
    """
    if not raw:
        return raw
    s = raw.strip().strip(".,;:!?\"'")
    m = re.search(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|live/|shorts/|embed/))([A-Za-z0-9_-]{6,})", s)
    if m:
        s = m.group(1)
    s = s.split("&")[0].split("?")[0].split("#")[0]
    return s.strip()


def sanitize_date(raw):
    """Pulls a YYYY-MM-DD out of whatever was typed, tolerating a stray
    trailing period/space/etc. from manual entry."""
    if not raw:
        return raw
    s = raw.strip().strip(".,;:!?\"'")
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return m.group(0) if m else s


def sanitize_url(raw):
    if not raw:
        return raw
    return raw.strip().strip(".,;:!?\"'")


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
    ap.add_argument("--live-poll", action="store_true",
                     help="One-shot live chat poll (meant to run every ~5 min via its own schedule) - "
                          "fast no-op if nothing's currently live")
    args = ap.parse_args()

    # Clean up manually-typed/pasted inputs before they hit anything else -
    # a stray trailing period, space, or a full URL where a bare ID was
    # expected has already caused real failures here.
    args.video_id = extract_video_id(args.video_id) if args.video_id else args.video_id
    args.date = sanitize_date(args.date) if args.date else args.date
    args.archive_base = sanitize_url(args.archive_base) if args.archive_base else args.archive_base
    args.channel_id = args.channel_id.strip() if args.channel_id else args.channel_id

    if args.live_poll:
        if not args.channel_id or not args.youtube_api_key:
            raise SystemExit("--live-poll requires --channel-id and --youtube-api-key")
        run_live_poll(args.channel_id, args.youtube_api_key, args.out)
        return

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

    video_duration_seconds = None
    # True once stream_start_iso comes straight from YouTube's own
    # liveStreamingDetails.actualStartTime - an exact, trustworthy anchor.
    # False for any of the approximate fallbacks below, which get refined
    # against the "begin simulation" caption cue a bit further down.
    stream_start_precise = True
    details = None

    if args.auto:
        if not args.channel_id or not args.youtube_api_key:
            raise SystemExit("--auto requires --channel-id and --youtube-api-key")
        video_id, details = find_latest_completed_video(args.channel_id, args.youtube_api_key)
        stream_start_iso = details["actualStartTime"]
        video_duration_seconds = details.get("duration_seconds")
        show_date = args.date or stream_start_iso[:10]
        status = "still running" if details.get("liveBroadcastContent") == "live" else "completed"
        print(f"Selected video {video_id} ({status}): {details['title']} ({stream_start_iso})")
        if status == "still running":
            print("Note: this show hasn't ended yet - chat will only cover what's happened so far, "
                  "and archive_windows likely has no data yet either.")
    else:
        if not args.video_id:
            raise SystemExit("Provide --video-id, or use --auto, or use --demo")
        video_id = args.video_id
        if args.youtube_api_key:
            details = get_video_details(video_id, args.youtube_api_key)
            if not details:
                raise SystemExit(f"YouTube API returned no video for id {video_id!r} - "
                                  f"double check it's just the ID, not a full URL.")
            # A video can come back from the API with no actualStartTime -
            # it's not a completed livestream (a regular upload, or one
            # that's upcoming/still live). Fall back to --date if given,
            # then to when it was published, rather than crashing.
            stream_start_iso = details.get("actualStartTime")
            if not stream_start_iso:
                stream_start_precise = False
                if args.date:
                    stream_start_iso = args.date + "T00:00:00Z"
                elif details.get("publishedAt"):
                    print(f"Video {video_id} has no live-stream start time "
                          f"(not a completed livestream?) - using its publish "
                          f"date instead. Pass --date explicitly if that's wrong.")
                    stream_start_iso = details["publishedAt"]
                else:
                    raise SystemExit(f"Video {video_id} has no live-stream start time and "
                                      f"no publish date either - pass --date explicitly.")
            video_duration_seconds = details.get("duration_seconds")
        elif args.date:
            # No API key available - assume the stream started at local
            # midnight of --date; this only matters for lining chat offsets
            # up with archive_windows, so pass --youtube-api-key when possible.
            stream_start_precise = False
            stream_start_iso = args.date + "T00:00:00Z"
        else:
            raise SystemExit("Provide --date or --youtube-api-key so the stream start time is known")
        show_date = args.date or stream_start_iso[:10]

    if not stream_start_precise:
        # The anchor above (--date at midnight, or a publish date) is only
        # a rough guess at when the video actually starts. Chino's spoken
        # "begin simulation" cue marks the real show start precisely - if
        # auto-captions catch it, use that offset from a real timestamp
        # anchor (the video's publish time, when available) instead of the
        # rough guess.
        anchor_iso = (details.get("publishedAt") if details else None) or stream_start_iso
        try:
            caption_path = download_auto_captions(video_id, workdir)
            caption_events = parse_caption_events(caption_path)
            start_offset = find_show_start_offset(caption_events)
        except SystemExit:
            start_offset = None
        if start_offset is not None:
            anchor_dt = datetime.fromisoformat(anchor_iso.replace("Z", "+00:00"))
            stream_start_iso = (anchor_dt + timedelta(seconds=start_offset)).isoformat()
            show_date = args.date or stream_start_iso[:10]
            print(f"Detected the 'begin simulation' cue at {start_offset:.0f}s into the video - "
                  f"using that as the precise show start ({stream_start_iso}).")
        else:
            print("Didn't detect a 'begin simulation' cue in the auto-captions - "
                  "keeping the approximate show start time from above.")

    if not args.archive_base:
        raise SystemExit("--archive-base is required (the SetPulse Apps Script /exec URL)")

    resolved_date, windows = fetch_archive_windows_with_lag_search(args.archive_base, show_date)
    window_source = "sheet_archive"
    if windows:
        print(f"Using archive_windows for {resolved_date}: {len(windows)} performer(s)")
    else:
        # Nothing was ever formally archived for this show (Finish Show may
        # never have been pressed, or this predates that data existing) -
        # fall back to detecting Chino's spoken cue from auto-captions.
        print(f"No archive_windows for {show_date} - falling back to auto-caption cue detection")
        caption_path = download_auto_captions(video_id, workdir)
        caption_events = parse_caption_events(caption_path)
        cue_transitions = find_cue_transitions(caption_events)
        if not cue_transitions:
            raise SystemExit(f"No archive_windows found for {show_date}, and no comic-introduction "
                              f"cues were detected in the auto-captions either. Nothing to build a "
                              f"per-comic report from.")
        windows = build_windows_from_cues(cue_transitions, video_duration_seconds, stream_start_iso)
        window_source = "caption_cue_auto_detected"
        print(f"Detected {len(windows)} comic transition(s) from auto-captions - "
              f"names are best-effort guesses, review before treating as final:")
        for w in windows:
            print(f"  - {w['name']}: {w['on_stage_at']} -> {w['done_at']}")

    chat_path = download_live_chat(video_id, workdir)
    chat_messages = parse_live_chat(chat_path)
    print(f"Parsed {len(chat_messages)} chat messages")

    rows, stats = build_report(chat_messages, windows, stream_start_iso, window_source=window_source)
    csv_path, json_path = write_outputs(rows, stats, args.out)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
