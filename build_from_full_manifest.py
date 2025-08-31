import argparse
import json
import os
import shutil
import subprocess
from typing import Optional, List, Dict, Any
import numpy as np
from index_db import init_db, add_song, add_hashes
from audio_hasher import make_hash_from_audio

DB_PATH = "audio_index.db"
YTDLP_DOMAINS = ("youtube.com/", "youtu.be/", "music.youtube.com/")

# ----------------------------
def run(cmd, timeout):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, (err or "") + "\n[timeout]"
    return p.returncode, out, err

# ----------------------------
def pick_http_url(text):
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return s
    return None

# ----------------------------
def get_best_audio_url_with_file(url: str, cookies_file: str) -> str:
    trials = [
        ["yt-dlp", "--geo-bypass", "--no-warnings", "-q",
         "--cookies", cookies_file, "-f", "bestaudio[ext=m4a]/bestaudio/best", "-g", url],
        ["yt-dlp", "--geo-bypass", "--no-warnings", "-q",
         "--cookies", cookies_file, "--extractor-args", "youtube:player_client=web,ios,android,tv",
         "-f", "bestaudio", "-g", url],
        ["yt-dlp", "--geo-bypass", "--no-warnings", "-q",
         "--cookies", cookies_file, "-f", "bestaudio", "-g", url],
    ]
    last_err = ""
    for cmd in trials:
        code, out, err = run(cmd, timeout=90)
        if code == 0:
            media = pick_http_url(out)
            if media:
                return media
        last_err = err or out or f"(exit {code})"
    raise RuntimeError(f"yt-dlp could not resolve audio URL (cookies file).\nLast error:\n{last_err.strip()}")

# ----------------------------
def get_best_audio_from_url(url, cookies_browser, cookies_file, artist, title):

    if cookies_file:
        if not os.path.exists(cookies_file):
            raise RuntimeError(f"cookies file not found: {cookies_file}")
        return get_best_audio_url_with_file(url, cookies_file)

    base = ["yt-dlp", "--geo-bypass", "--no-warnings", "-q"]
    if cookies_browser:
        base += ["--cookies-from-browser", cookies_browser]

    trials = [
        (["-f", "bestaudio[ext=m4a]/bestaudio/best", "-g", url], "best m4a/bestaudio"),
        (["--extractor-args", "youtube:player_client=web,ios,android,tv", "-f", "bestaudio", "-g", url], "multi client"),
        (["-f", "bestaudio", "-g", url], "simple bestaudio"),
    ]
    last_err = ""
    for extra, _label in trials:
        code, out, err = run(base + extra, timeout=90)
        if code == 0:
            media = pick_http_url(out)
            if media:
                return media
        last_err = err or out or f"(exit {code})"

    if artist and title:
        for q in [f"ytsearch5:{artist} - {title} official audio",
                  f"ytsearch5:{artist} - {title} audio",
                  f"ytsearch5:{artist} - {title} topic",
                  f"ytsearch5:{artist} {title}"]:
            code, out, err = run(base + ["-f", "bestaudio", "-g", q], timeout=60)
            if code == 0:
                media = pick_http_url(out)
                if media:
                    return media
            last_err = err or out or f"(exit {code})"

    raise RuntimeError(f"yt-dlp could not resolve audio URL.\nLast error:\n{last_err.strip()}")

# ----------------------------
def ffmpeg_stream_media_url(media_url, target_sr):
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", media_url,
        "-vn",
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed.\n{(err or b'').decode('utf-8', errors='ignore')}")
    return np.frombuffer(raw, dtype=np.float32)

# ----------------------------
def read_full_track_pcm(url, target_sr, cookies_browser, cookies_file, artist, title):

    media_url = get_best_audio_from_url(url, cookies_browser=cookies_browser, cookies_file=cookies_file, artist=artist, title=title,)
    audio = ffmpeg_stream_media_url(media_url, target_sr=target_sr)
    return audio, target_sr

# ----------------------------

def load_manifest_any(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip().rstrip(",")
                if not line or line in ("[", "]", "..."):
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                    elif isinstance(obj, list):
                        items.extend(obj)
                except json.JSONDecodeError:
                    continue
        if not items:
            raise
        return items

# ----------------------------
def normalize_items(items):
    clean: List[Dict[str, Any]] = []
    for it in items:
        if not all(k in it for k in ("song_id", "artist", "title", "url")):
            continue
        clean.append({
            "song_id": int(it["song_id"]),
            "artist": str(it["artist"]),
            "title":  str(it["title"]),
            "url":    str(it["url"]),
        })
    clean.sort(key=lambda x: x["song_id"])
    return clean

# ----------------------------
def song_exists(conn, song_id):
    cur = conn.cursor()
    row = cur.execute("SELECT 1 FROM songs WHERE song_id = ?", (int(song_id),)).fetchone()
    return row is not None

# ----------------------------
def delete_song(conn, song_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM hashes WHERE song_id = ?", (int(song_id),))
    cur.execute("DELETE FROM songs  WHERE song_id = ?", (int(song_id),))
    conn.commit()

# ----------------------------
def build_fingerprint_db_from_json(manifest_path, db_path, cookies_browser, cookies_file):
    items = normalize_items(load_manifest_any(manifest_path))

    conn = init_db(db_path)
    try:
        total = len(items)
        for i, item in enumerate(items, 1):
            sid, artist, title, url = item["song_id"], item["artist"], item["title"], item["url"]

            if song_exists(conn, sid):
                print(f"[{i}/{total}] SKIP (already in DB): {artist} - {title} (id={sid})")
                continue

            print(f"[{i}/{total}] Streaming: {artist} - {title} (id={sid})")
            try:
                audio, sr = read_full_track_pcm(
                    url, target_sr=8000,
                    cookies_browser=cookies_browser,
                    cookies_file=cookies_file,
                    artist=artist, title=title
                )
                hashes = make_hash_from_audio(audio, sr)
                if not hashes:
                    print("    -> no peaks found, skipping")
                    continue
                add_song(conn, sid, artist, title)
                add_hashes(conn, sid, hashes)
                conn.commit()
                print(f"    -> stored {len(hashes)} hashes, len={len(audio)/sr:.1f}s")
            except Exception as e:
                print(f"    !! FAILED: {e}")
                continue
    finally:
        conn.close()

# ----------------------------
def rebuild_specific_ids(manifest_path, db_path, ids, cookies_browser, cookies_file, force):
    want = set(int(x) for x in ids)
    items = normalize_items(load_manifest_any(manifest_path))
    items = [it for it in items if it["song_id"] in want]
    if not items:
        print("No matching song_id found in manifest.")
        return

    conn = init_db(db_path)
    try:
        total = len(items)
        for i, item in enumerate(items, 1):
            sid, artist, title, url = item["song_id"], item["artist"], item["title"], item["url"]

            if song_exists(conn, sid):
                if force:
                    print(f"[{i}/{total}] FORCE re-index: {artist} - {title} (id={sid})")
                    delete_song(conn, sid)
                else:
                    print(f"[{i}/{total}] SKIP (already in DB): {artist} - {title} (id={sid})")
                    continue
            else:
                print(f"[{i}/{total}] Re-index: {artist} - {title} (id={sid})")

            try:
                audio, sr = read_full_track_pcm(url, target_sr=8000, cookies_browser=cookies_browser, cookies_file=cookies_file, artist=artist, title=title)
                hashes = make_hash_from_audio(audio, sr)
                if not hashes:
                    print("    -> no peaks found, skipping")
                    continue
                add_song(conn, sid, artist, title)
                add_hashes(conn, sid, hashes)
                conn.commit()
                print(f"    -> stored {len(hashes)} hashes, len={len(audio)/sr:.1f}s")
            except Exception as e:
                print(f"    !! FAILED: {e}")
                continue
    finally:
        conn.close()

# ----------------------------
# main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build/Repair SQLite fingerprint DB by streaming FULL tracks (YouTube supported).")
    ap.add_argument("manifest", help="Path to manifest JSON (list or JSONL of {song_id, artist, title, url})")
    ap.add_argument("--db", default=DB_PATH, help="SQLite DB path (default audio_index.db)")
    ap.add_argument("--cookies", dest="cookies_file", help="Path to cookies.txt (exported for youtube.com & music.youtube.com)")
    ap.add_argument("--cookies-browser", help="Browser profile for yt-dlp cookies, e.g. 'edge:Default' or 'chrome:Default'")
    ap.add_argument("--ids", help="Comma-separated song_id list to (re)build, e.g. 12,45,78")
    ap.add_argument("--force", action="store_true", help="Overwrite existing DB rows for the given --ids")
    args = ap.parse_args()

    ids = None
    if args.ids:
        ids = [int(s.strip()) for s in args.ids.split(",") if s.strip().isdigit()]

    if ids:
        rebuild_specific_ids(args.manifest, db_path=args.db, ids=ids, cookies_browser=args.cookies_browser, cookies_file=args.cookies_file, force=args.force)
    else:
        build_fingerprint_db_from_json(args.manifest, db_path=args.db, cookies_browser=args.cookies_browser, cookies_file=args.cookies_file)

    print(f"Saved SQLite index to {args.db}")