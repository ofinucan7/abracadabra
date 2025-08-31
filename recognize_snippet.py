import argparse
import os
import sqlite3
import subprocess
import librosa
import numpy as np
from collections import defaultdict
from index_db import load_meta  
from audio_hasher import make_hash_from_audio 
from build_from_full_manifest import ytdlp_get_best_audio_url


DB_PATH = "audio_index.db"
# ----------------------------
# stream the song from a given URL
def stream_from_url(media_url, start_s, dur_s, target_sr):
    cmd = ["ffmpeg", "-v", "error"]
    if start_s is not None:
        cmd += ["-ss", str(start_s)]
    cmd += [
        "-i", media_url,
        "-t", str(dur_s),
        "-vn",
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for media url.\n{(err or b'').decode('utf-8', errors='ignore')}")
    return np.frombuffer(raw, dtype=np.float32)

# ----------------------------
# get the hash fingerprint from a URL
def hash_from_url(url, start_s, dur_s, target_sr, cookies_file, cookies_browser):
    song_from_url = ytdlp_get_best_audio_url(url, cookies_browser=cookies_browser, cookies_file=cookies_file, artist=None, title=None)
    audio_file = stream_from_url(song_from_url, start_s, dur_s, target_sr)
    return make_hash_from_audio(audio_file, target_sr)

# ----------------------------
# hash from a wav file
def hash_from_wav(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return make_hash_from_audio(y, sr)

# ----------------------------
# match the hash of a given song with a song in the DB and return the top song (one with most votes)
def match_hashes_sqlite(hashes, db_path=DB_PATH, topk=1):
    if not hashes:
        return []

    by_hash = defaultdict(list) 
    for h, t in hashes:
        by_hash[h].append(int(t))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    votes = defaultdict(lambda: defaultdict(int))  
    for hex_hash, t_queries in by_hash.items():
        hbin = bytes.fromhex(hex_hash)
        rows = list(cur.execute("SELECT song_id, t_song FROM hashes WHERE hash = ?", (hbin,)))
        if not rows:
            continue
        for song_id, t_song in rows:
            ts = int(t_song)
            for tq in t_queries:
                votes[song_id][ts - tq] += 1

    scored = []
    for song_id, offsets in votes.items():
        best_delta, count = max(offsets.items(), key=lambda x: x[1])
        scored.append((song_id, count, best_delta, sum(offsets.values())))

    scored.sort(key=lambda x: x[1], reverse=True)
    meta = load_meta(conn)
    conn.close()

    results = []
    for song_id, votes_count, offset, total_hits in scored[:topk]:
        artist, title = meta.get(song_id, ("?", "?"))
        results.append((artist, title, votes_count, offset, total_hits))
    return results

# ----------------------------
# Read non-empty, non-comment lines from a text file into a list
def read_list_file(list_path):
    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items

# ----------------------------
# inputs to cmd line
def main():
    parser = argparse.ArgumentParser(
        description="Identify songs from local WAV snippets and/or a list of URLs (YouTube supported)."
    )
    parser.add_argument("inputs", nargs="*", help="WAV paths or URLs (YouTube pages allowed).")
    parser.add_argument("--list", help="Path to a text file with one URL per line.")
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path (default audio_index.db)")
    parser.add_argument("--topk", type=int, default=1, help="How many candidates to show (default 1)")

    # URL segment controls
    parser.add_argument("--start", type=int, default=None, help="Start time (seconds) to fingerprint in each URL.")
    parser.add_argument("--dur", type=int, default=15, help="Duration (seconds) to fingerprint in each URL (default 15).")

    # cookies
    parser.add_argument("--cookies", dest="cookies_file", help="Path to cookies.txt (exported for youtube.com).")
    parser.add_argument("--cookies-browser", help="Browser profile for yt-dlp cookies, e.g., 'chrome:Default' or 'edge:Default'")

    args = parser.parse_args()

    targets = []
    if args.inputs:
        targets.extend(args.inputs)
    if args.list:
        targets.extend(read_list_file(args.list))

    if not targets:
        parser.error("Provide at least one input (WAV path or URL), or use --list <file>.")

    for item in targets:
        try:
            if item.startswith("http://") or item.startswith("https://"):
                print(f"\n[URL] {item}")
                hashes = hash_from_url(
                    item,
                    start_s=args.start,
                    dur_s=args.dur,
                    target_sr=8000,
                    cookies_file=args.cookies_file,
                    cookies_browser=args.cookies_browser,
                )
            else:
                if not os.path.exists(item):
                    print(f"\n[FILE] {item} -> not found, skipping")
                    continue
                print(f"\n[FILE] {item}")
                hashes = hash_from_wav(item)

            results = match_hashes_sqlite(hashes, db_path=args.db, topk=args.topk)
            if not results:
                print("  No match candidates found.")
            else:
                for rank, (artist, title, votes, offset, total) in enumerate(results, 1):
                    print(f"  #{rank}  {artist} - {title}  (votes={votes}, offset={offset}, hits={total})")

        except Exception as e:
            print(f"  !! FAILED: {e}")

# ----------------------------
# main
if __name__ == "__main__":
    main()
