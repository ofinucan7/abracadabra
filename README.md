# Abracadabra - Song Recognition Algorithm

## Repo Contents
Note: This repo does **not** contain the database file. With the list of songs in songs.json, it turns out to be ~21GB (hence why it is not present).
- `audio_hasher.py` — audio → landmarks → hash pairs
- `build_from_full_manifest.py` — stream full tracks and build the DB
- `index_db.py` — SQLite schema & helpers (`songs`, `hashes`)
- `recognize_snippet.py` — recognize a WAV file or URL segment using the DB
- `requirements.txt` — text file with required Python dependencies
- `song-test-key.txt` — Key to the songs-test-urls.txt
- `songs-test-urls.txt` — List of URLs to test algorithm
- `songs.json` — list of songs which I trained the algorithm on

## Prerequisites / Guide to Running
- **Python** 3.9–3.12 (tested most with 3.11)
- **ffmpeg** on your PATH  
  - Windows: install from [ffmpeg.org] and add the `/bin` folder to System PATH  
  - macOS: `brew install ffmpeg`  
  - Linux: `sudo apt-get install ffmpeg` (Debian/Ubuntu)
- Browser cookies for yt-dlp if some URLs require age/geofence/auth (in my experience, you will need the browser cookies to ensure yt-dlp properly gets the song from Youtube)

For the browser cookies, I used [https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc]

To Build the Database:
- Basic Running (no cookies): python build_from_full_manifest.py songs.json --db audio_index.db
- With Cookies (unspecified browser): python build_from_full_manifest.py songs.json --db audio_index.db --cookies cookies.txt
- With Cookies (Chrome): python build_from_full_manifest.py songs.json --db audio_index.db --cookies-browser "chrome:Default"

To Test Songs, you can either upload a YouTube URL or have a local file. By varying the topk, you will get a different number of songs returned. The first result will be the most likely song, the second will be the next most likely, and so on.
- YouTube URL: python recognize_snippet.py "https://www.youtube.com/watch?v=<Your Youtube URL>"   --db audio_index.db --start 30 --dur 15 --topk 3   --cookies-browser "chrome:Default"
- Local File: python recognize_snippet.py path/to/snippet.wav --db audio_index.db --topk 3
