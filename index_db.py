import sqlite3

def init_db(db_path='audio_index.db'):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("""CREATE TABLE IF NOT EXISTS songs (song_id INTEGER PRIMARY KEY, artist  TEXT, title   TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS hashes (hash    BLOB NOT NULL,   -- 16-byte MD5 song_id INTEGER NOT NULL, t_song  INTEGER NOT NULL)""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hash ON hashes(hash)")
    conn.commit()
    return conn
# ----------------------------

def add_song(conn, song_id, artist, title):
    conn.execute("INSERT OR REPLACE INTO songs(song_id, artist, title) VALUES (?,?,?)", (int(song_id), artist, title),)
# ----------------------------

def add_hashes(conn, song_id, hash_list):
    rows = [(bytes.fromhex(h), int(song_id), int(t)) for h, t in hash_list]
    conn.executemany("INSERT INTO hashes(hash, song_id, t_song) VALUES (?,?,?)", rows,)
    
# ----------------------------
def load_meta(conn):
    cur = conn.cursor()
    return {
        sid: (artist, title)
        for sid, artist, title in cur.execute("SELECT song_id, artist, title FROM songs")
    }
