import os
import pickle
import hashlib
import numpy as np
import librosa
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure
from collections import defaultdict

# --------------------------
# parameters
SAMPLE_RATE = 8000 # the frequency in Hz (ie 8kHz) - lower sample rate less data but faster processing             
FFT_SIZE = 2048 # size of the fast-fourier transform - larger value, better freq resolution, rougher time resolution
HOP_SIZE = 512 # size of the jump of the FFT - smaller hop, more data points / detail
PEAK_NEIGHBORHOOD = (16, 16)  # (freq_bins, time_frames) - size of rectangle used to get local maxes
TOP_PEAKS_PER_FRAME = 16 # number of peaks to keep (based on which ones are strongest)
TARGET_ZONE_T_FRAMES = (2, 64)   # (min frames ahead, max frames ahead) - when pairing anchor with target peaks, look at least 2 frame ahead, at most 64
TARGET_ZONE_F_BINS = 48          # only pair peaks who have diff freq of +/- 48
HASH_FANOUT = 8                  # from each anchor peak, make hashes w/ up to 8 target peaks within allowed zone
FINGERPRINTS = "audio_index.pkl"   # fingerprint DB

# --------------------------
# convert audio to mono & resample to SAMPLE_RATE
# Inputs: audio_sample - numpy array of audio samples | s_r - sample rate of audio_sample
# Output: spec_in_db - spectogram in dB | s_r - sample rate 
def spectrogram(audio_sample, s_r):
    # convert to mono
    audio_sample = librosa.to_mono(audio_sample)

    # resample to the desired sample rate
    if s_r != SAMPLE_RATE:
        audio_sample = librosa.resample(audio_sample, orig_sr=s_r, target_sr=SAMPLE_RATE)
        s_r = SAMPLE_RATE
    
    # run STFT w/ inputs then convert to dB scale
    spec = np.abs(librosa.stft(audio_sample, n_fft=FFT_SIZE, hop_length=HOP_SIZE))
    spec_in_dB = librosa.amplitude_to_db(spec, ref=np.max)
    return spec_in_dB, s_r

# --------------------------
# create "foundational points" for fingerprinting
# Inputs: spec_in_db - spectrogram in dB
# Output: NP array of size (freq_bins, time_frame)
def local_maxima(spec_in_dB):
    # run max filter on input 
    neighborhood = maximum_filter(spec_in_dB, footprint=np.ones((PEAK_NEIGHBORHOOD[0], PEAK_NEIGHBORHOOD[1])), mode='constant') # PEAK_NEIGHBORHOOD[0] = 2, PEAK_NEIGHBORHOOD[1] = 64
    
    # make boolean mask where spectogram value equals local neighborhood max (ie make candidate peaks)
    local_max = (spec_in_dB == neighborhood)

    # mark background with no signal (ie no max) as -inf to better isolate points
    struct = generate_binary_structure(2, 1)
    background_erroded = binary_erosion(spec_in_dB == -np.inf, structure=struct, border_value=1)

    # mark where BOTH local_maxes and background_erroded are true
    actual_peaks = local_max & (~background_erroded)

    # get (freq_bin, time_frame) of all peaks
    peaks = np.argwhere(actual_peaks)

    # group peaks by timeframe then store their freq_bin & dB
    peak_by_time_frame = defaultdict(list)
    for freq, time in peaks:
        peak_by_time_frame[time].append((freq, spec_in_dB[freq, time]))
    
    # sort peaks in each frame by strength, keep only top ones
    kept = []
    for time, arr in peak_by_time_frame.items():
        arr.sort(key=lambda x: x[1], reverse=True)
        for freq, _ in arr[:TOP_PEAKS_PER_FRAME]:
            kept.append((freq, time))
    kept.sort(key=lambda x: x[1])
    
    # return NP array of size (freq_bins, time_frame)
    return np.array(kept, dtype=int)

# --------------------------
# hash pair of anchor peak freq bin & target peak freq bin
# Inputs: f1 - anchor peak freq bin | f2 - target peak freq bin | dt - delta t
# Output: hex hash string
def hash_pair(anchor_peak_bin, target_peak_freq_bin, dt):
    # encode string of "anchor_peak_bin|target_peak_freq_bin|dt"
    raw = f"{int(anchor_peak_bin)}|{int(target_peak_freq_bin)}|{int(dt)}".encode()
    return hashlib.md5(raw).hexdigest()

# -----------------------------
# create at most HASH_FANOUT hashes per anchor to use
# Inputs: audio - audio sample | audio_sr - sample rate of audio
# Output: list of (hash, time_anchor) pairs
def make_hash_from_audio(audio, audio_sr):
    # get spectogram and get peaks
    spec_in_dB, audio_sr = spectrogram(audio, audio_sr)
    peaks = local_maxima(spec_in_dB)
    if len(peaks) == 0:
        return []
    

    hashes = []
    for i, (freq1, time1) in enumerate(peaks):
        # get "borders" of time block
        t_min = time1 + TARGET_ZONE_T_FRAMES[0]
        t_max = time1 + TARGET_ZONE_T_FRAMES[1]
        
        count = 0
        j = i + 1

        # ugly, grotesque, disgusting
        while j < len(peaks) and peaks[j][1] <= t_max:
            freq2, time2 = peaks[j]
            if time2 >= t_min and abs(int(freq2) - int(freq1)) <= TARGET_ZONE_F_BINS:
                h = hash_pair(freq1, freq2, time2 - time1)
                hashes.append((h, time1))  
                count += 1
                if count >= HASH_FANOUT:
                    break
            j += 1
    return hashes

# -----------------------------
# load the audio then call make_hash_from_audio
# Inputs: path to .wav file
# Output: list of (hash, time_anchor) pairs
def generate_hashes_from_file(path):
    audio_input, audio_sample_rate = librosa.load(path, sr=None, mono=True)
    return make_hash_from_audio(audio_input, audio_sample_rate)

# -----------------------------
# for every file, make hashes then store them
# Inputs: list of (song_id, path, meta_dict) for all songs
# Output: dict w/ index and songs metadata map
def build_index(library_files):
    index = {}
    songs = {}

    for song_id, path, meta in library_files:
        songs[song_id] = meta
        hashes = generate_hashes_from_file(path)
        for hash, time in hashes:
            index.setdefault(hash, []).append((song_id, time))
    return {"index": index, "songs": songs}

# -----------------------------
# save to pickle
# Inputs: index object, FINGERPRINTS (db) path
# Output: n/a
def save_index(obj, path=FINGERPRINTS):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# -----------------------------
# load from pickle
# Inputs: FINGERPRINTS (db) path
# Output: pickle file
def load_index(path=FINGERPRINTS):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# find the closest match to the snippet
# Inputs: snippet_path - path to the input snippet | db - database of fingerprints
# Output: most likely song (song_id, votes, best_offset, total_hits)
def match_snippet(snippet_path, db):

    hashes = generate_hashes_from_file(snippet_path)
    if not hashes:
        return []

    index = db["index"]
    votes = defaultdict(lambda: defaultdict(int))  # song_id -> offset -> count

    for hash, t_query in hashes:
        if hash not in index:
            continue
        for song_id, t_song in index[hash]:
            delta = int(t_song) - int(t_query)
            votes[song_id][delta] += 1

    scored = []
    for song_id, offs in votes.items():
        best_delta, count = max(offs.items(), key=lambda x: x[1])
        scored.append((song_id, count, best_delta, sum(offs.values())))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:1]