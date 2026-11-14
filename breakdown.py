#!/usr/bin/env python3
"""Sound Breakdown: split a song into stems and describe what every part sounds like.

Usage:  python breakdown.py SONG_FILE [-o OUTPUT_DIR] [--no-stems]

Free / open-source only: Demucs (htdemucs_6s) for stems, MIT AST (AudioSet) for
instrument + genre + mood tags, librosa for tempo / key / beats / energy / timbre.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
import librosa

SR = 44100      # stem / stereo work
ASR = 22050     # analysis rate
HOP = 512
STEM_ORDER = ["drums", "bass", "vocals", "guitar", "piano", "other"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ----------------------------------------------------------------- helpers
def ts(sec):
    sec = max(0, float(sec))
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def load_audio(path):
    """Decode anything ffmpeg can read to 44.1 kHz stereo float."""
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "in.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "2",
                        "-ar", str(SR), "-f", "wav", wav], check=True)
        y, _ = sf.read(wav, dtype="float32", always_2d=True)
    return y.T  # (2, n)


def to_mono(y, sr_from=SR, sr_to=ASR):
    m = y.mean(axis=0) if y.ndim == 2 else y
    return librosa.resample(m, orig_sr=sr_from, target_sr=sr_to)


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))


# ----------------------------------------------------------------- stems
def separate(y, out_dir, save=True):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    torch.set_num_threads(os.cpu_count() or 4)
    model = get_model("htdemucs_6s")
    model.eval()
    wav = torch.from_numpy(y)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, ((wav - mean) / std)[None], device="cpu",
                          split=True, overlap=0.25, progress=True)[0]
    src = src * std + mean
    stems = {name: src[i].numpy() for i, name in enumerate(model.sources)}
    if save:
        sd = os.path.join(out_dir, "stems")
        os.makedirs(sd, exist_ok=True)
        for name, s in stems.items():
            sf.write(os.path.join(sd, f"{name}.wav"), s.T, SR, subtype="PCM_16")
    return stems


# ----------------------------------------------------------------- tagging (AST)
GENERIC_TAGS = {"Music", "Musical instrument", "Silence", "Inside, small room",
                "Sound effect", "Noise", "Background music", "Theme music", "Song",
                "Music of Bollywood"}  # last one is noisy; kept off the instrument list
MOOD_TAGS = {"Happy music", "Funny music", "Sad music", "Tender music", "Exciting music",
             "Angry music", "Scary music"}
GENRE_TAGS = {"Pop music", "Hip hop music", "Beatboxing", "Rock music", "Heavy metal",
              "Punk rock", "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock",
              "Rhythm and blues", "Soul music", "Reggae", "Country", "Swing music",
              "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco",
              "Classical music", "Opera", "Electronic music", "House music", "Techno",
              "Dubstep", "Drum and bass", "Electronica", "Electronic dance music",
              "Ambient music", "Trance music", "Music of Latin America", "Salsa music",
              "Flamenco", "Blues", "Music for children", "New-age music", "Vocal music",
              "A capella", "Music of Africa", "Afrobeat", "Christian music", "Gospel music",
              "Music of Asia", "Carnatic music", "Ska", "Traditional music",
              "Independent music", "Lullaby", "Video game music", "Christmas music",
              "Dance music", "Wedding music", "Soundtrack music", "Film score"}


class Tagger:
    def __init__(self):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        n = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(n)
        self.model = ASTForAudioClassification.from_pretrained(n).eval()
        self.labels = self.model.config.id2label

    def windows(self, y16, win=10.0, hop=10.0):
        """Return (start_times, probs[n_windows, 527])."""
        sr = 16000
        n = len(y16)
        starts = np.arange(0, max(1, n - int(2 * sr)), int(hop * sr))
        chunks = [y16[s:s + int(win * sr)] for s in starts]
        probs = []
        for i in range(0, len(chunks), 8):
            batch = chunks[i:i + 8]
            x = self.fe(batch, sampling_rate=sr, return_tensors="pt")
            with self.torch.no_grad():
                p = self.torch.sigmoid(self.model(**x).logits).numpy()
            probs.append(p)
        return starts / sr, np.concatenate(probs)

    SINGING = {"Singing", "Choir", "Yodeling", "Chant", "Male singing", "Female singing",
               "Child singing", "Synthetic singing", "Rapping", "Humming"}
    SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking",
              "Child speech, kid speaking", "Narration, monologue", "Conversation",
              "Speech synthesizer"}

    def kind_of(self, i):
        lab = self.labels[int(i)]
        if 139 <= i <= 215 or lab in self.SINGING:
            return "instrument"
        if lab in self.SPEECH:
            return "speech"
        if 137 <= i <= 282:
            return "music-meta"
        return "sfx"

    def top(self, probs, k=12, kind="all", min_p=0.05):
        mean = probs.mean(0)
        peak = np.percentile(probs, 90, axis=0)
        score = 0.5 * mean + 0.5 * peak
        out = []
        for i in np.argsort(score)[::-1]:
            lab = self.labels[int(i)]
            if lab in GENERIC_TAGS:
                continue
            if kind == "genre" and lab not in GENRE_TAGS:
                continue
            if kind == "mood" and lab not in MOOD_TAGS:
                continue
            if kind in ("instrument", "sfx", "speech") and self.kind_of(i) != kind:
                continue
            if score[i] < min_p:
                break
            out.append((lab, float(score[i])))
            if len(out) >= k:
                break
        return out


# ----------------------------------------------------------------- core analysis
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def relative_key(k):
    n, mode = k.split()
    i = NOTE_NAMES.index(n)
    return f"{NOTE_NAMES[(i + 9) % 12]} minor" if mode == "major" else f"{NOTE_NAMES[(i + 3) % 12]} major"


def detect_key(y_harm, bass_note=None):
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=ASR, hop_length=HOP * 4)
    prof = chroma.mean(axis=1)
    best = []
    for mode, tmpl in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        for t in range(12):
            r = np.corrcoef(prof, np.roll(tmpl, t))[0, 1]
            best.append((r, f"{NOTE_NAMES[t]} {mode}"))
    best.sort(reverse=True)
    if bass_note:  # relative major/minor share notes: prefer the one the bass keeps landing on
        top = best[0][0]
        for r, name in best[:4]:
            if top - r < 0.2 and name.split()[0] == bass_note:
                best.remove((r, name)); best.insert(0, (r, name)); break
    conf = abs(best[0][0] - best[1][0])
    return best[0][1], float(best[0][0]), best[1][1], float(conf)


def detect_tempo(drums_m, mix_m):
    src = drums_m if np.sqrt(np.mean(drums_m ** 2)) > 0.01 * np.sqrt(np.mean(mix_m ** 2) + 1e-12) else mix_m
    env = librosa.onset.onset_strength(y=src, sr=ASR, hop_length=256)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=ASR, hop_length=256)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_t = librosa.frames_to_time(beats, sr=ASR, hop_length=256)
    if len(beat_t) > 8:  # precise tempo from the slope of beat times (robust to frame rounding)
        d = np.diff(beat_t)
        med = np.median(d)
        ok = np.abs(d - med) < 0.15 * med
        if ok.sum() > 4:
            tempo = 60.0 / float(np.mean(d[ok]))
    feel = None
    if tempo > 150:
        feel = f"could also be felt as half-time ({tempo / 2:.0f} BPM)"
    elif tempo < 80:
        feel = f"could also be felt as double-time ({tempo * 2:.0f} BPM)"
    return tempo, beat_t, feel


def find_downbeat_phase(beat_t, kick_env, env_times):
    """Pick which beat (0-3) is beat 1 of the bar: the one with the strongest kicks."""
    if len(beat_t) < 8:
        return 0
    vals = np.interp(beat_t, env_times, kick_env)
    scores = [vals[p::4].mean() for p in range(4)]
    return int(np.argmax(scores))


def drum_patterns(drums_m, beat_t):
    """Return a 16-step pattern per drum part, averaged over all bars."""
    S = np.abs(librosa.stft(drums_m, n_fft=1024, hop_length=128)) ** 2
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=1024)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=ASR, hop_length=128)
    bands = {"Kick": (30, 120), "Snare/Clap": (400, 4000), "Hi-hats/Cymbals": (7000, 11000)}
    envs = {}
    for name, (lo, hi) in bands.items():
        e = np.sqrt(S[(freqs >= lo) & (freqs < hi)].sum(axis=0))
        flux = np.maximum(0, np.diff(e, prepend=e[0]))
        envs[name] = flux / (np.percentile(flux, 99.5) + 1e-9)
    if len(beat_t) < 9:
        return None, envs, times
    phase = find_downbeat_phase(beat_t, envs["Kick"], times)
    b = beat_t[phase:]
    n_bars = (len(b) - 1) // 4
    if n_bars < 2:
        return None, envs, times
    grid = []
    for bar in range(n_bars):
        for beat in range(4):
            t0, t1 = b[bar * 4 + beat], b[bar * 4 + beat + 1]
            for s_ in range(4):
                grid.append(t0 + (t1 - t0) * s_ / 4)
    grid = np.array(grid).reshape(n_bars, 16)
    win = min(0.04, np.median(np.diff(beat_t)) / 4 * 0.35)
    Ms = {}
    for name, e in envs.items():
        M = np.zeros((n_bars, 16))
        for i in range(n_bars):
            for j in range(16):
                lo = np.searchsorted(times, grid[i, j] - win)
                hi = np.searchsorted(times, grid[i, j] + win)
                M[i, j] = e[lo:hi].max() if hi > lo else 0
        Ms[name] = M
    pats = {}
    kick_rel = None
    for name in ("Kick", "Snare/Clap", "Hi-hats/Cymbals"):
        M = Ms[name]
        active = M.max(axis=1) > 0.2
        if active.sum() < 2:
            continue
        avg = M[active].mean(axis=0)
        rel = avg / (avg.max() + 1e-9)
        if name == "Kick":
            kick_rel = rel
        elif name == "Snare/Clap" and kick_rel is not None:
            rel = np.clip(rel - 0.35 * kick_rel, 0, None)
            rel = rel / (rel.max() + 1e-9)
        hits = float((rel > 0.3).sum())
        pat = "".join("X" if v > 0.6 else ("x" if v > 0.3 else ".") for v in rel)
        pats[name] = {"pattern": " ".join(pat[i:i + 4] for i in range(0, 16, 4)),
                      "hits_per_bar": round(hits, 1),
                      "bars_playing_pct": round(100 * active.mean())}
    return pats, envs, times


def energy_curve(mix_m):
    rms = librosa.feature.rms(y=mix_m, hop_length=HOP)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=ASR, hop_length=HOP)
    # 1-second smoothing
    w = max(1, int(ASR / HOP))
    sm = np.convolve(rms, np.ones(w) / w, mode="same")
    sm_db = db(sm)
    lo, hi = np.percentile(sm_db, 5), np.percentile(sm_db, 99)
    norm = np.clip((sm_db - lo) / (hi - lo + 1e-9), 0, 1)
    return t, norm


def segment(mix_m, stems_m, beat_t, duration, tempo=None, forced=()):
    chroma = librosa.feature.chroma_cqt(y=mix_m, sr=ASR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=mix_m, sr=ASR, hop_length=HOP, n_mfcc=13)
    n = chroma.shape[1]
    stem_rows = []
    for k in STEM_ORDER:
        r = librosa.feature.rms(y=stems_m[k], hop_length=HOP)[0][:n]
        r = np.pad(r, (0, n - len(r)))
        stem_rows.append(np.clip((db(r) + 60) / 60, 0, 1) * 2.0)
    feat = np.vstack([chroma, librosa.util.normalize(mfcc, axis=1) * 0.5, np.array(stem_rows)])
    times = librosa.frames_to_time(np.arange(n), sr=ASR, hop_length=HOP)
    # unit = 1 bar when beats are known, else 2 seconds
    if len(beat_t) > 16 and tempo:
        bl = 4 * 60.0 / tempo
        a = beat_t[0] % bl
        bars = np.arange(a, duration, bl)
        if a > 0.3:
            bars = np.concatenate([[0.0], bars])
    else:
        bars = np.arange(0, duration, 2.0)
    bf = librosa.time_to_frames(bars, sr=ASR, hop_length=HOP)
    bf = bf[bf < n]
    sync = librosa.util.sync(feat, bf, aggregate=np.mean)
    k = int(np.clip(round(duration / 16), 3, 14))
    k = min(k, sync.shape[1] - 1)
    bounds = librosa.segment.agglomerative(sync, k)
    edge_frames = np.concatenate([[0], bf])
    bt = sorted(set([0.0] + [float(times[min(edge_frames[i], n - 1)]) for i in bounds] + [duration]))
    unit = float(np.median(np.diff(bars))) if len(bars) > 2 else 2.0
    # arrangement changes (instruments in/out, drops) are always section edges, snapped to a bar
    for f in forced:
        j = int(np.argmin(np.abs(bars - f)))
        snap = float(bars[j]) if abs(bars[j] - f) < unit * 0.6 else float(f)
        bt = [x for x in bt if abs(x - snap) >= max(6, 2 * unit) * 0.75 or x in (0.0, duration)]
        bt.append(snap)
    bt = sorted(set(bt))
    merged = [bt[0]]
    for x in bt[1:]:
        if x - merged[-1] < max(6, 2 * unit) and x != duration:
            continue
        merged.append(x)
    if len(merged) > 2 and merged[-1] - merged[-2] < max(6, 2 * unit):
        merged.pop(-2)
    vecs = []
    for a, b in zip(merged[:-1], merged[1:]):
        sel = (times >= a) & (times < b)
        v = feat[:, sel].mean(1)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    letters, protos = [], []
    for v in vecs:
        sims = [float(v @ p) for p in protos]
        if sims and max(sims) > 0.97:
            letters.append(chr(65 + int(np.argmax(sims))))
        else:
            protos.append(v)
            letters.append(chr(65 + len(protos) - 1))
    return list(zip(merged[:-1], merged[1:], letters))


def measure_tuning(stems_m, stem_info):
    """Cents away from A4 = 440 Hz, from the strongest tonal peaks of the pitched stems."""
    per = {}
    for name in ("guitar", "other", "piano", "vocals", "bass"):
        if not stem_info[name].get("present"):
            continue
        y = stems_m[name]
        lo, hi, nfft = (30, 300, 8192) if name == "bass" else (80, 2000, 4096)
        p, m = librosa.piptrack(y=y, sr=ASR, n_fft=nfft, hop_length=1024, fmin=lo, fmax=hi)
        if not (m > 0).any():
            continue
        sel = m > np.percentile(m[m > 0], 90)
        pv = p[sel]
        pv = pv[pv > 0]
        if len(pv) < 50:
            continue
        per[name] = float(librosa.pitch_tuning(pv, resolution=0.01) * 100)
    if not per:
        return None
    # bass is least reliable (few cycles per window): use it only if nothing else exists
    main = [v for k, v in per.items() if k not in ("bass", "vocals")] or list(per.values())
    cents = float(np.median(main))
    spread = float(np.max(main) - np.min(main)) if len(main) > 1 else 0.0
    a4 = 440 * 2 ** (cents / 1200)
    if abs(cents) <= 10:
        verdict = "standard concert tuning (A = 440 Hz)"
    elif abs(a4 - 432) < 2:
        verdict = "A = 432 Hz tuning"
    elif abs(a4 - 442) < 1.5 or abs(a4 - 443) < 1.5:
        verdict = f"orchestral / bright tuning (A ≈ {a4:.0f} Hz)"
    else:
        verdict = f"non-standard tuning ({cents:+.0f} cents, A ≈ {a4:.1f} Hz), maybe pitch-shifted or vintage tape"
    return {"cents_from_440": round(cents, 1), "a4_hz": round(a4, 1), "verdict": verdict,
            "per_stem_cents": {k: round(v, 1) for k, v in per.items()},
            "agreement": "high" if spread < 12 else "medium" if spread < 25 else "low"}


def guitar_tuning(guitar_m, key):
    """Guess the guitar's tuning from the lowest notes it actually plays."""
    C = np.abs(librosa.cqt(guitar_m, sr=ASR, hop_length=512, fmin=librosa.note_to_hz("A1"), n_bins=36))
    Cd = librosa.amplitude_to_db(C, ref=np.max)
    low = []
    for j in range(C.shape[1]):
        col = Cd[:, j]
        if col.max() > -30:
            low.append(int(np.where(col > col.max() - 10)[0][0]))
    if len(low) < 100:
        return None
    low = np.array(low)
    base = librosa.note_to_midi("A1")
    share = lambda note: float((low == librosa.note_to_midi(note) - base).mean())
    below_e = float((low < librosa.note_to_midi("E2") - base).mean())
    d2, cs2, c2 = share("D2"), share("C#2"), share("C2")
    lowest = librosa.midi_to_note(base + int(np.percentile(low, 3)))
    if below_e < 0.08:
        guess, conf = "standard tuning (E A D G B E)", "medium"
    elif below_e < 0.22 and d2 < 0.08:
        guess, conf = "probably standard tuning (the few notes below low E look like bass bleeding into the guitar stem)", "low"
    elif d2 >= max(cs2, c2) and d2 > 0.05:
        guess = "drop D (D A D G B E) or D standard"
        conf = "medium" if key and key.startswith("D") else "low"
    elif c2 > 0.05:
        guess, conf = "drop C / C standard (down-tuned)", "low"
    else:
        guess, conf = "lower than standard, but unclear which tuning", "low"
    return {"guess": guess, "confidence": conf, "lowest_note": lowest,
            "share_below_E2_pct": round(100 * below_e, 1)}


# ----------------------------------------------------------------- per-stem timbre
def describe_stem(name, stem, mix_rms):
    m = to_mono(stem)
    rms_all = np.sqrt(np.mean(m ** 2) + 1e-12)
    rel_db = float(db(rms_all) - db(mix_rms))
    rms = librosa.feature.rms(y=m, hop_length=HOP)[0]
    rms_db = db(rms)
    thr = max(np.percentile(rms_db, 95) - 25, -55)
    active = rms_db > thr
    info = {"level_vs_mix_db": round(rel_db, 1), "active_pct": round(100 * active.mean())}
    if rel_db < -24 or active.mean() < 0.03:
        info["present"] = False
        return info, rms, active
    info["present"] = True
    S = np.abs(librosa.stft(m, n_fft=2048, hop_length=HOP))[:, active] if active.any() else None
    if S is None or S.shape[1] == 0:
        info["present"] = False
        return info, rms, active
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=2048)
    cent = float(np.median(librosa.feature.spectral_centroid(S=S, sr=ASR)))
    flat = float(np.median(librosa.feature.spectral_flatness(S=S)))
    roll = float(np.median(librosa.feature.spectral_rolloff(S=S, sr=ASR, roll_percent=0.9)))
    power = (S ** 2).sum(axis=1)
    def share(lo, hi):
        return float(power[(freqs >= lo) & (freqs < hi)].sum() / (power.sum() + 1e-12))
    sub, low, mid, high = share(20, 60), share(60, 250), share(250, 4000), share(4000, 11025)
    onsets = librosa.onset.onset_detect(y=m, sr=ASR, hop_length=HOP, units="time")
    act_sec = active.sum() * HOP / ASR
    density = len(onsets) / max(act_sec, 1)
    a = rms[active]
    crest = float(np.percentile(a, 95) / (np.median(a) + 1e-9))
    dyn = float(np.percentile(rms_db[active], 95) - np.percentile(rms_db[active], 10))
    # stereo width from side / mid energy
    mid_s, side_s = (stem[0] + stem[1]) / 2, (stem[0] - stem[1]) / 2
    width = float(np.sqrt(np.mean(side_s ** 2)) / (np.sqrt(np.mean(mid_s ** 2)) + 1e-9))
    pan = float(db(np.sqrt(np.mean(stem[1] ** 2))) - db(np.sqrt(np.mean(stem[0] ** 2))))

    words = []
    if name == "bass":
        words.append("deep sub-heavy (808-like)" if sub > 0.35 else
                     "round, warm low end" if cent < 350 else
                     "punchy, mid-forward bass" if cent < 800 else "growly / gritty, harmonically rich")
    elif name == "drums":
        words.append("hard-hitting low end (big kick)" if (sub + low) > 0.45 else "tight, light low end")
        words.append("crisp, bright top (prominent hats/cymbals)" if cent > 3500 else
                     "balanced top end" if cent > 1800 else "dark / muted top end")
    else:
        words.append("dark, muffled" if cent < 900 else "warm" if cent < 1800 else
                     "balanced" if cent < 3000 else "bright" if cent < 4500 else "very bright, airy")
    if name not in ("drums",):
        words.append("noisy / distorted / breathy texture" if flat > 0.12 else
                     "slightly gritty" if flat > 0.04 else "clean, pure tone")
    words.append("plucky, percussive (short notes)" if crest > 2.5 else
                 "rhythmic with sustain" if crest > 1.6 else "smooth, sustained (pad-like / held notes)")
    words.append("sparse" if density < 1 else "steady" if density < 3 else
                 "busy" if density < 6 else "very busy / rapid notes")
    words.append("mono / dead centre" if width < 0.08 else "slightly wide" if width < 0.3 else
                 "wide stereo" if width < 0.6 else "very wide / spacious")
    if abs(pan) > 2:
        words.append(f"leaning {'right' if pan > 0 else 'left'}")
    words.append("heavily compressed, consistent level" if dyn < 6 else
                 "moderately dynamic" if dyn < 14 else "very dynamic (big loud/soft swings)")

    info.update({"brightness_hz": round(cent), "rolloff_hz": round(roll), "noisiness": round(flat, 3),
                 "notes_per_sec": round(density, 2), "stereo_width": round(width, 2),
                 "pan_db_right_minus_left": round(pan, 1), "dynamic_range_db": round(dyn, 1),
                 "spectrum_share": {"sub<60": round(sub, 2), "low60-250": round(low, 2),
                                    "mid250-4k": round(mid, 2), "high>4k": round(high, 2)},
                 "sound": words})

    if name in ("guitar", "piano", "other"):
        ch = librosa.feature.chroma_cqt(y=m, sr=ASR, hop_length=HOP * 4)
        act4 = librosa.util.fix_length(active[::4], size=ch.shape[1])
        if act4.any():
            prof = ch[:, act4].mean(1)
            info["most_used_notes"] = [NOTE_NAMES[i] for i in np.argsort(prof)[::-1][:4]]
    if name in ("bass", "vocals") and flat < 0.2:
        fmin, fmax = (30, 400) if name == "bass" else (70, 1400)
        f0 = librosa.yin(m, fmin=fmin, fmax=fmax, sr=ASR, hop_length=HOP * 2)
        act2 = librosa.util.fix_length(active[::2], size=len(f0))
        f = f0[act2]
        if len(f) > 20:
            lo, hi = np.percentile(f, 10), np.percentile(f, 90)
            info["pitch_range"] = f"{librosa.hz_to_note(lo)} – {librosa.hz_to_note(hi)}"
            midi = np.round(librosa.hz_to_midi(f)).astype(int) % 12
            counts = np.bincount(midi, minlength=12)
            info["most_used_notes"] = [NOTE_NAMES[i] for i in np.argsort(counts)[::-1][:4]]
    return info, rms, active


def entrances(stem_rms, duration, min_gap=4.0):
    """Times where each stem comes in / drops out (after 1-second smoothing)."""
    events = []
    for name, rms in stem_rms.items():
        w = max(1, int(ASR / HOP))
        sm = np.convolve(rms, np.ones(w) / w, mode="same")
        sdb = db(sm)
        on = sdb > max(np.percentile(sdb, 95) - 22, -55)
        t = librosa.frames_to_time(np.arange(len(on)), sr=ASR, hop_length=HOP)
        state, last = False, -99
        for i in range(0, len(on), w // 2 or 1):
            seg = on[i:i + w * 2]
            now = seg.mean() > 0.6
            if now != state and t[i] - last >= min_gap:
                events.append((float(t[i]), name, "in" if now else "out"))
                state, last = now, t[i]
    events.sort()
    return events


def peaks_and_drops(t, energy, duration):
    from scipy.signal import find_peaks
    step = max(1, int(ASR / HOP))
    e = energy[::step]
    tt = t[::step]
    pk, _ = find_peaks(e, prominence=0.12, distance=8)
    peaks = sorted([(float(tt[i]), float(e[i])) for i in pk], key=lambda x: -x[1])[:5]
    # drops: big jumps up within ~2 s after a quieter stretch
    drops, dips = [], []
    for i in range(4, len(e) - 2):
        before = e[max(0, i - 4):i].mean()
        after = e[i:i + 2].mean()
        if after - before > 0.28 and (not drops or tt[i] - drops[-1][0] > 8):
            drops.append((float(tt[i]), float(after - before)))
        if before - after > 0.28 and (not dips or tt[i] - dips[-1][0] > 8):
            dips.append((float(tt[i]), float(before - after)))
    return sorted(peaks), drops, dips


# ----------------------------------------------------------------- plotting
def plot(out, t, energy, sections, peaks, drops, stem_rms, duration, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in STEM_ORDER if n in stem_rms]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.4]})
    cols = plt.cm.Pastel1(np.linspace(0, 1, 9))
    for i, (s, e, lab) in enumerate(sections):
        a1.axvspan(s, e, color=cols[(ord(lab) - 65) % 9], alpha=0.6, lw=0)
        a1.text((s + e) / 2, 1.04, f"{lab}", ha="center", fontsize=9, weight="bold")
    a1.plot(t, energy, color="#222", lw=1.3)
    for pt, pv in peaks:
        a1.plot(pt, pv, "v", color="#d62728", ms=9)
        a1.annotate(f"peak {ts(pt)}", (pt, pv), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#d62728")
    for dt, _ in drops:
        a1.axvline(dt, color="#1f77b4", ls="--", lw=1)
        a1.text(dt, 0.02, f" drop {ts(dt)}", color="#1f77b4", fontsize=8, rotation=90, va="bottom")
    a1.set_ylim(0, 1.12)
    a1.set_ylabel("energy")
    a1.set_title(title, loc="left")
    rows = []
    for n in names:
        r = stem_rms[n]
        w = max(1, int(ASR / HOP))
        sm = db(np.convolve(r, np.ones(w) / w, mode="same"))
        rows.append(sm)
    L = min(len(r) for r in rows)
    M = np.array([r[:L] for r in rows])
    M = np.clip((M + 60) / 60, 0, 1)
    a2.imshow(M, aspect="auto", cmap="magma", extent=[0, duration, len(names) - 0.5, -0.5],
              interpolation="nearest")
    a2.set_yticks(range(len(names)))
    a2.set_yticklabels(names)
    a2.set_xlabel("time")
    xt = np.arange(0, duration + 1, 15 if duration < 240 else 30)
    a2.set_xticks(xt)
    a2.set_xticklabels([ts(x) for x in xt])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- report
def section_role(i, n, e_mean, all_means, rising):
    hi, lo = np.percentile(all_means, 70), np.percentile(all_means, 35)
    if i == 0 and e_mean < hi:
        return "Intro"
    if i == n - 1 and e_mean < hi:
        return "Outro"
    if e_mean >= hi:
        return "Peak (chorus / drop)"
    if rising:
        return "Build-up"
    if e_mean <= lo:
        return "Breakdown / quiet part"
    return "Verse / groove"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--no-stems", action="store_true", help="skip saving stem WAVs")
    args = ap.parse_args()
    base = os.path.splitext(os.path.basename(args.song))[0]
    out = args.out or os.path.join("breakdowns", base)
    os.makedirs(out, exist_ok=True)

    print("[1/6] loading audio"); sys.stdout.flush()
    y = load_audio(args.song)
    duration = y.shape[1] / SR
    mix_m = to_mono(y)
    mix_rms = np.sqrt(np.mean(mix_m ** 2) + 1e-12)

    print("[2/6] splitting into stems (Demucs 6-stem, takes a few minutes on CPU)"); sys.stdout.flush()
    stems = separate(y, out, save=not args.no_stems)
    stems_m = {k: to_mono(v) for k, v in stems.items()}

    print("[3/6] tempo, key, beats, drum patterns"); sys.stdout.flush()
    tempo, beat_t, feel = detect_tempo(stems_m["drums"], mix_m)
    harm = sum(v for k, v in stems_m.items() if k not in ("drums",))
    pats, _, _ = drum_patterns(stems_m["drums"], beat_t)

    print("[4/6] energy curve, sections, peaks"); sys.stdout.flush()
    t, energy = energy_curve(mix_m)
    peaks, drops, dips = peaks_and_drops(t, energy, duration)

    print("[5/6] describing each instrument"); sys.stdout.flush()
    stem_info, stem_rms = {}, {}
    for name in STEM_ORDER:
        info, rms, _ = describe_stem(name, stems[name], mix_rms)
        stem_info[name] = info
        stem_rms[name] = rms
    bass_note = (stem_info["bass"].get("most_used_notes") or [None])[0]
    key, key_r, key_alt, key_conf = detect_key(harm, bass_note)
    events = [e for e in entrances({k: v for k, v in stem_rms.items() if stem_info[k]["present"]}, duration)
              if e[0] > 1.0]

    big = [e for e in events if e[1] in ("drums", "bass", "vocals")]
    forced = sorted(set([round(e[0], 1) for e in big] + [d for d, _ in drops] + [d for d, _ in dips]))
    sections = segment(mix_m, stems_m, beat_t, duration, tempo, forced)

    print("[6/6] recognising instruments, genre and mood (AST)"); sys.stdout.flush()
    tagger = Tagger()
    r16 = lambda x: librosa.resample(x, orig_sr=ASR, target_sr=16000)
    _, probs_mix = tagger.windows(r16(mix_m))
    instrumental = sum(v for k, v in stems_m.items() if k != "vocals")
    win_t, probs = tagger.windows(r16(instrumental))  # voice removed, so instruments aren't drowned out
    instruments = tagger.top(probs, k=14, kind="instrument", min_p=0.02)
    genres = tagger.top(probs, k=5, kind="genre", min_p=0.015)
    moods = tagger.top(probs, k=3, kind="mood", min_p=0.01)
    sfx = []
    seen = {}
    for wi, wt in enumerate(win_t):
        for i in np.argsort(probs_mix[wi])[::-1][:12]:
            if tagger.kind_of(i) == "sfx" and probs_mix[wi, i] > 0.04 and tagger.labels[int(i)] not in GENERIC_TAGS:
                seen.setdefault(tagger.labels[int(i)], []).append(float(wt))
    for lab, tl in sorted(seen.items(), key=lambda x: x[1][0]):
        sfx.append((lab, tl))
    voice_type = None
    if stem_info["vocals"]["present"]:
        _, p_v = tagger.windows(r16(stems_m["vocals"]))
        lab = {v: k for k, v in tagger.labels.items()}
        sp = float(np.percentile(p_v[:, [lab[x] for x in tagger.SPEECH if x in lab]].max(1), 75))
        sg = float(np.percentile(p_v[:, [lab[x] for x in tagger.SINGING if x in lab]].max(1), 75))
        voice_type = ("spoken voice (narration / voiceover)" if sp > sg * 1.5 else
                      "singing" if sg > sp * 1.5 else "mix of singing and speech")
    tuning = measure_tuning(stems_m, stem_info)
    gtune = guitar_tuning(stems_m["guitar"], key) if stem_info["guitar"]["present"] else None
    other_tags = []
    if stem_info["other"]["present"]:
        _, p_o = tagger.windows(r16(stems_m["other"]))
        other_tags = tagger.top(p_o, k=6, kind="instrument", min_p=0.05)
    stem_check = {}
    for n in ("guitar", "piano", "bass", "drums"):
        if stem_info[n]["present"]:
            _, p_s = tagger.windows(r16(stems_m[n]))
            stem_check[n] = tagger.top(p_s, k=4, kind="instrument", min_p=0.05)

    merged_inst = dict(instruments)
    for tags in [other_tags] + list(stem_check.values()):
        for n, pr in tags:
            if tagger.kind_of({v: k for k, v in tagger.labels.items()}[n]) == "instrument":
                merged_inst[n] = max(merged_inst.get(n, 0), pr)
    instruments = sorted(merged_inst.items(), key=lambda x: -x[1])[:14]

    # per-section summary
    sec_rows = []
    means = [energy[(t >= s) & (t < e)].mean() for s, e, _ in sections]
    for i, (s, e, lab) in enumerate(sections):
        sel = (t >= s) & (t < e)
        seg_e = energy[sel]
        half = len(seg_e) // 2
        rising = half > 0 and seg_e[half:].mean() - seg_e[:half].mean() > 0.12
        playing = []
        for n in STEM_ORDER:
            if not stem_info[n]["present"]:
                continue
            r = stem_rms[n]
            ft = librosa.frames_to_time(np.arange(len(r)), sr=ASR, hop_length=HOP)
            rs = r[(ft >= s) & (ft < e)]
            if len(rs) and db(np.percentile(rs, 75)) > db(np.percentile(r, 95)) - 20:
                playing.append(n)
        wsel = (win_t + 5 >= s) & (win_t + 5 < e)
        tags = tagger.top(probs[wsel], k=3, kind="instrument", min_p=0.05) if wsel.any() else []
        sec_rows.append({"start": s, "end": e, "part": lab,
                         "role": section_role(i, len(sections), means[i], means, rising),
                         "energy": round(float(means[i]), 2), "playing": playing,
                         "heard": [x[0] for x in tags]})

    # ---------- write report
    present = [n for n in STEM_ORDER if stem_info[n]["present"]]
    title = f"{base} · {tempo:.0f} BPM · {key} · {ts(duration)}"
    plot(os.path.join(out, "timeline.png"), t, energy, sections, peaks, drops, stem_rms, duration, title)

    L = [f"# Sound breakdown: {base}", ""]
    L += ["## Overview", "",
          f"- **Length:** {ts(duration)}",
          f"- **Tempo:** {tempo:.0f} BPM" + (f" ({feel})" if feel else ""),
          f"- **Key:** {key} (same notes as {relative_key(key)})" + (f"; could also be {key_alt}" if key_conf < 0.03 and key_alt != relative_key(key) else ""),
          f"- **Genre (model guess):** " + (", ".join(f"{g} ({p:.0%})" for g, p in genres) or "unclear"),
          f"- **Mood (model guess):** " + (", ".join(f"{m}" for m, _ in moods) or "unclear"),
          f"- **Instruments heard (model, voice removed):** " + (", ".join(f"{n} ({p:.0%})" for n, p in instruments) or "unclear"),
          f"- **Voice:** " + (voice_type or "no voice"),
          f"- **Sound design / non-music sounds:** " + ("; ".join(f"{n} ({', '.join(ts(x) for x in tl[:3])})" for n, tl in sfx[:25]) or "none detected"),
          ""]
    L += ["## Tuning", ""]
    if tuning:
        L.append(f"- **Reference pitch:** A4 ≈ {tuning['a4_hz']} Hz ({tuning['cents_from_440']:+.0f} cents from 440), so **{tuning['verdict']}**. Agreement between instruments: {tuning['agreement']} "
                 f"({', '.join(f'{k} {v:+.0f}¢' for k, v in tuning['per_stem_cents'].items())})")
    else:
        L.append("- Reference pitch: not enough pitched sound to measure")
    if gtune:
        L.append(f"- **Guitar tuning (guess):** {gtune['guess']} (confidence: {gtune['confidence']}; lowest note played ≈ {gtune['lowest_note']}, {gtune['share_below_E2_pct']}% of notes below standard low E)")
    L.append("")
    L += ["## Song map (where the heights are)", "", "![timeline](timeline.png)", "",
          "| Time | Part | What it is | Energy | Playing |", "|---|---|---|---|---|"]
    for r in sec_rows:
        bar = "█" * int(round(r["energy"] * 10)) + "░" * (10 - int(round(r["energy"] * 10)))
        L.append(f"| {ts(r['start'])}–{ts(r['end'])} | {r['part']} | {r['role']} | {bar} | "
                 f"{', '.join(r['playing']) or '—'} |")
    L += ["", "Parts with the same letter sound alike (e.g. two choruses).", ""]
    if peaks:
        L.append("**Highest points:** " + ", ".join(f"{ts(p)}" for p, _ in sorted(peaks, key=lambda x: -x[1])))
    if drops:
        L.append("\n**Drops / big hits (energy jumps up):** " + ", ".join(ts(d) for d, _ in drops))
    if dips:
        L.append("\n**Pull-backs (energy falls away):** " + ", ".join(ts(d) for d, _ in dips))
    if events:
        L += ["", "**Arrangement moves:**", ""]
        for tt_, n, w in events[:40]:
            L.append(f"- {ts(tt_)} — {n} {'comes in' if w == 'in' else 'drops out'}")
    L.append("")

    L += ["## The beat", ""]
    if pats:
        L += [f"One bar of 4/4 in 16th notes (X = strong hit, x = softer hit, . = nothing), averaged over the song:", "",
              "```", "                 1... 2... 3... 4..."]
        for n, p in pats.items():
            L.append(f"{n:<17}{p['pattern']}   ~{p['hits_per_bar']} hits/bar, in {p['bars_playing_pct']}% of bars")
        L += ["```", ""]
    else:
        L += ["No steady drum pattern found (the song may have no drums or a free tempo).", ""]

    L += ["## Each instrument, what it sounds like", ""]
    for n in STEM_ORDER:
        i = stem_info[n]
        if not i["present"]:
            L.append(f"### {n.capitalize()}: not really in this song (only faint bleed)\n")
            continue
        L.append(f"### {n.capitalize()}")
        L.append(f"- **Sound:** {', '.join(i['sound'])}")
        L.append(f"- **Loudness in the mix:** {i['level_vs_mix_db']:+.1f} dB vs full song · playing {i['active_pct']}% of the time")
        sp = i["spectrum_share"]
        L.append(f"- **Where its energy sits:** sub {sp['sub<60']:.0%} · low {sp['low60-250']:.0%} · mid {sp['mid250-4k']:.0%} · high {sp['high>4k']:.0%} (brightness ≈ {i['brightness_hz']} Hz)")
        if "pitch_range" in i:
            L.append(f"- **Pitch range:** {i['pitch_range']} · most used notes: {', '.join(i['most_used_notes'])}")
        elif "most_used_notes" in i:
            L.append(f"- **Most used notes (chords/harmony):** {', '.join(i['most_used_notes'])}")
        if n == "other" and other_tags:
            L.append(f"- **What the 'other' stem holds (model):** " + ", ".join(f"{a} ({p:.0%})" for a, p in other_tags))
        if n in stem_check:
            L.append(f"- **Model check on this stem:** " + (", ".join(f"{a} ({p:.0%})" for a, p in stem_check[n]) or "no clear instrument; may be bleed from other parts"))
        L.append("")

    # prompt-ready description
    desc_parts = [f"{', '.join(g for g, _ in genres[:2]) or 'Instrumental'} track at {tempo:.0f} BPM in {key}"
                  + (f", {tuning['verdict']}" if tuning else "")]
    if instruments:
        desc_parts.append("instruments: " + ", ".join(n for n, _ in instruments[:6]))
    if voice_type:
        desc_parts.append(f"voice: {voice_type}")
    if moods:
        desc_parts.append(f"mood: {moods[0][0].replace(' music', '').lower()}")
    for n in present:
        desc_parts.append(f"{n}: {', '.join(stem_info[n]['sound'][:3])}")
    if pats and "Kick" in pats:
        desc_parts.append(f"kick pattern {pats['Kick']['pattern']}")
    roles = " → ".join(dict.fromkeys(r["role"].split(" (")[0] for r in sec_rows))
    desc_parts.append(f"structure: {roles}")
    if peaks:
        desc_parts.append(f"biggest moment around {ts(max(peaks, key=lambda x: x[1])[0])}")
    L += ["## Quick description (paste into any AI)", "", "> " + "; ".join(desc_parts) + ".", ""]
    L += ["---", "Measured with Demucs htdemucs_6s, MIT AST (AudioSet) and librosa. Numbers are measured; "
          "genre/mood/instrument names and guitar tuning are model guesses. Stems are in `stems/`."]

    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L))
    data = {"file": args.song, "duration_sec": duration, "tempo_bpm": tempo, "key": key,
            "key_alternative": key_alt, "genres": genres, "moods": moods, "instruments": instruments,
            "sections": sec_rows, "peaks": peaks, "drops": drops, "pullbacks": dips,
            "arrangement_events": events, "drum_patterns": pats, "stems": stem_info,
            "other_stem_tags": other_tags, "stem_checks": stem_check,
            "tuning": tuning, "guitar_tuning": gtune, "voice": voice_type, "sound_design": sfx}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"\nDone → {out}/report.md")


if __name__ == "__main__":
    main()
