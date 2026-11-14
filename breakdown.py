#!/usr/bin/env python3
"""Sound Breakdown: split a song into stems and describe what every part sounds like.

Usage:  python breakdown.py SONG_FILE [-o OUTPUT_DIR] [--no-stems]

Free / open-source only: Demucs (htdemucs_6s) for stems, LAION CLAP to identify
instruments, genre and mood, MIT AST for voice and sound effects, librosa for
tempo / key / tuning / chords / beats / energy / timbre.
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

import sharp as S

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
        f = f[(f > fmin * 1.04) & (f < fmax * 0.96)]   # values stuck at the search limits are not real notes
        if len(f) > 20:
            lo, hi = np.percentile(f, 10), np.percentile(f, 90)
            info["pitch_range"] = f"{librosa.hz_to_note(lo)} – {librosa.hz_to_note(hi)}"
            midi = np.round(librosa.hz_to_midi(f)).astype(int) % 12
            counts = np.bincount(midi, minlength=12)
            info["most_used_notes"] = [NOTE_NAMES[i] for i in np.argsort(counts)[::-1][:4]]
    return info, rms, active


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
    if i == n - 1 and e_mean < np.median(all_means):
        return "Outro"
    if e_mean >= hi:
        return "Peak (chorus / drop)"
    if rising:
        return "Build-up"
    if e_mean <= lo:
        return "Breakdown / quiet part"
    return "Verse / groove"


def load_or_separate(y, out, save, fresh):
    sd = os.path.join(out, "stems")
    paths = {n: os.path.join(sd, f"{n}.wav") for n in STEM_ORDER}
    if not fresh and all(os.path.exists(p) for p in paths.values()):
        stems = {n: sf.read(p, dtype="float32", always_2d=True)[0].T for n, p in paths.items()}
        if all(abs(s.shape[1] - y.shape[1]) < SR // 10 for s in stems.values()):
            print("      (reusing stems already saved for this song; pass --fresh to redo)")
            return stems
    return separate(y, out, save=save)


BLEED_HINTS = {"piano": {"guitar", "strings"},
               "guitar": {"piano", "synth pluck", "pizzicato strings", "harp"}}
NOT_AN_INSTRUMENT = {"sound effects and noise, not drums", "sound effects and field recordings",
                     "vinyl crackle and noise"}
GENERIC = {"drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano"}
CONF_WORD = {"clear": "confident", "likely": "fairly confident", "unsure": "not sure"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--no-stems", action="store_true", help="skip saving stem WAVs")
    ap.add_argument("--fresh", action="store_true", help="redo stem separation even if stems exist")
    args = ap.parse_args()
    base = os.path.splitext(os.path.basename(args.song))[0]
    out = args.out or os.path.join("breakdowns", base)
    os.makedirs(out, exist_ok=True)
    step = lambda i, msg: (print(f"[{i}/7] {msg}"), sys.stdout.flush())

    step(1, "loading audio")
    y = load_audio(args.song)
    duration = y.shape[1] / SR
    mix_m = to_mono(y)
    mix_rms = np.sqrt(np.mean(mix_m ** 2) + 1e-12)

    step(2, "splitting into stems (Demucs 6-stem, about a minute on CPU)")
    stems = load_or_separate(y, out, not args.no_stems, args.fresh)
    stems_m = {k: to_mono(v) for k, v in stems.items()}

    step(3, "tempo, beats, bar lines, drum pattern")
    tempo, beat_t, feel = detect_tempo(stems_m["drums"], mix_m)
    pats, envs, env_t = drum_patterns(stems_m["drums"], beat_t)
    phase = find_downbeat_phase(beat_t, envs["Kick"], env_t)
    bars, bar_len, has_beat = S.bar_grid(beat_t, phase, duration)

    step(4, "energy curve, peaks and drops")
    t, energy = energy_curve(mix_m)
    peaks, drops, dips = peaks_and_drops(t, energy, duration)

    step(5, "measuring each instrument, key, tuning, chords, sections")
    stem_info, stem_rms = {}, {}
    for name in STEM_ORDER:
        info, rms, _ = describe_stem(name, stems[name], mix_rms)
        stem_info[name], stem_rms[name] = info, rms
    present = [n for n in STEM_ORDER if stem_info[n]["present"]]
    harm_all = sum(v for k, v in stems_m.items() if k != "drums")
    bass_note = (stem_info["bass"].get("most_used_notes") or [None])[0]
    key_raw, _, key_alt, key_conf = detect_key(harm_all, bass_note)
    key = S.pretty_key(key_raw)
    tuning = measure_tuning(stems_m, stem_info)
    gtune = guitar_tuning(stems_m["guitar"], key_raw) if stem_info["guitar"]["present"] else None

    edges = np.concatenate([bars, [duration]])
    half = bar_len > 2.6                      # slow songs: two chords per bar can matter
    c_edges = np.sort(np.concatenate([edges, (edges[:-1] + edges[1:]) / 2])) if half else edges
    chords = S.chord_track(stems_m, c_edges, key_raw)
    cu = 2 if half else 1
    Ch = librosa.feature.chroma_cqt(y=harm_all, sr=ASR, hop_length=HOP * 2)
    cft = librosa.frames_to_time(np.arange(Ch.shape[1]), sr=ASR, hop_length=HOP * 2)
    chroma_bar = S.per_unit(Ch, cft, edges)
    chroma_bar = chroma_bar / (chroma_bar.max(axis=0, keepdims=True) + 1e-9)
    sections = S.find_sections(stem_rms, present, energy, t, chroma_bar, bars, bar_len, duration)
    mix = S.mix_report(y, SR)

    step(6, "listening: which instruments, genre, mood (CLAP) + voice and sound effects (AST)")
    tagger = Tagger()
    r16 = lambda x: librosa.resample(x, orig_sr=ASR, target_sr=16000)
    win_t, probs_mix = tagger.windows(r16(mix_m))
    seen = {}
    for wi, wt in enumerate(win_t):
        for i in np.argsort(probs_mix[wi])[::-1][:12]:
            if tagger.kind_of(i) == "sfx" and probs_mix[wi, i] > 0.04 and tagger.labels[int(i)] not in GENERIC_TAGS:
                seen.setdefault(tagger.labels[int(i)], []).append(float(wt))
    sfx = sorted(seen.items(), key=lambda x: x[1][0])
    voice_type, singing = None, False
    if "vocals" in present:
        _, p_v = tagger.windows(r16(stems_m["vocals"]))
        lab = {v: k for k, v in tagger.labels.items()}
        sp = float(np.percentile(p_v[:, [lab[x] for x in tagger.SPEECH if x in lab]].max(1), 75))
        sg = float(np.percentile(p_v[:, [lab[x] for x in tagger.SINGING if x in lab]].max(1), 75))
        voice_type = ("spoken voice (narration / voiceover)" if sp > sg * 1.5 else
                      "singing" if sg > sp * 1.5 else "mix of singing and speech")
        singing = not voice_type.startswith("spoken")

    spoken = voice_type is not None and not singing
    ear = S.Listener()
    # a narrator would be mistaken for part of the music, so judge genre without the voice stem
    genre_src = sum(v for k, v in stems_m.items() if k != "vocals") if spoken else mix_m
    _, pm = ear.listen(genre_src, ASR, ["genre", "mood", "production"])
    genres = S.rank(pm["genre"], S.VOCAB["genre"], 4)
    moods = S.rank(pm["mood"], S.VOCAB["mood"], 3)
    production = S.rank(pm["production"], S.VOCAB["production"], 2)

    skip = {"vocal chops and vocal samples"} if spoken else set()   # that would be narrator bleed
    ident, layers = {}, []
    for n in present:
        if n == "vocals" and not singing:
            continue
        wt, pr = ear.listen(stems_m[n], ASR, [n])
        if wt is None:
            continue
        ident[n] = S.identify(pr[n], n, skip)
        if n == "other":
            layers = S.layers_over_time(wt, pr[n], S.VOCAB[n], skip=skip)
    vocal_style = ident["vocals"]["detail"] if singing and "vocals" in ident else None

    # short instrument phrase per stem, for prompts
    phrases = {}
    for n in sorted(present, key=lambda k: -stem_info[k]["level_vs_mix_db"]):
        idn = ident.get(n)
        if n == "vocals" or not idn:
            continue
        if n == "other":
            good = [l["label"] for l in layers if l["label"] not in NOT_AN_INSTRUMENT][:2]
            if not good and idn["detail"] not in NOT_AN_INSTRUMENT:
                good = [idn["detail"]]
            if good:
                phrases[n] = good
            continue
        if idn["detail"] in BLEED_HINTS.get(n, ()) or idn["family"] in NOT_AN_INSTRUMENT:
            continue                            # stem is really something else bleeding in
        if idn["confidence"] == "unsure":
            phrases[n] = [GENERIC[n]]
        elif idn["detail_p"] >= 0.5 * idn["family_p"]:
            phrases[n] = [idn["detail"]]        # specific sound is clear
        else:
            phrases[n] = [idn["family"]]

    step(7, "writing report")
    # ---- sections
    sec_rows = []
    means = [float(energy[(t >= s) & (t < e)].mean()) for s, e, *_ in sections]
    for i, (s, e, letter, a, b) in enumerate(sections):
        seg_e = energy[(t >= s) & (t < e)]
        h = len(seg_e) // 2
        rising = h > 0 and seg_e[h:].mean() - seg_e[:h].mean() > 0.12
        playing = []
        for n in present:
            r = stem_rms[n]
            ft = librosa.frames_to_time(np.arange(len(r)), sr=ASR, hop_length=HOP)
            rs = r[(ft >= s) & (ft < e)]
            if len(rs) and db(np.percentile(rs, 75)) > db(np.percentile(r, 95)) - 20:
                playing.append(n)
        prog, loop = S.progression(chords[a * cu:b * cu])
        sec_rows.append({"start": s, "end": e, "part": letter, "bars": b - a,
                         "role": section_role(i, len(sections), means[i], means, rising),
                         "energy": round(means[i], 2), "playing": playing,
                         "chords": prog, "chords_loop": loop})
    loops = [r["chords"] for r in sec_rows if r["chords"] and r["chords_loop"]]
    main_prog = max(set(loops), key=loops.count) if loops else None
    loop_bars = sum(r["bars"] for r in sec_rows if r["chords_loop"] and r["chords"] == main_prog)
    loop_share = loop_bars / max(1, sum(r["bars"] for r in sec_rows))
    vals = [c for c in chords if c]
    common_chords = sorted(set(vals), key=lambda c: -vals.count(c))[:4]
    if main_prog and loop_share >= 0.5:      # only trust the loop for the key if it runs most of the song
        rel = S.pretty_key(relative_key(key_raw))
        rn, rmode = rel.split()
        if main_prog.split(" → ")[0] == rn + ("m" if rmode == "minor" else ""):
            key_raw = relative_key(key_raw)
            key = rel

    # ---- key moments
    moments = []
    for d_, _ in sorted(drops, key=lambda x: -x[1])[:5]:
        moments.append((d_, "big hit / drop"))
    for d_, _ in sorted(dips, key=lambda x: -x[1])[:3]:
        moments.append((d_, "pull-back"))
    for p_, _ in sorted(peaks, key=lambda x: -x[1])[:2]:
        moments.append((p_, "peak"))
    moments.sort()
    kept = []
    order = {"big hit / drop": 0, "pull-back": 1, "peak": 2}
    for m in moments:
        if kept and m[0] - kept[-1][0] < 3:
            if order[m[1]] < order[kept[-1][1]]:
                kept[-1] = m
            continue
        kept.append(m)
    moment_rows = []
    for tm, kind in kept:
        ins, outs = S.what_changed(tm, stem_rms, [n for n in present if not (spoken and n == "vocals")])
        moment_rows.append({"time": tm, "kind": kind, "enter": ins, "leave": outs})
    entries = S.first_entries(stem_rms, present)

    title = f"{base} · {tempo:.0f} BPM · {key} · {ts(duration)}"
    plot(os.path.join(out, "timeline.png"), t, energy, [(s, e, l) for s, e, l, *_ in sections],
         peaks, drops, stem_rms, duration, title)

    prompt_data = {"genres": genres, "moods": moods, "tempo": tempo, "key": key,
                   "instrument_phrases": phrases, "sections": sec_rows, "singing": singing,
                   "vocal_style": vocal_style, "production": production[0][0] if production else None}
    prompts = S.suno_prompts(prompt_data)

    # ================================================= markdown
    flat = key_raw in S.FLAT_KEYS
    spell = lambda notes: ", ".join(S.NOTE_FLAT[S.NOTE_SHARP.index(x)] if flat and x in S.NOTE_SHARP else x for x in notes)
    pct = lambda items, k=3: ", ".join(f"{n} ({p:.0%})" for n, p in items[:k])
    L = [f"# Sound breakdown: {base}", ""]
    inst_line = S.join_words([x for v in phrases.values() for x in v]) or "unclear"
    L += [f"**In one line:** {genres[0][0]} at {tempo:.0f} BPM in {key}, {moods[0][0]}, "
          f"built from {inst_line}"
          + (f", with {vocal_style}" if vocal_style else ", with a spoken voiceover" if voice_type else ", no voice")
          + ".", ""]

    L += ["## Overview", "", "| | | How sure |", "|---|---|---|",
          f"| Length | {ts(duration)} | measured |",
          f"| Tempo | {tempo:.0f} BPM" + (f" ({feel})" if feel else "") + f" | {'measured' if has_beat else 'no steady beat found'} |",
          f"| Key | {key} (same notes as {S.pretty_key(relative_key(key_raw))}) | measured, medium |",
          f"| Tuning | " + (f"A4 ≈ {tuning['a4_hz']} Hz ({tuning['cents_from_440']:+.0f} cents): {tuning['verdict']} | measured, {tuning['agreement']} agreement |"
                           if tuning else "not enough pitched sound | — |"),
          f"| Genre | {pct(genres)} | model guess |",
          f"| Mood | {pct(moods)} | model guess |",
          f"| Production | {pct(production, 2)} | model guess |",
          f"| Voice | {(vocal_style + ' (' + voice_type + ')') if vocal_style else (voice_type or 'none')} | model guess |",
          (f"| Main chord loop | {main_prog} (in {loop_share:.0%} of the song) | measured, {'good' if loop_share >= 0.5 else 'low'} |"
           if main_prog else "| Main chord loop | no steady loop found | — |"),
          f"| Most common chords | {', '.join(common_chords) or '—'} | measured, medium |", ""]

    L += ["## Instruments", "",
          "Loudest first. Each one was separated from the mix, then identified by a music model and measured.", ""]
    for n in sorted(present, key=lambda k: -stem_info[k]["level_vs_mix_db"]):
        i = stem_info[n]
        idn = ident.get(n)
        if n == "vocals":
            head = vocal_style or voice_type
        elif idn:
            head = idn["call"] + (f" ({idn['detail']})" if idn["confidence"] != "unsure" and idn["detail"] != idn["family"] else "")
        else:
            head = "unidentified"
        L.append(f"### {n.capitalize()}: {head}")
        if idn:
            L.append(f"- **Identified as:** {pct(idn['families'])} ({CONF_WORD[idn['confidence']]}). "
                     f"Closest specific sounds: {pct(idn['closest'])}")
            top = idn["detail"]
            if top in BLEED_HINTS.get(n, ()):
                L.append(f"- ⚠️ The separator put this in the '{n}' stem, but it sounds like **{top}**. "
                         f"Probably not a real {n}; treat it as bleed from another instrument.")
            if top in NOT_AN_INSTRUMENT:
                L.append("- ⚠️ This stem is mostly non-musical sound (effects, noise), not a played instrument.")
        if n == "other" and layers:
            L.append("- **Layers found over time:** " + "; ".join(
                f"{l['label']} ({', '.join(ts(a) + '–' + ts(min(b, duration)) for a, b in l['ranges'][:4])})"
                for l in layers[:5]))
        L.append(f"- **Sound:** {', '.join(i['sound'])}")
        L.append(f"- **In the mix:** {i['level_vs_mix_db']:+.1f} dB vs the full song, playing {i['active_pct']}% of the time"
                 + (f", first comes in at {ts(entries[n])}" if n in entries else ""))
        sp = i["spectrum_share"]
        L.append(f"- **Frequency:** sub {sp['sub<60']:.0%} · low {sp['low60-250']:.0%} · mid {sp['mid250-4k']:.0%} · high {sp['high>4k']:.0%} (brightness ≈ {i['brightness_hz']} Hz)")
        if n == "vocals" and spoken:
            pass
        elif "pitch_range" in i:
            L.append(f"- **Notes:** range {i['pitch_range']}, most used {spell(i['most_used_notes'])}")
        elif "most_used_notes" in i:
            L.append(f"- **Notes:** most used {spell(i['most_used_notes'])}")
        if n == "guitar" and gtune:
            L.append(f"- **Guitar tuning (guess):** {gtune['guess']} (confidence {gtune['confidence']}; lowest note ≈ {gtune['lowest_note']})")
        L.append("")
    absent = [n for n in STEM_ORDER if n not in present]
    if absent:
        L += [f"Not in this song (only faint bleed): {', '.join(absent)}.", ""]

    L += ["## Song map", "", "![timeline](timeline.png)", "",
          "| Time | Bars | Part | What it is | Energy | Playing | Chords |", "|---|---|---|---|---|---|---|"]
    for r in sec_rows:
        bar = "█" * int(round(r["energy"] * 10)) + "░" * (10 - int(round(r["energy"] * 10)))
        L.append(f"| {ts(r['start'])}–{ts(r['end'])} | {r['bars'] if has_beat else '—'} | {r['part']} | {r['role']} | {bar} | "
                 f"{', '.join(r['playing']) or '—'} | {(r['chords'] or '—') + (' (loop)' if r['chords_loop'] else '')} |")
    L += ["", "Parts with the same letter sound alike. Chords are per bar; treat them as a guide.", ""]

    L += ["## Key moments", ""]
    for m in moment_rows:
        ch = []
        if m["enter"]:
            ch.append(f"{S.join_words(m['enter'])} come{'s' if len(m['enter']) == 1 else ''} in")
        if m["leave"]:
            ch.append(f"{S.join_words(m['leave'])} drop{'s' if len(m['leave']) == 1 else ''} out")
        L.append(f"- **{ts(m['time'])}** {m['kind']}" + (f": {'; '.join(ch)}" if ch else ""))
    if not moment_rows:
        L.append("- The energy stays even; no big hits or drops found.")
    L.append("")

    L += ["## The beat", ""]
    drum_id = ident.get("drums", {}).get("family")
    if pats and "drums" in present and drum_id not in NOT_AN_INSTRUMENT:
        L += ["One bar of 4/4 in 16th notes (X = strong hit, x = softer hit, . = nothing), averaged over the song:", "",
              "```", "                 1... 2... 3... 4..."]
        for n, p in pats.items():
            L.append(f"{n:<17}{p['pattern']}   ~{p['hits_per_bar']:.0f} hits/bar, in {p['bars_playing_pct']}% of bars")
        L += ["```", ""]
        if ident.get("drums", {}).get("confidence") == "unsure":
            L += ["The drum sound wasn't clearly identified, so this pattern may include non-drum hits.", ""]
    else:
        L += ["No reliable drum pattern (no drums, a free tempo, or the drum stem is mostly effects).", ""]

    L += ["## Mix and production", "",
          f"- **Overall:** {', '.join(mix['words'])}",
          f"- **Loudness:** loud parts at {mix['loud_parts_dbfs']} dBFS, peaks at {mix['peak_dbfs']} dBFS; "
          f"{mix['section_contrast_db']} dB between loud and quiet parts",
          "- **Tonal balance:** " + " · ".join(f"{k} {v:.0%}" for k, v in mix["balance"].items()),
          f"- **Stereo width:** {mix['stereo_width']} (0 = mono, 1 = extremely wide)", ""]

    if sfx:
        L += ["## Sound design (non-music sounds)", "",
              "; ".join(f"{n} ({', '.join(ts(x) for x in tl[:3])})" for n, tl in sfx[:20]), ""]

    L += ["## Prompts to recreate this", "",
          "**Suno, Simple mode** (paste into the one box):", "", "```", prompts["simple"], "```", "",
          "**Suno, Advanced mode, Styles box:**", "", "```", prompts["styles"], "```", ""]
    if prompts["instrumental"]:
        L += ["**Lyrics box:** leave empty (this track has no singing).", ""]
    else:
        L += ["**Lyrics box:** your own lyrics, using this section order:", "", "```", prompts["lyrics_tags"], "```", ""]
    L += ["---", "Stems: Demucs htdemucs_6s. Instrument, genre and mood names: LAION CLAP. Voice and sound effects: MIT AST. "
          "Tempo, key, tuning, chords, energy and timbre: measured with librosa. Stems are in `stems/`."]

    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L))
    data = {"file": args.song, "duration_sec": duration, "tempo_bpm": tempo, "key": key,
            "key_alternative": key_alt, "tuning": tuning, "guitar_tuning": gtune,
            "genres": genres, "moods": moods, "production": production,
            "voice": voice_type, "vocal_style": vocal_style,
            "instruments": ident, "other_layers": layers, "stems": stem_info,
            "sections": sec_rows, "main_chord_loop": main_prog, "loop_share": loop_share, "common_chords": common_chords, "chords_per_unit": chords,
            "chord_unit": "half bar" if half else "bar", "bar_starts": [float(x) for x in bars],
            "moments": moment_rows, "first_entries": entries, "drum_patterns": pats,
            "mix": mix, "sound_design": sfx, "prompts": prompts}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(data, f, indent=2, default=float, ensure_ascii=False)
    print(f"\nDone → {out}/report.md")


if __name__ == "__main__":
    main()
