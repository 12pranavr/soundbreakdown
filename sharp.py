"""Sharper analysis: instrument identification (CLAP), chords, sections, mix, prompts."""
import numpy as np
import librosa

ASR = 22050
HOP = 512
NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
FLAT_KEYS = {"F major", "A# major", "D# major", "G# major", "C# major",
             "D minor", "G minor", "C minor", "F minor", "A# minor", "D# minor"}


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))


def ts(sec):
    sec = max(0, float(sec))
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def pretty_key(key):
    """'A# minor' -> 'B♭ minor' when the key is normally written with flats."""
    n, mode = key.split()
    names = NOTE_FLAT if key in FLAT_KEYS else NOTE_SHARP
    return f"{names[NOTE_SHARP.index(n)]} {mode}"


def join_words(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ================================================================= CLAP listener
VOCAB = {
    "drums": [
        "acoustic drum kit", "electronic drum machine", "trap drums with fast 808 hi-hats",
        "boom bap hip hop drums", "four on the floor house kick", "breakbeat drums",
        "cinematic trailer percussion with big impacts", "taiko drums", "orchestral timpani and cymbals",
        "brushed jazz drums", "rock drums with loud snare", "tabla", "dhol", "dholak",
        "hand percussion, congas and shakers", "claps and finger snaps", "lo-fi dusty drums",
        "reggaeton dembow drums", "afrobeat percussion", "sound effects and noise, not drums"],
    "bass": [
        "808 sub bass", "deep sine sub bass", "electric bass guitar", "slap bass", "synth bass",
        "distorted reese bass", "acid synth bass", "upright double bass", "cello and double bass section",
        "low synth drone", "tuba or low brass", "distorted bass guitar"],
    "guitar": [
        "clean electric guitar", "distorted electric guitar", "electric guitar with delay and reverb",
        "palm muted electric guitar", "acoustic guitar strumming", "fingerpicked acoustic guitar",
        "nylon string classical guitar", "slide guitar", "funk rhythm guitar", "electric guitar solo",
        "sitar", "mandolin", "banjo", "ukulele", "tumbi", "oud", "harp", "synth pluck",
        "pizzicato strings", "piano"],
    "piano": [
        "grand piano", "soft felt piano", "upright piano", "electric piano rhodes", "synth keys",
        "harpsichord", "organ", "music box or celesta", "harmonium", "guitar", "strings"],
    "other": [
        "string section", "solo violin", "solo cello", "pizzicato strings", "full orchestra",
        "brass section", "french horns", "trumpet", "saxophone", "clarinet", "flute", "choir",
        "church organ", "harp", "bells and glockenspiel", "marimba or xylophone", "accordion",
        "warm synth pad", "bright synth lead", "arpeggiated synthesizer", "supersaw synth chords",
        "ambient drone and atmosphere", "electric piano", "piano", "acoustic guitar", "electric guitar",
        "sitar", "harmonium", "bansuri bamboo flute", "sarangi", "shehnai", "santoor", "veena",
        "vocal chops and vocal samples", "sound effects and field recordings", "vinyl crackle and noise"],
    "vocals": [
        "male singing voice", "female singing voice", "rap vocals", "choir or group vocals",
        "autotuned vocals", "soft breathy vocals", "powerful belted vocals", "falsetto vocals",
        "indian classical vocals", "child singing", "spoken word", "whispering", "vocal chops"],
    "genre": [
        "pop", "indie pop", "synth-pop", "k-pop", "rock", "indie rock", "hard rock", "metal", "punk",
        "hip hop", "trap", "boom bap", "drill", "lo-fi hip hop", "r&b", "soul", "funk", "disco", "jazz",
        "blues", "country", "folk", "acoustic singer-songwriter", "reggae", "reggaeton", "latin pop",
        "afrobeats", "house", "deep house", "techno", "trance", "drum and bass", "dubstep",
        "future bass", "edm", "synthwave", "ambient", "downtempo chillout", "classical orchestral",
        "epic cinematic trailer music", "emotional film score", "video game soundtrack",
        "corporate advertising music", "bollywood film song", "punjabi bhangra", "punjabi hip hop",
        "indian classical", "sufi qawwali", "ghazal", "hindi indie pop", "devotional bhajan",
        "arabic music", "gospel", "world fusion", "piano ballad", "phonk", "hyperpop"],
    "mood": [
        "happy and uplifting", "sad and melancholic", "epic and heroic", "dark and menacing",
        "calm and peaceful", "energetic and driving", "romantic and tender", "nostalgic and bittersweet",
        "tense and suspenseful", "aggressive and angry", "dreamy and ethereal", "playful and fun",
        "inspiring and hopeful", "mysterious", "triumphant", "lonely and introspective", "sensual and groovy",
        "spiritual and devotional"],
    "production": [
        "clean polished modern studio production", "lo-fi, tape-saturated and dusty",
        "huge reverb, spacious and atmospheric", "dry, close and intimate", "raw live band recording",
        "loud, heavily compressed and punchy", "vintage warm analog sound", "wide cinematic orchestral sound",
        "distorted and gritty", "minimal and sparse"],
}


FAMILY = {
    "drums": {
        "electronic drums": ["electronic drum machine", "trap drums with fast 808 hi-hats", "boom bap hip hop drums",
                             "four on the floor house kick", "breakbeat drums", "lo-fi dusty drums",
                             "reggaeton dembow drums"],
        "acoustic drum kit": ["acoustic drum kit", "brushed jazz drums", "rock drums with loud snare"],
        "cinematic percussion": ["cinematic trailer percussion with big impacts", "taiko drums",
                                 "orchestral timpani and cymbals"],
        "Indian percussion": ["tabla", "dhol", "dholak"],
        "hand percussion": ["hand percussion, congas and shakers", "claps and finger snaps", "afrobeat percussion"],
    },
    "bass": {
        "synth / sub bass": ["808 sub bass", "deep sine sub bass", "synth bass", "distorted reese bass",
                             "acid synth bass", "low synth drone"],
        "bass guitar": ["electric bass guitar", "slap bass", "distorted bass guitar"],
        "acoustic low instruments": ["upright double bass", "cello and double bass section", "tuba or low brass"],
    },
    "guitar": {
        "electric guitar": ["clean electric guitar", "distorted electric guitar", "electric guitar with delay and reverb",
                            "palm muted electric guitar", "slide guitar", "funk rhythm guitar", "electric guitar solo"],
        "acoustic guitar": ["acoustic guitar strumming", "fingerpicked acoustic guitar", "nylon string classical guitar"],
        "other plucked strings": ["sitar", "mandolin", "banjo", "ukulele", "tumbi", "oud", "harp"],
    },
    "piano": {
        "acoustic piano": ["grand piano", "soft felt piano", "upright piano"],
        "electric keys": ["electric piano rhodes", "synth keys", "organ", "harmonium"],
    },
    "other": {
        "orchestral strings": ["string section", "solo violin", "solo cello", "pizzicato strings", "full orchestra"],
        "brass": ["brass section", "french horns", "trumpet"],
        "woodwinds": ["saxophone", "clarinet", "flute", "bansuri bamboo flute", "shehnai"],
        "synths": ["warm synth pad", "bright synth lead", "arpeggiated synthesizer", "supersaw synth chords",
                   "ambient drone and atmosphere"],
        "keys": ["electric piano", "piano", "church organ", "harmonium", "accordion"],
        "guitars": ["acoustic guitar", "electric guitar"],
        "Indian melodic instruments": ["sitar", "sarangi", "santoor", "veena"],
        "tuned percussion": ["bells and glockenspiel", "marimba or xylophone", "harp"],
    },
    "vocals": {
        "sung vocals": ["male singing voice", "female singing voice", "autotuned vocals", "soft breathy vocals",
                        "powerful belted vocals", "falsetto vocals", "indian classical vocals", "child singing",
                        "choir or group vocals"],
    },
}


def identify(probs, stem, skip=()):
    """Family-level call (robust) plus the closest specific match inside that family."""
    labels = VOCAB[stem]
    m = probs.mean(0).copy()
    for i, l in enumerate(labels):
        if l in skip:
            m[i] = 0
    m = m / (m.sum() + 1e-9)
    fam_of = {l: f for f, ls in FAMILY.get(stem, {}).items() for l in ls}
    fam = {}
    for l, p in zip(labels, m):
        fam[fam_of.get(l, l)] = fam.get(fam_of.get(l, l), 0) + float(p)
    fr = sorted(fam.items(), key=lambda x: -x[1])
    (fa, pa), (fb, pb) = fr[0], (fr[1] if len(fr) > 1 else ("", 0))
    conf = "clear" if pa >= 0.55 and pa >= 2 * pb else "likely" if pa >= 0.35 and pa >= 1.4 * pb else "unsure"
    order = np.argsort(m)[::-1]
    closest = [(labels[i], float(m[i])) for i in order[:3]]
    inside = [(labels[i], float(m[i])) for i in order if fam_of.get(labels[i], labels[i]) == fa]
    detail, pd = inside[0]
    return {"family": fa, "family_p": pa, "families": fr[:3], "confidence": conf,
            "detail": detail, "detail_p": pd, "closest": closest,
            "call": fa if conf != "unsure" else f"{fa} or {fb}"}


class Listener:
    """Zero-shot listening with LAION CLAP (music + speech checkpoint)."""
    NAME = "laion/larger_clap_music_and_speech"
    SR = 48000

    def __init__(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.proc = ClapProcessor.from_pretrained(self.NAME)
        self.model = ClapModel.from_pretrained(self.NAME).eval()

    def listen(self, y, sr, groups, win=10.0, gate_db=-25.0):
        """Score 10-second windows of `y` against each label group.

        Returns (window_start_times, {group: probs[n_windows, n_labels]}) or (None, None) if silent.
        Probabilities are a softmax inside each group, so they answer 'which of these is it?'.
        """
        y48 = librosa.resample(y, orig_sr=sr, target_sr=self.SR) if sr != self.SR else y
        n = int(win * self.SR)
        starts = list(range(0, max(1, len(y48) - n // 3), n))
        chunks = [y48[s:s + n] for s in starts]
        lev = np.array([db(np.sqrt(np.mean(c ** 2) + 1e-12)) for c in chunks])
        keep = [i for i in range(len(chunks)) if lev[i] > max(lev.max() + gate_db, -60)]
        if not keep:
            return None, None
        texts, spans, pos = [], {}, 0
        for g in groups:
            labs = VOCAB[g]
            texts += [f"the sound of {l}" for l in labs]
            spans[g] = (pos, pos + len(labs))
            pos += len(labs)
        logits = []
        for i in range(0, len(keep), 6):
            batch = [chunks[j] for j in keep[i:i + 6]]
            inp = self.proc(text=texts, audio=batch, sampling_rate=self.SR, return_tensors="pt", padding=True)
            with self.torch.no_grad():
                logits.append(self.model(**inp).logits_per_audio)
        logits = self.torch.cat(logits)
        out = {g: self.torch.softmax(logits[:, a:b], dim=1).numpy() for g, (a, b) in spans.items()}
        return np.array([starts[i] / self.SR for i in keep]), out


def rank(probs, labels, k=3):
    m = probs.mean(0)
    order = np.argsort(m)[::-1][:k]
    return [(labels[i], float(m[i])) for i in order]


def verdict(ranked):
    """Turn ranked (label, prob) into an honest one-line call."""
    (a, pa), (b, pb) = ranked[0], ranked[1]
    if pa >= 0.5 and pa >= 2 * pb:
        return a, "clear"
    if pa >= 0.3 and pa >= 1.4 * pb:
        return a, "likely"
    return f"{a} or {b}", "unsure"


def layers_over_time(win_t, probs, labels, win=10.0, min_p=0.35, skip=()):
    """Which label wins each window -> merged time ranges per label (for mixed stems)."""
    probs = probs.copy()
    for i, l in enumerate(labels):
        if l in skip:
            probs[:, i] = 0
    top = probs.argmax(1)
    conf = probs.max(1)
    found = {}
    for t, i, c in zip(win_t, top, conf):
        if c >= min_p:
            found.setdefault(labels[i], []).append(float(t))
    out = []
    for lab, times in found.items():
        if len(times) < max(2, 0.12 * len(win_t)) and len(win_t) > 4:
            continue
        ranges, s, prev = [], times[0], times[0]
        for t in times[1:]:
            if t - prev > win * 1.5:
                ranges.append((s, prev + win))
                s = t
            prev = t
        ranges.append((s, prev + win))
        out.append({"label": lab, "share": len(times) / len(win_t), "ranges": ranges})
    return sorted(out, key=lambda x: -x["share"])


# ================================================================= bar grid
def bar_grid(beat_t, phase, duration):
    """Bar start times across the whole song, from tracked beats (beat `phase` is beat 1)."""
    if len(beat_t) < 9:
        step = 2.0
        return np.arange(0, duration, step), step, False
    b = list(beat_t[phase::4])
    bl = float(np.median(np.diff(b)))
    while b[0] - bl > -0.05:
        b.insert(0, b[0] - bl)
    while b[-1] + bl < duration - 0.25 * bl:
        b.append(b[-1] + bl)
    b = [max(0.0, x) for x in b]
    if b[0] > 0.3 * bl:
        b.insert(0, 0.0)
    else:
        b[0] = 0.0
    return np.array(b), bl, True


def per_unit(x, x_times, edges, agg=np.mean):
    """Aggregate a (features x frames) or (frames,) array over [edges[i], edges[i+1])."""
    idx = np.searchsorted(x_times, edges)
    one = x.ndim == 1
    X = x[None] if one else x
    out = np.zeros((X.shape[0], len(edges) - 1))
    for i in range(len(edges) - 1):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        seg = X[:, a:b]
        if seg.shape[1]:
            out[:, i] = agg(seg, axis=1)
    return out[0] if one else out


# ================================================================= chords
def chord_track(stems_m, edges, key):
    """One chord per unit (bar or half bar): root from harmony + bass, quality from the third."""
    names = NOTE_FLAT if key in FLAT_KEYS else NOTE_SHARP
    harm = stems_m["guitar"] + stems_m["piano"] + stems_m["other"]
    hop = HOP * 2
    Ch = librosa.feature.chroma_cqt(y=harm, sr=ASR, hop_length=hop)
    Cb = librosa.feature.chroma_cqt(y=stems_m["bass"], sr=ASR, hop_length=hop, n_octaves=4)
    ft = librosa.frames_to_time(np.arange(Ch.shape[1]), sr=ASR, hop_length=hop)
    lev_h = db(librosa.feature.rms(y=harm, hop_length=hop)[0][:len(ft)])
    lev_b = db(librosa.feature.rms(y=stems_m["bass"], hop_length=hop)[0][:len(ft)])
    H = per_unit(Ch, ft, edges)
    B = per_unit(Cb, ft, edges)
    lh, lb = per_unit(lev_h, ft, edges), per_unit(lev_b, ft, edges)
    h_on = lh > max(np.percentile(lev_h, 95) - 30, -60)
    b_on = lb > max(np.percentile(lev_b, 95) - 25, -60)
    kn, kmode = key.split()
    tonic = NOTE_SHARP.index(kn) if kmode == "major" else (NOTE_SHARP.index(kn) + 3) % 12
    in_key = {((tonic + o) % 12, q) for o, q in ((0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m"))}
    if kmode == "minor":
        in_key.add(((NOTE_SHARP.index(kn) + 7) % 12, ""))      # major V in a minor key
    out = []
    for i in range(H.shape[1]):
        if not h_on[i] and not b_on[i]:
            out.append(None)
            continue
        h = H[:, i] / (np.linalg.norm(H[:, i]) + 1e-9) if h_on[i] else np.zeros(12)
        b = B[:, i] / (B[:, i].max() + 1e-9) if b_on[i] else np.zeros(12)
        best, best_s = None, -1
        for r in range(12):
            for q, third in (("", 4), ("m", 3)):
                tri = (h[r] + h[(r + third) % 12] + h[(r + 7) % 12]) / np.sqrt(3)
                s = tri + 0.45 * b[r] + (0.12 if (r, q) in in_key else 0)
                if s > best_s:
                    best, best_s = (r, q), s
        r, q = best
        if h_on[i]:
            thirds = max(h[(r + 3) % 12], h[(r + 4) % 12])
            if thirds < 0.45 * max(h[r], h[(r + 7) % 12]):
                q = "5"
        else:
            q = " (bass only)"
        out.append(names[r] + q)
    # smooth single-unit blips
    for i in range(1, len(out) - 1):
        if out[i - 1] == out[i + 1] and out[i] != out[i - 1] and out[i - 1] is not None:
            out[i] = out[i - 1]
    return out


def progression(seq):
    """Collapse a per-unit chord list to a readable progression; detect loops."""
    run = []
    for c in seq:
        if c is None:
            continue
        if not run or run[-1] != c:
            run.append(c)
    if not run:
        return None, False
    if len(run) == 1:
        return run[0] + " (held)", False
    for L in range(2, 7):
        if len(run) >= 2 * L:
            match = np.mean([run[i] == run[i % L] for i in range(len(run))])
            if match >= 0.8:
                return " → ".join(run[:L]), True
    if len(run) <= 4:
        return " → ".join(run), False
    vals = [c for c in seq if c is not None]
    common = sorted(set(vals), key=lambda c: -vals.count(c))[:3]
    return "mostly " + ", ".join(common) + " (no steady loop)", False


# ================================================================= sections
def find_sections(stem_rms, present, energy, e_times, chroma_bar, bars, bar_len, duration):
    """Split the song where the arrangement, energy or harmony changes, on bar lines."""
    from scipy.signal import find_peaks
    edges = np.concatenate([bars, [duration]])
    rows = []
    for n in present:
        r = stem_rms[n]
        ft = librosa.frames_to_time(np.arange(len(r)), sr=ASR, hop_length=HOP)
        ref = np.percentile(db(r), 95)
        rows.append(per_unit(np.clip((db(r) - ref + 40) / 40, 0, 1), ft, edges))
    rows.append(per_unit(energy, e_times, edges) * 1.5)
    F = np.vstack(rows + [chroma_bar * 0.6]).T          # bars x features
    nb = F.shape[0]
    w = int(np.clip(round(7.0 / bar_len), 2, 4))
    nov = np.zeros(nb)
    for b in range(2, nb - 1):
        nov[b] = np.linalg.norm(F[b:b + w].mean(0) - F[max(0, b - w):b].mean(0))
    min_bars = max(2, int(np.ceil(6.0 / bar_len)))
    thr = max(0.3 * nov.max(), np.median(nov) + 0.8 * nov.std())
    pk, _ = find_peaks(nov, height=thr, distance=min_bars)
    cuts = [0] + [int(p) for p in pk if min_bars <= p <= nb - min_bars] + [nb]
    cuts = sorted(set(cuts))
    vecs = [F[a:b].mean(0) for a, b in zip(cuts[:-1], cuts[1:])]
    letters, protos = [], []
    tol = 0.18 * np.sqrt(F.shape[1])
    for v in vecs:
        d = [np.linalg.norm(v - p) for p in protos]
        if d and min(d) < tol:
            letters.append(chr(65 + int(np.argmin(d))))
        else:
            protos.append(v)
            letters.append(chr(65 + min(len(protos) - 1, 25)))
    return [(float(edges[a]), float(edges[b]), letters[i], a, b)
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]


# ================================================================= moments
def what_changed(t, stem_rms, present, span=3.0):
    """Which stems come in or drop out around time t."""
    ins, outs = [], []
    for n in present:
        r = stem_rms[n]
        ft = librosa.frames_to_time(np.arange(len(r)), sr=ASR, hop_length=HOP)
        before = r[(ft >= t - span) & (ft < t - 0.4)]
        after = r[(ft >= t + 0.4) & (ft < t + span)]
        if not len(before) or not len(after):
            continue
        ref = np.percentile(db(r), 95)
        d_b, d_a = db(np.sqrt(np.mean(before ** 2))), db(np.sqrt(np.mean(after ** 2)))
        if d_a - d_b > 8 and d_a > ref - 18:
            ins.append(n)
        elif d_b - d_a > 8 and d_b > ref - 18:
            outs.append(n)
    return ins, outs


def first_entries(stem_rms, present):
    out = {}
    for n in present:
        r = stem_rms[n]
        w = max(1, int(ASR / HOP))
        sm = db(np.convolve(r, np.ones(w) / w, mode="same"))
        on = np.where(sm > np.percentile(sm, 95) - 20)[0]
        if len(on):
            out[n] = float(on[0] * HOP / ASR)
    return out


# ================================================================= mix
def mix_report(y, sr):
    """Loudness, dynamics, stereo width and tonal balance of the full mix."""
    mono = y.mean(0)
    peak = float(db(np.abs(y).max()))
    rms = float(db(np.sqrt(np.mean(mono ** 2) + 1e-12)))
    win = sr
    frames = [mono[i:i + win] for i in range(0, len(mono) - win, win)]
    lv = np.array([db(np.sqrt(np.mean(f ** 2) + 1e-12)) for f in frames]) if frames else np.array([rms])
    loud = float(np.percentile(lv, 95))
    swing = float(np.percentile(lv, 95) - np.percentile(lv, 10))
    mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2
    width = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-9))
    S = np.abs(librosa.stft(librosa.resample(mono, orig_sr=sr, target_sr=ASR), n_fft=4096, hop_length=2048)) ** 2
    f = librosa.fft_frequencies(sr=ASR, n_fft=4096)
    p = S.sum(1)
    share = lambda lo, hi: float(p[(f >= lo) & (f < hi)].sum() / (p.sum() + 1e-12))
    bal = {"sub (<60 Hz)": share(20, 60), "bass (60–250)": share(60, 250),
           "mids (250–4k)": share(250, 4000), "highs (>4k)": share(4000, 11025)}
    words = []
    crest = peak - loud
    words.append("very loud, heavily limited master" if crest < 8 else
                 "loud, modern master" if crest < 12 else
                 "moderate loudness with room to breathe" if crest < 17 else "quiet, very dynamic master")
    words.append("almost no loud/soft contrast" if swing < 5 else
                 "some loud/soft contrast" if swing < 12 else "big loud/soft contrast between sections")
    low = bal["sub (<60 Hz)"] + bal["bass (60–250)"]
    words.append("bass-heavy" if low > 0.75 else "bright, top-heavy" if bal["highs (>4k)"] > 0.08 else
                 "mid-forward" if bal["mids (250–4k)"] > 0.5 else "balanced tone")
    words.append("narrow / near-mono" if width < 0.15 else "moderately wide" if width < 0.4 else "very wide stereo")
    return {"peak_dbfs": round(peak, 1), "loud_parts_dbfs": round(loud, 1), "crest_db": round(crest, 1),
            "section_contrast_db": round(swing, 1), "stereo_width": round(width, 2),
            "balance": {k: round(v, 2) for k, v in bal.items()}, "words": words}


# ================================================================= prompts
def suno_prompts(d):
    """Build Suno 'Simple' sentence, 'Advanced' styles and a neutral description from results."""
    strong = lambda items: [n for n, p in items[:2] if p >= 0.4 * items[0][1]]
    genres = strong(d["genres"])
    moods = strong(d["moods"])
    bpm, key = round(d["tempo"]), d["key"]
    inst = d["instrument_phrases"]            # ordered {stem: phrase}
    secs = d["sections"]
    sing = d["singing"]

    def names(stems):
        out = []
        for s in stems:
            for x in inst.get(s, []) if s != "vocals" else []:
                if x not in out:
                    out.append(x)
        return out

    first = names(secs[0]["playing"]) if secs else []
    peak = max(secs, key=lambda s: s["energy"]) if secs else None
    later = [x for x in names(peak["playing"]) if x not in first] if peak else []
    mid = secs[1:-1]
    has_break = any("Breakdown" in s["role"] for s in mid)
    parts = [f"{genres[0].capitalize()}" + (f" with {genres[1]} influence" if len(genres) > 1 else "")
             + f" at {bpm} BPM in {key}" + (f", {', '.join(moods)}" if moods else "")]
    if first:
        parts.append(f"starting with {join_words(first[:3])}")
    if later:
        parts.append(f"then building with {join_words(later[:3])}")
    if peak and peak is not secs[0]:
        parts.append(f"into a powerful peak around {ts(peak['start'])}")
    if has_break:
        parts.append("dropping into a quieter breakdown before rising again")
    if d.get("vocal_style") and sing:
        parts.append(f"with {d['vocal_style']}")
    else:
        parts.append("fully instrumental with no vocals")
    simple = ", ".join(parts)

    tags = genres + [f"{bpm} BPM", key] + moods + names(list(inst))
    if d.get("production"):
        tags.append(d["production"])
    if d.get("vocal_style") and sing:
        tags.append(d["vocal_style"])
    else:
        tags.append("instrumental")
    seen, styles = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            styles.append(t)
    structure = []
    for s in secs:
        tag = s["role"].split(" (")[0].split(" /")[0]
        if not structure or structure[-1] != tag:
            structure.append(tag)
    return {"simple": simple, "styles": ", ".join(styles),
            "lyrics_tags": "\n".join(f"[{t}]" for t in structure),
            "instrumental": not sing}
