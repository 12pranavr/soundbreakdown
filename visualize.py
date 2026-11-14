#!/usr/bin/env python3
"""Sargam visualizer: a vertical video of the voice's notes and the instruments around it.

Usage:  python visualize.py SONG [-o OUTPUT_DIR] [--sa C] [--title "Name"] [--start S --end S]

Needs the breakdown of the same song (stems + report.json); runs it first if missing.
Writes sargam.mp4, melody.json and sargam.txt into the song's breakdown folder.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
from PIL import Image, ImageDraw, ImageFont

import breakdown as B

ASR = 22050
HOP = 256
W, H, SS = 720, 1280, 2                       # output size, supersampling factor
NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DEV = ["सा", "रे", "रे", "ग", "ग", "म", "म", "प", "ध", "ध", "नि", "नि"]
LAT = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]
KOMAL, TIVRA = {1, 3, 8, 10}, {6}
THAATS = {"Bilawal": [0, 2, 4, 5, 7, 9, 11], "Kalyan": [0, 2, 4, 6, 7, 9, 11], "Khamaj": [0, 2, 4, 5, 7, 9, 10],
          "Bhairav": [0, 1, 4, 5, 7, 8, 11], "Poorvi": [0, 1, 4, 6, 7, 8, 11], "Marwa": [0, 1, 4, 6, 7, 9, 11],
          "Kafi": [0, 2, 3, 5, 7, 9, 10], "Asavari": [0, 2, 3, 5, 7, 8, 10], "Bhairavi": [0, 1, 3, 5, 7, 8, 10],
          "Todi": [0, 1, 3, 6, 7, 8, 11]}

BG, GRID, DIM, TEXT = (11, 14, 20), (38, 44, 56), (92, 102, 120), (232, 236, 242)
VOICE, HOT = (255, 180, 84), (92, 207, 230)
STEM_COL = {"drums": (240, 113, 120), "bass": (199, 146, 234), "guitar": (127, 217, 98),
            "piano": (115, 208, 255), "other": (255, 209, 115)}


def find_font(names):
    roots = ["/usr/share/fonts", "/System/Library/Fonts", "/Library/Fonts",
             os.path.expanduser("~/Library/Fonts"), "C:/Windows/Fonts"]
    for n in names:
        for r in roots:
            hits = glob.glob(os.path.join(r, "**", n), recursive=True)
            if hits:
                return hits[0]
    return None


F_DEV = find_font(["NotoSansDevanagari-Bold.ttf", "Kohinoor.ttc", "KohinoorDevanagari*.ttc",
                   "DevanagariSangamMN.ttc", "Nirmala.ttf", "FreeSansBold.ttf"])
F_MONO = find_font(["JetBrainsMono-Medium.ttf", "JetBrainsMono-Regular.ttf", "Menlo.ttc",
                    "DejaVuSansMono.ttf", "consola.ttf"])
F_SANS = find_font(["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc", "arialbd.ttf"]) or F_MONO
_fonts = {}


def font(path, size):
    k = (path, size)
    if k not in _fonts:
        try:
            _fonts[k] = ImageFont.truetype(path, size * SS, layout_engine=ImageFont.Layout.RAQM)
        except Exception:
            _fonts[k] = ImageFont.truetype(path, size * SS)
    return _fonts[k]


def mix(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


# ================================================================= melody analysis
def track_pitch(y, fmin, fmax):
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=ASR, frame_length=2048, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0][:len(f0)]
    gate = B.db(rms) > max(np.percentile(B.db(rms), 95) - 32, -60)
    midi = librosa.hz_to_midi(f0)
    midi[~(voiced & gate)] = np.nan
    t = librosa.frames_to_time(np.arange(len(midi)), sr=ASR, hop_length=HOP)
    return t, midi, rms


def find_sa(midi, key_hint=None):
    """Tonic guess: the pitch class the voice rests on most, helped by its fifth and the song's key."""
    v = midi[~np.isnan(midi)]
    cents = (v * 100) % 100
    ang = np.angle(np.mean(np.exp(2j * np.pi * cents / 100)))
    offset = (ang / (2 * np.pi)) * 100                      # fine tuning of the singer, in cents
    pc = np.round(v - offset / 100).astype(int) % 12
    h = np.bincount(pc, minlength=12) / max(1, len(pc))
    score = h + 0.6 * np.roll(h, -7) + 0.25 * np.roll(h, -5)
    if key_hint is not None:
        score[key_hint] += 0.35 * score.max()
    return int(np.argmax(score)), offset / 100, h


def segment_notes(t, midi, rms, sa_midi, min_dur=0.055, passing=0.13):
    """Turn the pitch curve into discrete notes (semitones above Sa)."""
    rel = midi - sa_midi
    sm = rel.copy()
    for i in range(1, len(rel) - 1):                         # 3-frame median, ignoring gaps
        w = rel[i - 1:i + 2]
        if not np.isnan(w).any():
            sm[i] = np.median(w)
    semi = np.where(np.isnan(sm), -999, np.round(sm)).astype(int)
    dt = t[1] - t[0]
    runs, s = [], 0
    for i in range(1, len(semi) + 1):
        if i == len(semi) or semi[i] != semi[s]:
            if semi[s] != -999 and (i - s) * dt >= min_dur:
                runs.append([s, i, int(semi[s])])
            s = i
    lev = B.db(rms)

    def dips(a, b):
        """Frames inside [a, b) where the voice clearly re-attacks (level dips, then comes back)."""
        out, seg = [], lev[a:b]
        for i in range(3, len(seg) - 3):
            if seg[i] == seg[max(0, i - 3):i + 4].min() and seg[:i].max() - seg[i] > 5 and seg[i:].max() - seg[i] > 5:
                if not out or (a + i - out[-1]) * dt > 0.07:
                    out.append(a + i)
        return out

    # re-join the same note across a tiny wobble, unless the level dips in between (a re-attack)
    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2] and (r[0] - merged[-1][1]) * dt < 0.05:
            gap = lev[merged[-1][1]:r[0] + 1]
            if not len(gap) or lev[merged[-1][0]:merged[-1][1]].max() - gap.min() < 5:
                merged[-1][1] = r[1]
                continue
        merged.append(r)
    notes = []
    for a, b, sv in merged:
        cuts = [c for c in dips(a, b) if (c - a) * dt > 0.06 and (b - c) * dt > 0.06] if (b - a) * dt > 0.16 else []
        edges = [a] + cuts + [b]
        for s0, s1 in zip(edges[:-1], edges[1:]):
            notes.append({"start": float(t[s0]), "end": float(t[min(s1, len(t) - 1)]), "semi": sv})
    # notes outside the song's scale that only flash by are slides between real notes, not swaras
    w = np.zeros(12)
    for n in notes:
        w[n["semi"] % 12] += n["end"] - n["start"]
    scale = {i for i in range(12) if w[i] > 0.04 * w.sum()}
    return [n for n in notes if n["semi"] % 12 in scale or n["end"] - n["start"] >= passing]


def thaat_guess(notes):
    w = np.zeros(12)
    for n in notes:
        w[n["semi"] % 12] += n["end"] - n["start"]
    if w.sum() == 0:
        return None, []
    w /= w.sum()
    used = [i for i in range(12) if w[i] > 0.02]
    best = max(THAATS, key=lambda k: sum(w[i] for i in THAATS[k]))
    return (best, float(sum(w[i] for i in THAATS[best]))), used


def phrases(notes, max_len=8, gap=0.3):
    lines, cur = [], []
    for i, n in enumerate(notes):
        if cur and (n["start"] - notes[i - 1]["end"] > gap or len(cur) >= max_len):
            lines.append(cur)
            cur = []
        cur.append(i)
    if cur:
        lines.append(cur)
    return lines


def repeats(notes, gap=0.3):
    out, i = [], 0
    while i < len(notes):
        j = i
        while j + 1 < len(notes) and notes[j + 1]["semi"] == notes[i]["semi"] \
                and notes[j + 1]["start"] - notes[j]["end"] < gap:
            j += 1
        if j - i >= 2:
            out.append((i, j))
        i = j + 1
    return out


def drum_hits(drums_m):
    S = np.abs(librosa.stft(drums_m, n_fft=1024, hop_length=128)) ** 2
    f = librosa.fft_frequencies(sr=ASR, n_fft=1024)
    tt = librosa.frames_to_time(np.arange(S.shape[1]), sr=ASR, hop_length=128)
    hits = []
    for row, (lo, hi) in enumerate([(30, 120), (400, 4000), (7000, 11000)]):
        e = np.sqrt(S[(f >= lo) & (f < hi)].sum(axis=0))
        flux = np.maximum(0, np.diff(e, prepend=e[0]))
        flux = flux / (np.percentile(flux, 99.5) + 1e-9)
        pk = librosa.util.peak_pick(flux, pre_max=4, post_max=4, pre_avg=12, post_avg=12, delta=0.25, wait=8)
        hits.append([(float(tt[i]), float(min(1, flux[i]))) for i in pk])
    return hits


def envelope(y, fps=60):
    hop = ASR // fps
    r = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    d = B.db(r)
    return np.clip((d - (np.percentile(d, 98) - 36)) / 36, 0, 1), hop / ASR


# ================================================================= drawing
class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, f, fill, anchor="la"):
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=f, fill=fill, anchor=anchor)

    def line(self, pts, fill, width=1):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=int(width * SS), joint="curve")

    def rect(self, box, fill=None, outline=None, r=0, width=1):
        b = [v * SS for v in box]
        self.d.rounded_rectangle(b, radius=r * SS, fill=fill, outline=outline, width=int(width * SS))

    def dot(self, x, y, r, fill):
        self.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=fill)

    def width(self, s, f):
        return self.d.textlength(s, font=f) / SS

    def swara(self, cx, y, semi, size, fill, latin=None, latin_fill=None):
        """Devanagari swara centred at cx with komal underline, tivra tick and octave dots."""
        idx, octv = semi % 12, semi // 12
        f = font(F_DEV, size)
        w = self.width(DEV[idx], f)
        self.text((cx, y), DEV[idx], f, fill, anchor="mt")
        base = y + size * 1.28
        if idx in KOMAL:
            self.line([(cx - w * 0.38, base), (cx + w * 0.38, base)], fill, max(1.5, size / 18))
        if idx in TIVRA:
            self.line([(cx, y - size * 0.08), (cx, y + size * 0.2)], fill, max(1.5, size / 16))
        r = max(1.6, size / 16)
        if octv > 0:
            self.dot(cx + w * 0.5 + r * 2, y + size * 0.3, r, fill)
        if octv < 0:
            self.dot(cx, base + r * 3, r, fill)
        if latin:
            self.text((cx, base + size * 0.3), LAT[idx], font(F_MONO, max(9, int(size * 0.3))),
                      latin_fill or fill, anchor="mt")
        return w


def render(args, song, out, rep, stems_m, mel, total_dur):
    notes, t, midi_rel = mel["notes"], mel["t"], mel["rel"]
    lines, reps = phrases(notes), repeats(notes)
    line_of = {i: li for li, ln in enumerate(lines) for i in ln}
    starts = np.array([n["start"] for n in notes]) if notes else np.array([])
    # robust range: a few octave-jump errors must not stretch the ladder
    if notes:
        sv = np.array([n["semi"] for n in notes])
        wt = np.array([n["end"] - n["start"] for n in notes])
        order = np.argsort(sv)
        cum = np.cumsum(wt[order]) / wt.sum()
        r_lo, r_hi = int(sv[order][np.searchsorted(cum, 0.015)]), int(sv[order][min(len(sv) - 1, np.searchsorted(cum, 0.985))])
    else:
        r_lo, r_hi = 0, 12
    scale = mel["scale"]
    used = [x for x in range(r_lo, r_hi + 1) if x % 12 in scale] or [0]
    lo, hi = r_lo - 1.5, r_hi + 1.5
    span = r_hi - r_lo
    peak_speed = max((int(np.sum((starts >= s) & (starts < s + 1))) for s in starts), default=0)

    present = [n for n in ["drums", "bass", "guitar", "piano", "other"] if rep["stems"][n]["present"]]
    env = {n: envelope(stems_m[n]) for n in present}
    hits = drum_hits(stems_m["drums"]) if "drums" in present else None
    bass_notes = mel.get("bass_notes", [])
    bars = rep.get("bar_starts", [])
    chords = rep.get("chords_per_unit", [])
    c_edges = np.array(bars + [total_dur])
    if rep.get("chord_unit") == "half bar":
        c_edges = np.sort(np.concatenate([c_edges, (c_edges[:-1] + c_edges[1:]) / 2]))

    def label(n):
        idn = rep["instruments"].get(n)
        if not idn:
            return n.upper()
        name = idn["detail"] if idn["confidence"] != "unsure" and idn["detail_p"] >= 0.5 * idn["family_p"] else idn["call"]
        return f"{n.upper()} · {name}"

    # geometry (in output pixels)
    PX0, PX1, PY0, PY1 = 84, 632, 190, 640
    WIN, HEAD = 5.0, 0.74
    xs = lambda tt, now: PX0 + (tt - now + WIN * HEAD) / WIN * (PX1 - PX0)
    ys = lambda sv: PY1 - (sv - lo) / (hi - lo) * (PY1 - PY0)
    LY0 = 905
    lane_h = min(72, (1215 - LY0 - 20) / max(1, len(present)))
    sa_name = NOTE_SHARP[mel["sa_pc"]]
    title = args.title or os.path.splitext(os.path.basename(song))[0].replace("_", " ")
    thaat = mel["thaat"]

    t0, t1 = args.start or 0.0, min(args.end or total_dur, total_dur)
    nframes = int((t1 - t0) * args.fps)
    mp4 = os.path.join(out, "sargam.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(args.fps), "-i", "-", "-ss", str(t0), "-t", str(t1 - t0), "-i", song,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fm, fs, fl = font(F_MONO, 13), font(F_MONO, 11), font(F_SANS, 22)

    for fi in range(nframes):
        now = t0 + fi / args.fps
        c = Canvas()
        sung = int(np.searchsorted(starts, now, side="right"))
        cur = sung - 1 if sung and now <= notes[sung - 1]["end"] + 0.12 else None

        # ---- header
        c.text((40, 44), title.upper(), fl, TEXT)
        c.text((W - 40, 52), f"SA = {sa_name}  ·  {rep['tempo_bpm']:.0f} BPM", fm, VOICE, anchor="ra")
        chip = f"{len(notes)} SWARAS · PEAK {peak_speed}/s · RANGE {span} SEMITONES"
        cw = c.width(chip, fs) + 20
        c.rect([40, 92, 40 + cw, 116], outline=mix(HOT, 0.5), r=4)
        c.text((50, 98), chip, fs, HOT)
        c.text((W - 40, 98), "VOICE" if args.lead == "vocals" else f"LEAD: {args.lead.upper()}", fs, DIM, anchor="ra")

        # ---- pitch ladder
        cur_semi = notes[cur]["semi"] if cur is not None else None
        for sv in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
            if sv not in used:
                continue
            y = ys(sv)
            on = sv == cur_semi
            is_sa = sv % 12 == 0
            c.line([(PX0, y), (PX1, y)], mix(VOICE, 0.55) if on else mix(TEXT, 0.22 if is_sa else 0.09), 1.4 if on else 1)
            col = VOICE if on else (mix(TEXT, 0.75) if sv in used else DIM)
            if (PY1 - PY0) / (hi - lo) < 16 and not on and sv % 12 not in (0, 7):
                continue                                   # crowded ladder: label only Sa, Pa and the live note
            c.swara(PX0 - 30, y - 13, sv, 17, col)
            note = NOTE_SHARP[(mel["sa_pc"] + sv) % 12]
            c.text((PX1 + 12, y - 7), f"{LAT[sv % 12]}", fs, col)
            c.text((PX1 + 50, y - 7), note, fs, mix(col, 0.6))
        px = xs(now, now)
        c.line([(px, PY0), (px, PY1)], mix(TEXT, 0.12), 1)

        # ---- repeat brackets
        for a, b in reps:
            if notes[a + 2]["start"] <= now and notes[a]["start"] > now - WIN * HEAD:
                done = sum(1 for k in range(a, b + 1) if notes[k]["start"] <= now)
                x0, x1 = xs(notes[a]["start"], now), min(px, xs(notes[b]["end"], now))
                y = ys(notes[a]["semi"]) - 22
                c.line([(x0, y + 6), (x0, y), (x1, y), (x1, y + 6)], mix(HOT, 0.7), 1.2)
                c.text(((x0 + x1) / 2, y - 16), f"×{done}", fs, HOT, anchor="ma")

        # ---- voice pitch line (only what has been sung)
        i0, i1 = np.searchsorted(t, now - WIN * HEAD), np.searchsorted(t, now)
        seg = []
        for i in range(i0, i1):
            if np.isnan(midi_rel[i]) or not (lo <= midi_rel[i] <= hi):
                if len(seg) > 1:
                    c.line(seg, VOICE, 3)
                seg = []
            else:
                seg.append((xs(t[i], now), ys(midi_rel[i])))
        if len(seg) > 1:
            c.line(seg, VOICE, 3)
        if seg and now - t[i1 - 1] < 0.08:
            c.dot(seg[-1][0], seg[-1][1], 6, VOICE)
            c.dot(seg[-1][0], seg[-1][1], 2.5, BG)

        # ---- swara text (current line + next)
        if lines:
            li = line_of[min(sung, len(notes) - 1)] if sung < len(notes) else len(lines) - 1
            if sung and sung < len(notes) and now < notes[sung]["start"] and cur is not None:
                li = line_of[cur]
            for row, l in enumerate([li, li + 1]):
                if l >= len(lines):
                    continue
                ids = lines[l]
                size = 46 if row == 0 else 34
                gap = size * 1.25
                x = W / 2 - gap * (len(ids) - 1) / 2
                y = 668 + row * 108
                for k in ids:
                    if k == cur:
                        col = VOICE
                    elif k < sung:
                        col = TEXT if row == 0 else mix(TEXT, 0.7)
                    else:
                        col = mix(TEXT, 0.28)
                    c.swara(x, y, notes[k]["semi"], size, col, latin=row == 0,
                            latin_fill=VOICE if k == cur else mix(TEXT, 0.4))
                    x += gap

        # ---- counter + progress
        c.text((40, 858), f"SWARAS {'SUNG' if args.lead == 'vocals' else 'PLAYED'}  {sung:03d} / {len(notes):03d}", fs, DIM)
        c.rect([250, 862, W - 40, 866], fill=GRID, r=2)
        c.rect([250, 862, 250 + (W - 290) * (now - t0) / max(1e-6, t1 - t0), 866], fill=VOICE, r=2)

        # ---- instruments
        c.text((40, LY0 - 26), "INSTRUMENTS", fs, DIM)
        ci = int(np.searchsorted(c_edges, now, side="right")) - 1
        if 0 <= ci < len(chords) and chords[ci]:
            c.text((W - 40, LY0 - 28), f"CHORD  {chords[ci]}", fm, TEXT, anchor="ra")
        for k, n in enumerate(present):
            col = STEM_COL[n]
            y0 = LY0 + k * lane_h
            mid = y0 + lane_h * 0.62
            e, edt = env[n]
            lvl = e[min(len(e) - 1, int(now / edt))]
            c.dot(46, y0 + 8, 4, mix(col, 0.25 + 0.75 * lvl))
            c.text((58, y0 + 1), label(n)[:58], fs, mix(col, 0.55 + 0.45 * lvl))
            a_, b_ = now - WIN * HEAD, now + WIN * (1 - HEAD)
            if n == "drums" and hits:
                for row, hh in enumerate(hits):
                    yy = y0 + lane_h * (0.42 + 0.2 * row)
                    for ht, hv in hh:
                        if a_ <= ht <= b_:
                            past = ht <= now
                            flash = max(0, 1 - (now - ht) / 0.12) if past else 0
                            r = (2.2 + 2.2 * hv) * (1 + 0.8 * flash) * (1.25 if row == 0 else 1)
                            c.dot(xs(ht, now), yy, r, mix(col, (0.9 if past else 0.3)))
                continue
            amp = lane_h * 0.26
            j0, j1 = max(0, int(a_ / edt)), min(len(e), int(b_ / edt) + 1)
            step = max(1, (j1 - j0) // 300)
            for part, a_col in ((slice(j0, min(j1, int(now / edt) + 1)), 0.85), (slice(int(now / edt), j1), 0.25)):
                idx = list(range(part.start, part.stop, step))
                if len(idx) > 1:
                    top = [(xs(j * edt, now), mid - e[j] * amp - 0.5) for j in idx]
                    bot = [(xs(j * edt, now), mid + e[j] * amp + 0.5) for j in idx]
                    c.d.polygon([(x * SS, y * SS) for x, y in top + bot[::-1]], fill=mix(col, a_col))
            if n == "bass":
                for bn in bass_notes:
                    if bn["start"] <= now < bn["end"] + 0.15:
                        c.text((W - 40, y0 + 1), f"{LAT[bn['semi'] % 12]} · {NOTE_SHARP[(mel['sa_pc'] + bn['semi']) % 12]}",
                               fs, col, anchor="ra")
                        break
        c.line([(px, LY0 - 6), (px, LY0 + lane_h * len(present))], mix(TEXT, 0.18), 1)

        # ---- footer
        foot = "underline = komal · tick above = tivra · dot = other octave"
        if thaat:
            foot = f"notes fit {thaat[0]} thaat ({thaat[1]:.0%}) · " + foot
        c.text((W / 2, 1240), foot, font(F_MONO, 10), DIM, anchor="ma")

        ff.stdin.write(c.im.reduce(SS).tobytes())
        if fi % (args.fps * 10) == 0:
            print(f"      rendering {now - t0:5.0f}s / {t1 - t0:.0f}s")
            sys.stdout.flush()
    ff.stdin.close()
    ff.wait()
    return mp4, {"peak_swaras_per_sec": peak_speed, "range_semitones": span}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--sa", default=None, help="force the tonic, e.g. C or F#")
    ap.add_argument("--title", default=None)
    ap.add_argument("--lead", default=None, choices=["vocals", "guitar", "piano", "other", "bass"],
                    help="which part to draw as the melody line (default: vocals, or the lead instrument if the voice is spoken)")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true", help="only write melody.json and sargam.txt")
    args = ap.parse_args()
    base = os.path.splitext(os.path.basename(args.song))[0]
    out = args.out or os.path.join("breakdowns", base)
    rj = os.path.join(out, "report.json")
    if not os.path.exists(rj) or not os.path.exists(os.path.join(out, "stems", "vocals.wav")):
        print("No breakdown found for this song yet; running it first.")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "breakdown.py"),
                        args.song, "-o", out], check=True)
    rep = json.load(open(rj))
    stems_m = {}
    for n in B.STEM_ORDER:
        y, sr = sf.read(os.path.join(out, "stems", f"{n}.wav"), dtype="float32", always_2d=True)
        stems_m[n] = librosa.resample(y.mean(1), orig_sr=sr, target_sr=ASR)
    total = len(stems_m["vocals"]) / ASR

    lead = args.lead
    if not lead:
        spoken = (rep.get("voice") or "").startswith("spoken")
        sung = rep["stems"]["vocals"]["present"] and not spoken
        melodic = [n for n in ("guitar", "other", "piano") if rep["stems"][n]["present"]]
        lead = "vocals" if sung or not melodic else max(melodic, key=lambda n: rep["stems"][n]["level_vs_mix_db"])
        if lead != "vocals":
            print(f"      no singing in this track, so the melody line follows the {lead} part (change with --lead)")
    args.lead = lead
    print(f"[1/3] tracking the pitch of: {lead}")
    sys.stdout.flush()
    t, midi, rms = track_pitch(stems_m[lead], 30 if lead == "bass" else 75, 300 if lead == "bass" else 1200)
    if np.isnan(midi).all():
        sys.exit("No clear pitch found in that part.")
    key_pc = NOTE_SHARP.index({"D♭": "C#", "E♭": "D#", "G♭": "F#", "A♭": "G#", "B♭": "A#"}.get(
        rep["key"].split()[0], rep["key"].split()[0]))
    sa_pc, off, hist = find_sa(midi, key_pc)
    sa_auto = sa_pc
    if args.sa:
        sa_pc = NOTE_SHARP.index(args.sa.strip().upper().replace("DB", "C#").replace("EB", "D#")
                                 .replace("GB", "F#").replace("AB", "G#").replace("BB", "A#"))
    med = np.nanmedian(midi)
    sa_midi = sa_pc + off + 12 * np.floor((med - sa_pc - off) / 12)      # Sa of the octave the voice sits in
    if med - sa_midi > 9:                                                # mostly above Dha: treat next Sa as home
        sa_midi += 0
    print(f"      Sa = {NOTE_SHARP[sa_pc]}" + ("" if args.sa else " (auto; override with --sa)")
          + f", singer is {off * 100:+.0f} cents from A440")

    print("[2/3] finding the notes")
    sys.stdout.flush()
    notes = segment_notes(t, midi, rms, sa_midi)
    if notes:                                   # put the octave the melody lives in at the centre (no marks)
        sv = np.array([n["semi"] for n in notes])
        wt = np.array([n["end"] - n["start"] for n in notes])
        o = np.argsort(sv)
        med_semi = sv[o][np.searchsorted(np.cumsum(wt[o]) / wt.sum(), 0.5)]
        shift = 12 * int(np.floor(med_semi / 12))
        if shift:
            sa_midi += shift
            for n in notes:
                n["semi"] -= shift
            med_semi -= shift
        # pitch trackers jump octaves on instruments; if the range is implausibly wide, fold it back
        cum = np.cumsum(wt[o]) / wt.sum()
        p_lo, p_hi = sv[o][np.searchsorted(cum, 0.015)], sv[o][min(len(sv) - 1, np.searchsorted(cum, 0.985))]
        if p_hi - p_lo > 24:
            fold_lo, fold_hi = med_semi - 9, med_semi + 14
            print("      very wide pitch range found (likely octave errors); folding it into two octaves")
            for n in notes:
                while n["semi"] < fold_lo:
                    n["semi"] += 12
                while n["semi"] >= fold_hi:
                    n["semi"] -= 12
            midi = midi.copy()
            relc = midi - sa_midi
            relc = np.where(relc < fold_lo, relc + 12 * np.ceil((fold_lo - relc) / 12), relc)
            relc = np.where(relc >= fold_hi, relc - 12 * np.ceil((relc - fold_hi + 0.01) / 12), relc)
            midi = relc + sa_midi
    thaat, used = thaat_guess(notes)
    rel = midi - sa_midi
    disp = rel.copy()
    for i in range(3, len(rel) - 3):                      # 7-frame median: steadier line, slides kept
        w = rel[i - 3:i + 4]
        if not np.isnan(w).any():
            disp[i] = np.median(w)
    mel = {"t": t, "rel": disp, "notes": notes, "sa_pc": sa_pc, "thaat": thaat, "scale": set(used)}
    if rep["stems"]["bass"]["present"]:
        bt, bm, br = track_pitch(stems_m["bass"], 30, 300)
        if not np.isnan(bm).all():
            mel["bass_notes"] = segment_notes(bt, bm, br, sa_midi, min_dur=0.09)

    lines = phrases(notes)
    with open(os.path.join(out, "sargam.txt"), "w") as f:
        f.write(f"Sa = {NOTE_SHARP[sa_pc]} · {len(notes)} swaras\n")
        f.write("lowercase = komal, Ma' = tivra, + / - = octave above / below\n\n")
        for ln in lines:
            sw = " ".join(LAT[notes[i]["semi"] % 12] + ("+" if notes[i]["semi"] >= 12 else "-" if notes[i]["semi"] < 0 else "")
                          for i in ln)
            f.write(f"{B.ts(notes[ln[0]]['start']):>5}  {sw}\n")
    stats = {}
    mp4 = None
    if not args.no_video:
        print("[3/3] rendering the video")
        sys.stdout.flush()
        mp4, stats = render(args, args.song, out, rep, stems_m, mel, total)
    json.dump({"sa": NOTE_SHARP[sa_pc], "sa_auto_guess": NOTE_SHARP[sa_auto], "lead": lead, "cents_from_a440": off * 100,
               "thaat_guess": thaat, "swaras_used": [LAT[i] for i in used], "count": len(notes),
               "notes": [{**n, "swara": LAT[n["semi"] % 12], "octave": n["semi"] // 12,
                          "note": NOTE_SHARP[(sa_pc + n["semi"]) % 12]} for n in notes], **stats},
              open(os.path.join(out, "melody.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\nDone → {mp4 or os.path.join(out, 'sargam.txt')}")


if __name__ == "__main__":
    main()
