# Sound Breakdown

Upload any song and get a plain-language breakdown of how it's built: the instruments, what each one sounds like, the beat, the tuning, and where the big moments are. It's made for describing music you love to a person or an AI.

Free and open-source only:

- [Demucs](https://github.com/facebookresearch/demucs) splits the song into drums, bass, vocals, guitar, piano and other.
- [LAION CLAP](https://huggingface.co/laion/larger_clap_music_and_speech) listens to each separated stem and names the instrument (for example "808 sub bass", "palm muted electric guitar", "tabla"), plus genre, mood and production style.
- [MIT AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) tells singing from narration and finds sound effects.
- [librosa](https://librosa.org) measures tempo, key, tuning, chords, beats, energy and timbre.

## Setup (once)

```bash
./setup.sh
```

Needs Python 3.10+ and `ffmpeg`. On the first run it downloads the models from `dl.fbaipublicfiles.com` and `huggingface.co`.

## Use

```bash
./run.sh songs/my_song.mp3
```

This writes `breakdowns/my_song/`:

| File | What's in it |
|---|---|
| `report.md` | The full breakdown |
| `timeline.png` | Energy curve with sections, peaks and drops, plus a heatmap of when each instrument plays |
| `report.json` | All the numbers, for other tools |
| `stems/*.wav` | drums, bass, vocals, guitar, piano, other as separate files |

Use `--no-stems` to skip saving the WAVs.

## What the report covers

- **In one line:** genre, tempo, key, mood and the instruments
- **Overview:** tempo, key, tuning (A = 440, 432, …), genre, mood, production style, voice, main chord loop, each with how sure it is
- **Instruments:** loudest first. Each gets an instrument family with a confidence level, the closest specific sounds, a description (bright or dark, clean or gritty, plucky or sustained, wide or narrow), when it first comes in, and the notes it uses. It warns when a stem is really another instrument bleeding in.
- **Song map:** sections on bar lines with their role, energy, what's playing and the chords
- **Key moments:** the biggest hits, drops and pull-backs, and which instruments come in or drop out there
- **The beat:** kick, snare and hi-hat patterns on a 16-step grid
- **Mix and production:** loudness, loud/soft contrast, tonal balance, stereo width
- **Sound design:** non-music sounds with timestamps
- **Prompts:** ready-to-paste Suno prompts (Simple mode sentence and Advanced mode Styles)

## How reliable each part is

| Measured (reliable) | Model guesses (treat as hints) |
|---|---|
| Tempo, beat pattern, energy curve, drops, when instruments enter, tuning in cents, bass notes, timbre numbers, chord loops that repeat | Genre, mood, instrument names, guitar tuning, section names, chords in sections with no steady loop |

Known limits: piano and guitar stems bleed into each other, sound effects and narration confuse the drum and genre calls, quiet changes inside a section (a soft melody joining) are not split out, and it has been checked against one synthetic song with known answers and one real soundtrack.

Stems are reused if they already exist for a song; pass `--fresh` to redo the separation.

## Sargam video

```bash
../.venv/bin/python visualize.py songs/my_song.mp3 --title "My song"
```

Makes `breakdowns/my_song/sargam.mp4`, a vertical 720×1280 video with the song's audio:

- the voice's pitch drawn live on a Sa–Re–Ga–Ma ladder (with Western note names alongside)
- the swaras in Devanagari and Latin, lighting up as they are sung, with komal, tivra and octave marks
- repeat brackets (×3, ×4), a swara counter, peak speed and range
- a lane per instrument: drum hits, bass with its current note, and the other parts, each labelled with what the breakdown identified, plus the current chord

It also writes `sargam.txt` (the notation with timestamps) and `melody.json` (every note). The tonic is found automatically; force it with `--sa C`. Use `--start` and `--end` (seconds) to render part of a song, or `--no-video` for the notation only. A Devanagari font is needed (Noto Sans Devanagari, or the ones built into macOS and Windows).

Checked against one clip with published notation: it found the same Sa and the same seven swaras, and counted 255 swaras where the reference shows 270. Very fast runs lose some notes, and lyric words are not detected.

## Test

```bash
../.venv/bin/python test/make_test_song.py   # synthetic 120 BPM song with a known beat
./run.sh test/test_song.wav -o test/out
```
