# Sound Breakdown

Upload any song and get a plain-language breakdown of how it's built: the instruments, what each one sounds like, the beat, the tuning, and where the big moments are. It's made for describing music you love to a person or an AI.

Free and open-source only: [Demucs](https://github.com/facebookresearch/demucs) (stem separation), [MIT AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (instrument / genre / mood / sound recognition), [librosa](https://librosa.org) (tempo, key, tuning, beats, energy, timbre).

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

- **Overview:** tempo, key (plus its relative major/minor), genre, mood, instruments (tagged with the voice removed), voice type (singing or narration), sound effects with timestamps
- **Tuning:** reference pitch (A = 440, 432, …), measured in cents from the instruments, plus a guess at the guitar tuning (standard, drop D, …)
- **Song map:** sections (intro, build, peak, breakdown, outro), highest points, drops, pull-backs, and when each instrument comes in or drops out
- **The beat:** kick, snare and hi-hat patterns on a 16-step grid
- **Each instrument:** brightness, grit, plucky or sustained, busy or sparse, stereo width, pan, dynamics, pitch range, notes used
- **Quick description:** one paragraph you can paste into any AI

## How reliable each part is

| Measured (reliable) | Model guesses (treat as hints) |
|---|---|
| Tempo, beat pattern, energy curve, drops, when instruments enter, tuning in cents, bass notes, timbre numbers | Genre, mood, instrument names, guitar tuning, section names |

Known limits: the piano and guitar stems bleed into each other, sound effects can end up in the drums stem, and non-Western instruments are recognised poorly.

## Test

```bash
../.venv/bin/python test/make_test_song.py   # synthetic 120 BPM song with a known beat
./run.sh test/test_song.wav -o test/out
```
