#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np


def run(cmd):
    subprocess.run(cmd, check=True)


def load_wav(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Expected 16-bit PCM WAV")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return rate, audio


def detect(rate, audio, count=6):
    frame_s = 0.40
    hop_s = 0.10
    frame = max(1, int(rate * frame_s))
    hop = max(1, int(rate * hop_s))
    if len(audio) < frame:
        raise ValueError("Audio is too short")

    starts = np.arange(0, len(audio) - frame + 1, hop)
    rms = np.sqrt(np.array([
        np.mean(audio[s:s + frame] ** 2) for s in starts
    ]) + 1e-12)
    times = (starts + frame / 2) / rate

    baseline_frames = max(10, int(18 / hop_s))
    baseline = np.empty_like(rms)
    for i in range(len(rms)):
        left = max(0, i - baseline_frames)
        window = rms[left:i] if i > left else rms[:1]
        baseline[i] = np.median(window)

    global_floor = max(float(np.percentile(rms, 40)), 1e-5)
    relative_db = 20 * np.log10((rms + 1e-6) / (baseline + 1e-6))
    absolute_db = 20 * np.log10((rms + 1e-6) / global_floor)

    sustain_n = max(1, int(1.2 / hop_s))
    kernel = np.ones(sustain_n, dtype=np.float32) / sustain_n
    sustained = np.convolve(absolute_db, kernel, mode="same")
    score = relative_db * 1.35 + sustained * 0.65

    duration = len(audio) / rate
    edge_guard = 4.0 if duration < 90.0 else 20.0
    eligible = (times >= edge_guard) & (times <= max(edge_guard, duration - edge_guard))
    indices = np.argsort(np.where(eligible, score, -np.inf))[::-1]

    chosen = []
    min_separation = 24.0
    for idx in indices:
        if not np.isfinite(score[idx]):
            continue
        t = float(times[idx])
        if all(abs(t - x["peak_seconds"]) >= min_separation for x in chosen):
            chosen.append({
                "rank": len(chosen) + 1,
                "peak_seconds": round(t, 3),
                "clip_start": round(max(0.0, t - 12.0), 3),
                "clip_duration": round(min(20.0, duration - max(0.0, t - 12.0)), 3),
                "score": round(float(score[idx]), 3),
                "relative_db": round(float(relative_db[idx]), 3),
            })
            if len(chosen) >= count:
                break
    return duration, chosen


def extract_candidates(video, candidates, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in candidates:
        out = output_dir / f"candidate_{item['rank']:02d}.mp4"
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(item["clip_start"]), "-i", str(video),
            "-t", str(item["clip_duration"]),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(out)
        ])


def render_vertical(candidate, output):
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    filt = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=38,eq=brightness=-0.42[bg2];"
        "[fg]scale=1080:-2[main];"
        "[bg2][main]overlay=(W-w)/2:400[tmp];"
        "[tmp]drawbox=x=55:y=1060:w=970:h=285:color=black@0.62:t=fill,"
        f"drawtext=fontfile={font}:text='HOW DID HE':"
        "fontcolor=white:fontsize=82:x=(w-text_w)/2:y=1100:"
        "shadowcolor=black@0.8:shadowx=4:shadowy=4,"
        f"drawtext=fontfile={font}:text='SURVIVE THAT?':"
        "fontcolor=white:fontsize=82:x=(w-text_w)/2:y=1200:"
        "shadowcolor=black@0.8:shadowx=4:shadowy=4[v]"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(candidate), "-filter_complex", filt,
        "-map", "[v]", "-map", "0:a?",
        "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    rate, audio = load_wav(args.wav)
    duration, candidates = detect(rate, audio, args.count)
    if not candidates:
        raise RuntimeError("No audio peaks detected")

    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    extract_candidates(Path(args.video), candidates, candidate_dir)
    render_vertical(candidate_dir / "candidate_01.mp4", output_dir / "brawlhalla_trial.mp4")

    manifest = {
        "source_duration": round(duration, 3),
        "method": "caster vocal-band loudness peaks",
        "title": "HOW DID HE SURVIVE THAT?",
        "selected_rank": 1,
        "candidates": candidates,
    }
    (output_dir / "peaks.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
