#!/usr/bin/env python3
"""Generate Coldstar MCA reminder voiceover using Matthew's cloned voice.

Short voice memo (~45 seconds) reminding devsyrem about the 3 items
that still need his personal attention for the MCA grant package.
"""

import os
import sys
import shutil
import time
import gc
import subprocess
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODELS_DIR = os.path.join(os.getcwd(), "models")
VOICES_DIR = os.path.join(os.getcwd(), "voices")
OUTPUT_DIR = os.path.expanduser("~/Desktop/Coldstar-MCA-Package")

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Confident, direct, collaborative tone
INSTRUCT = "Speaking casually but clearly, like leaving a quick voice memo for a collaborator. Friendly, direct, no filler. Get to the point."

SEGMENTS = [
    ("s01-hey",
     "Hey Syrem. Quick note on the MCA package. "
     "Everything's ready to go on our end. PDFs, pitch deck, strategy doc, budget template, all of it. "
     "But there are three things that only you can do."),

    ("s02-video",
     "First. Record a two-minute project intro video. "
     "Doesn't need to be fancy. Screen recording of the CLI in action is perfect. "
     "Show a cold-signed transaction from start to finish. "
     "Walk through the USB initialization, signing, and broadcast. "
     "Grant committees love seeing working product."),

    ("s03-partners",
     "Second. If you have any contacts at Squads, Backpack, Solflare, or any Solana project, "
     "reach out and ask for a letter of interest. "
     "Even a short email saying they'd be interested in Coldstar integration goes a long way with foundations. "
     "It's not required, but it's a strong signal."),

    ("s04-metrics",
     "Third. Pull your community metrics. "
     "Twitter follower count, Discord or Telegram members if you have them, GitHub stars and forks. "
     "The GitHub stats are already in the package, but your social numbers should be in there too."),

    ("s05-close",
     "Once you've got those three, we're ready to send the outreach message to MCA "
     "and kick off the free assessment. Let me know if you need anything."),
]


def get_smart_path(folder_name):
    full_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full_path):
        return None
    snapshots_dir = os.path.join(full_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subfolders = [f for f in os.listdir(snapshots_dir) if not f.startswith('.')]
        if subfolders:
            return os.path.join(snapshots_dir, subfolders[0])
    return full_path


def generate_silence(duration_sec, output_path, sample_rate=24000):
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(duration_sec),
        "-c:a", "pcm_s16le", output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(REF_TEXT_FILE, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()

    model_folder = "Qwen3-TTS-12Hz-1.7B-Base-8bit"
    model_path = get_smart_path(model_folder)
    if not model_path:
        print(f"Error: Model not found at {os.path.join(MODELS_DIR, model_folder)}")
        sys.exit(1)

    print(f"Loading Base model from {model_path}...")
    model = load_model(model_path)
    print(f"Model loaded. Using voice: {REF_AUDIO}")
    print(f"Instruct: {INSTRUCT}\n")

    total = len(SEGMENTS)
    temp_files = []

    for i, (seg_name, text) in enumerate(SEGMENTS, 1):
        print(f"[{i}/{total}] Generating {seg_name}...")

        temp_dir = f"temp_{int(time.time())}_{seg_name}"
        try:
            generate_audio(
                model=model,
                text=text,
                ref_audio=REF_AUDIO,
                ref_text=ref_text,
                instruct=INSTRUCT,
                output_path=temp_dir,
            )

            source = os.path.join(temp_dir, "audio_000.wav")
            dest = os.path.join(OUTPUT_DIR, f"memo-{seg_name}.wav")
            if os.path.exists(source):
                shutil.move(source, dest)
                temp_files.append(dest)
                print(f"  -> Saved {dest}")
            else:
                print(f"  !! No audio generated for {seg_name}")
        except Exception as e:
            print(f"  !! Error: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

    # Generate silence gap
    silence_path = os.path.join(OUTPUT_DIR, "memo-silence.wav")
    print("\nGenerating silence gap...")
    generate_silence(0.6, silence_path)

    # Concatenate
    print("Concatenating segments...")
    concat_file = os.path.join(OUTPUT_DIR, "memo-concat.txt")
    wavs = sorted([f for f in temp_files if os.path.exists(f)])

    with open(concat_file, "w") as f:
        for j, w in enumerate(wavs):
            f.write(f"file '{w}'\n")
            if j < len(wavs) - 1:
                f.write(f"file '{silence_path}'\n")

    output_wav = os.path.join(OUTPUT_DIR, "mca-reminder-voicememo.wav")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:a", "pcm_s16le", output_wav
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> WAV: {output_wav}")
    except subprocess.CalledProcessError as e:
        print(f"  !! Concat failed: {e}")

    # M4A
    output_m4a = os.path.join(OUTPUT_DIR, "mca-reminder-voicememo.m4a")
    cmd = [
        "ffmpeg", "-y", "-i", output_wav,
        "-c:a", "aac", "-b:a", "192k", output_m4a
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> M4A: {output_m4a}")
    except subprocess.CalledProcessError as e:
        print(f"  !! M4A conversion failed: {e}")

    # Cleanup temp segment files
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    for f in [silence_path, concat_file, output_wav]:
        if os.path.exists(f):
            os.remove(f)

    print("\nVoice memo generation complete!")

    # Duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", output_m4a],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        mins = int(duration // 60)
        secs = int(duration % 60)
        print(f"Duration: {mins}:{secs:02d}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
