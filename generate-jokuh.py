#!/usr/bin/env python3
"""Generate Jokuh pitch video voiceover using Matthew's cloned voice — Approach C."""

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
OUTPUT_DIR = os.path.expanduser("~/Desktop/jokuh-video/audio-matthew-tts")

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Emotion instruct for keynote delivery
INSTRUCT = "Speaking with conviction and authority, like presenting a keynote to investors. Deliberate pacing with dramatic pauses. Confident, visionary tone."

SEGMENTS = [
    ("seg01", "Every once in a while... a shift comes along that redefines the entire operating system layer. In nineteen eighty-four, the Mac gave us the graphical interface. In two thousand and seven, the iPhone put the internet in our pockets. Today... I want to tell you about the next one."),
    ("seg02", "Look at your phone right now. You've got eighty, maybe a hundred apps. None of them talk to each other. You are the integration layer. And every single one is watching you. You're not the customer. You're the product."),
    ("seg03", "They bolt AI on top. Siri still can't order you dinner. Google still tracks everything. It's duct tape on a broken foundation. Nobody has rethought the foundation itself. Until now."),
    ("seg04", "Jokuh is a new kind of operating system. Privacy-first. AI-native. Intent-driven. One system. Built from the ground up."),
    ("seg05", "You say: Order my favorite pizza and text my mom. One sentence. No apps opened. No data harvested. No friction. That's not an improvement. That's a reinvention."),
    ("seg06", "Three layers. Decentralized identity. End-to-end encryption. Post-quantum cryptography. AI is powerful. Centralized AI is dangerous. We built the antidote."),
    ("seg07", "Federated learning. Every user makes the system smarter without exposing their data. Swarm intelligence."),
    ("seg08", "Collective intelligence in nature. We've built that for humans and AI."),
    ("seg09", "Microsoft, two hundred and forty-five billion. Apple, three hundred and ninety-one billion. Google, three hundred and fifty billion. Just the OS layer. We're defining a new category."),
    ("seg10", "The only fully distributed, native agentic OS with privacy at the foundation and a built-in token economy."),
    ("seg11", "SaaS plus Commerce plus Hardware plus Enterprise."),
    ("seg12", "Twenty twenty-five, Software. Twenty twenty-six, Agent Economy. Twenty twenty-seven, Hardware. Twenty twenty-eight, Full Decentralization."),
    ("seg13", "Twelve million dollar seed round. Every generation gets one shot at redefining the OS layer. This is that shot. That's Jokuh."),
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
    print(f"Model loaded. Using voice reference: {REF_AUDIO}")
    print(f"Instruct: {INSTRUCT}")
    print(f"Output: {OUTPUT_DIR}\n")

    for i, (seg_name, text) in enumerate(SEGMENTS):
        print(f"[{i+1}/{len(SEGMENTS)}] Generating {seg_name}...")
        start = time.time()

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
            dest = os.path.join(OUTPUT_DIR, f"{seg_name}.wav")
            if os.path.exists(source):
                shutil.move(source, dest)
                elapsed = time.time() - start
                print(f"  -> Saved {dest} ({elapsed:.1f}s)")
            else:
                print(f"  !! No audio generated for {seg_name}")

        except Exception as e:
            print(f"  !! Error: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        gc.collect()

    # Concatenate all segments into one voiceover
    print("\nConcatenating all segments...")
    wavs = sorted([
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("seg") and f.endswith(".wav")
    ])

    if wavs:
        concat_file = os.path.join(OUTPUT_DIR, "concat.txt")
        with open(concat_file, "w") as f:
            for w in wavs:
                f.write(f"file '{w}'\n")

        vo_path = os.path.expanduser("~/Desktop/jokuh-video/JOKUH-VOICEOVER-MATTHEW-V2.mp3")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:a", "libmp3lame", "-b:a", "192k", vo_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  -> Voiceover: {vo_path}")
        except subprocess.CalledProcessError as e:
            print(f"  !! Concat failed: {e}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
