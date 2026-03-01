#!/usr/bin/env python3
"""Generate spoken word vocals over a dark cinematic trap beat using Matthew's voice clone."""
import os
import shutil
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
OUTPUT_DIR = os.path.expanduser("~/Desktop/AI-Music/vocals")

REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

INSTRUCT = "Dark, intense spoken word delivery. Low and deliberate, like narrating a film trailer. Confident, slightly menacing undertone. Not rushed — let each word land."

SEGMENTS = [
    ("intro", "They said it couldn't be done. Build something from nothing. Ship it alone. No team, no funding, just code and conviction."),
    ("verse", "Every line of code is a bet against the odds. Every commit, a small rebellion. While they wait for permission, we're already live."),
    ("hook", "This is the sound of building in the dark. No applause, no audience. Just the hum of the machine and the fire in the chest."),
]

def get_smart_path(folder_name):
    full_path = os.path.join(MODELS_DIR, folder_name)
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

    model_path = get_smart_path("Qwen3-TTS-12Hz-1.7B-Base-8bit")
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded.")

    generated_files = []
    for i, (name, text) in enumerate(SEGMENTS):
        print(f"\n--- Segment {i+1}/{len(SEGMENTS)}: {name} ---")
        print(f"Text: {text}")
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{name}")
        os.makedirs(temp_dir, exist_ok=True)

        generate_audio(
            model=model,
            text=text,
            ref_audio=REF_AUDIO,
            ref_text=ref_text,
            instruct=INSTRUCT,
            speed=0.9,
            temperature=0.7,
            output_path=temp_dir,
        )

        # Move output to named file
        src = os.path.join(temp_dir, "audio_000.wav")
        dst = os.path.join(OUTPUT_DIR, f"vocal-{name}.wav")
        if os.path.exists(src):
            shutil.move(src, dst)
            generated_files.append(dst)
            print(f"Saved: {dst}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n=== Done! {len(generated_files)} vocal segments generated ===")
    for f in generated_files:
        print(f"  {f}")

if __name__ == "__main__":
    main()
