#!/usr/bin/env python3
"""Generate full spoken word track with verses, ad-libs, and hook using Matthew's voice clone."""
import os
import shutil
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
OUTPUT_DIR = os.path.expanduser("~/Desktop/AI-Music/vocals-v2")

REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

MAIN_INSTRUCT = "Dark, intense spoken word delivery. Low and deliberate, like narrating a film trailer. Confident, slightly menacing undertone. Not rushed — let each word land."
ADLIB_INSTRUCT = "Short, punchy, aggressive. Like a hype man. Quick and sharp, almost a bark."
HOOK_INSTRUCT = "Intense and rhythmic. Almost chanting. Hypnotic repetition with building energy."

SEGMENTS = [
    # Verse 1 (intro)
    ("verse1", MAIN_INSTRUCT, "They said it couldn't be done. Build something from nothing. Ship it alone. No team, no funding, just code and conviction."),
    # Verse 2
    ("verse2", MAIN_INSTRUCT, "Every line of code is a bet against the odds. Every commit, a small rebellion. While they wait for permission, we're already live."),
    # Verse 3 (new)
    ("verse3", MAIN_INSTRUCT, "Midnight deployments and broken pipelines. Debugging dreams at three AM. They'll never understand the weight of building something real."),
    # Verse 4 (new)
    ("verse4", MAIN_INSTRUCT, "From zero to shipped. From silence to signal. The blockchain doesn't sleep and neither do we. This is how empires start. In the dark. Alone."),
    # Ad-libs
    ("adlib1", ADLIB_INSTRUCT, "Let's go."),
    ("adlib2", ADLIB_INSTRUCT, "Yeah."),
    ("adlib3", ADLIB_INSTRUCT, "Ship it."),
    ("adlib4", ADLIB_INSTRUCT, "No sleep."),
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
    for i, (name, instruct, text) in enumerate(SEGMENTS):
        print(f"\n--- Segment {i+1}/{len(SEGMENTS)}: {name} ---")
        print(f"Text: {text}")
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{name}")
        os.makedirs(temp_dir, exist_ok=True)

        generate_audio(
            model=model,
            text=text,
            ref_audio=REF_AUDIO,
            ref_text=ref_text,
            instruct=instruct,
            speed=0.9,
            temperature=0.7,
            output_path=temp_dir,
        )

        src = os.path.join(temp_dir, "audio_000.wav")
        dst = os.path.join(OUTPUT_DIR, f"{name}.wav")
        if os.path.exists(src):
            shutil.move(src, dst)
            generated_files.append(dst)
            print(f"Saved: {dst}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n=== Done! {len(generated_files)} segments generated ===")
    for f in generated_files:
        print(f"  {f}")

if __name__ == "__main__":
    main()
