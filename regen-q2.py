#!/usr/bin/env python3
"""Regenerate only Q2 (why-schneider) with Dixie Stampede opening."""

import os, sys, shutil, time, gc, subprocess, warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODELS_DIR = os.path.join(os.getcwd(), "models")
VOICES_DIR = os.path.join(os.getcwd(), "voices")
OUTPUT_DIR = "/Volumes/Virtual Server/projects/schneider-ops/audio/matthew-clone-v3"

REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

INSTRUCT = "Speaking naturally in a relaxed conversation, like talking to a recruiter you respect. Confident and warm, with natural pauses between thoughts. Not presenting — just talking."

Q2_SEGMENTS = [
    ("q2-part1", "A couple things, honestly. First. I actually grew up around horses. I worked at Dixie Stampede in Branson, Missouri when I was younger. Thirty-two horses. Full tack room. Nightly shows. So when I saw this role, it wasn't just another tech posting. I know what a well-fitted saddle means. I know how riders think about their gear."),
    ("q2-part2", "I've spent the last two years building AI automation systems. MCP servers. Development infrastructure. Fifty-seven repos. Almost two thousand commits. But all of it has been digital infrastructure talking to other digital infrastructure."),
    ("q2-part3", "What drew me to Schneider is that the work lands somewhere real. Four thousand saddles. A thousand pieces of gear. Customers who genuinely love their horses. Eric has three hundred fifty YouTube videos because he truly cares about this. That's not a CEO chasing a tech trend. That's a founder building something he believes in."),
    ("q2-part4", "I want my AI work to express itself in something physical. Faster fulfillment. Smarter inventory. A customer experience that matches the passion your buyers already have. Most Head of AI postings are at startups burning cash. Schneider is a profitable business with real operations. That's the foundation I want to build on."),
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
    with open(REF_TEXT_FILE, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()

    model_path = get_smart_path("Qwen3-TTS-12Hz-1.7B-Base-8bit")
    if not model_path:
        print("Error: Model not found")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded.\n")

    question_dir = os.path.join(OUTPUT_DIR, "q2-why-schneider")
    os.makedirs(question_dir, exist_ok=True)

    for i, (seg_name, text) in enumerate(Q2_SEGMENTS, 1):
        print(f"[{i}/{len(Q2_SEGMENTS)}] Generating {seg_name}...")
        temp_dir = f"temp_{int(time.time())}_{seg_name}"
        try:
            generate_audio(
                model=model, text=text, ref_audio=REF_AUDIO,
                ref_text=ref_text, instruct=INSTRUCT, output_path=temp_dir,
            )
            source = os.path.join(temp_dir, "audio_000.wav")
            dest = os.path.join(question_dir, f"{seg_name}.wav")
            if os.path.exists(source):
                shutil.move(source, dest)
                print(f"  -> Saved {dest}")
            else:
                print(f"  !! No audio for {seg_name}")
        except Exception as e:
            print(f"  !! Error: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

    # Re-concatenate Q2
    print("\nConcatenating Q2...")
    wavs = sorted([os.path.join(question_dir, f) for f in os.listdir(question_dir) if f.endswith(".wav")])
    if wavs:
        concat_file = os.path.join(question_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for w in wavs:
                f.write(f"file '{w}'\n")
        q2_out = os.path.join(OUTPUT_DIR, "schneider-q2-why-schneider.m4a")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                        "-c:a", "aac", "-b:a", "192k", q2_out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> {q2_out}")

    # Rebuild full practice file
    print("\nRebuilding full practice file...")
    all_m4as = sorted([os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
                       if f.startswith("schneider-q") and f.endswith(".m4a")])
    if all_m4as:
        full_concat = os.path.join(OUTPUT_DIR, "full-concat.txt")
        with open(full_concat, "w") as f:
            for m in all_m4as:
                f.write(f"file '{m}'\n")
        full_output = os.path.join(OUTPUT_DIR, "schneider-full-practice-v3.m4a")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", full_concat,
                        "-c:a", "aac", "-b:a", "192k", full_output],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> {full_output}")

    # Copy to Desktop
    src = os.path.join(OUTPUT_DIR, "schneider-full-practice-v3.m4a")
    if os.path.exists(src):
        shutil.copy2(src, os.path.expanduser("~/Desktop/schneider-full-practice-v3.m4a"))
        print("  Copied to Desktop")

    print("\nDone! Q2 regenerated with Dixie Stampede.")


if __name__ == "__main__":
    main()
