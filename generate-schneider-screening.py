#!/usr/bin/env python3
"""Generate Schneider screening interview answers — voice clone practice audio.

Short, warm, conversational answers for a 15-minute AI screening call.
Uses Matthew's cloned voice with Approach C (punchy text + emotion instruct).
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
OUTPUT_DIR = "/Volumes/Virtual Server/projects/schneider-ops/audio/matthew-clone-screening"

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Emotion instruct for conversational delivery
INSTRUCT = "Speaking naturally in a relaxed conversation, like talking to a recruiter you respect. Confident and warm, with natural pauses between thoughts. Not presenting — just talking."

# ── Screening Interview Answers ──
# Short, warm, 30-60 seconds each. Leave them wanting more.

ANSWERS = {
    "q0-greeting": [
        ("q0-part1", "I'm doing great, thanks for asking. Really excited to chat about the Head of Technology role. I've been looking forward to this one."),
    ],
    "q1-legal": [
        ("q1-part1", "Yes, fully authorized, no visa needed. And I can start pretty quickly. I've been building full-time as a founder, so no notice period to deal with."),
    ],
    "q2-why-schneider": [
        ("q2-part1", "Honestly? I grew up around horses. I worked at Dixie Stampede in Branson, Missouri when I was younger. Thirty-two horses, full tack room, nightly shows. So when I saw this posting, it wasn't just another tech role for me. I actually understand the customer."),
        ("q2-part2", "I've spent the last two years building AI automation systems. And I'm looking for somewhere to apply that work to something physical. Real products, real customers, real operations. Schneider's been doing this for seventy-eight years. That's the kind of foundation I want to build on."),
    ],
    "q3-background": [
        ("q3-part1", "Sure. I started as a travel content creator. Seventy countries, built an audience, brand partnerships. But I realized I was building on rented platforms. That pushed me into engineering."),
        ("q3-part2", "I founded Purple Squirrel Media and went deep into AI-native development. Python, TypeScript, Rust. In the last two years I've shipped fifty-seven repos and close to two thousand commits. But what really sets me apart is that I don't just use AI tools. I've built the infrastructure that makes AI development reliable. Custom automation, deployment systems, the whole workflow."),
        ("q3-part3", "Now I'm looking for the right place to apply that. And Schneider feels like a great fit."),
    ],
    "q4-ai-workflow": [
        ("q4-part1", "AI is my primary developer. I'm the architect and the reviewer. I've built a full development workflow around it. Automation, testing, deployment. Almost two thousand commits, all AI-generated, all reviewed, all in production."),
        ("q4-part2", "The thing that gets me excited about Schneider is that I know your ERP, Fulfil, has AI integration built in. I've built similar integrations myself. So there's a real opportunity to deliver value quickly. Not a six-month roadmap, but actual tools the team can use in the first few weeks."),
    ],
    "q5-first-months": [
        ("q5-part1", "First priority would be listening. Understanding the team, the operations, the customers. I'd audit the tech stack. I know you're working with Shopify Plus, Fulfil, BigQuery. And I'd figure out where AI can have the most immediate impact."),
        ("q5-part2", "Then I'd focus on quick wins the team can actually feel. Not dashboards nobody looks at. Real tools that save people time. And I'd make sure the team trusts that the technology is there to help them, not replace them."),
    ],
    "q6-questions": [
        ("q6-part1", "Yeah, I do. What does the current tech team look like? Would I be building from scratch or inheriting a team?"),
    ],
    "q7-closing": [
        ("q7-part1", "Thank you. I really enjoyed this. I'm genuinely excited about what Schneider is building. Looking forward to hearing about next steps."),
    ],
}


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

    # Load reference transcript
    with open(REF_TEXT_FILE, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()

    # Load Base model (supports voice cloning)
    model_folder = "Qwen3-TTS-12Hz-1.7B-Base-8bit"
    model_path = get_smart_path(model_folder)
    if not model_path:
        print(f"Error: Model not found at {os.path.join(MODELS_DIR, model_folder)}")
        sys.exit(1)

    print(f"Loading Base model from {model_path}...")
    model = load_model(model_path)
    print(f"Model loaded. Using voice reference: {REF_AUDIO}")
    print(f"Instruct: {INSTRUCT}")
    print(f"Reference transcript: {ref_text[:80]}...\n")

    total_segments = sum(len(parts) for parts in ANSWERS.values())
    current = 0

    for question_key, segments in ANSWERS.items():
        question_dir = os.path.join(OUTPUT_DIR, question_key)
        os.makedirs(question_dir, exist_ok=True)

        for seg_name, text in segments:
            current += 1
            print(f"[{current}/{total_segments}] Generating {question_key}/{seg_name}...")

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
                dest = os.path.join(question_dir, f"{seg_name}.wav")
                if os.path.exists(source):
                    shutil.move(source, dest)
                    print(f"  -> Saved {dest}")
                else:
                    print(f"  !! No audio generated for {seg_name}")

            except Exception as e:
                print(f"  !! Error: {e}")
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

            gc.collect()

    # Concatenate per question
    print("\nConcatenating segments per question...")
    for question_key in ANSWERS:
        question_dir = os.path.join(OUTPUT_DIR, question_key)
        wavs = sorted([
            os.path.join(question_dir, f)
            for f in os.listdir(question_dir)
            if f.endswith(".wav")
        ])
        if not wavs:
            continue

        concat_file = os.path.join(question_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for w in wavs:
                f.write(f"file '{w}'\n")

        output_path = os.path.join(OUTPUT_DIR, f"schneider-{question_key}.m4a")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:a", "aac", "-b:a", "192k", output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  -> {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  !! Concat failed for {question_key}: {e}")

    # Full practice file
    print("\nBuilding full practice file...")
    all_m4as = sorted([
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("schneider-q") and f.endswith(".m4a")
    ])
    if all_m4as:
        full_concat = os.path.join(OUTPUT_DIR, "full-concat.txt")
        with open(full_concat, "w") as f:
            for m in all_m4as:
                f.write(f"file '{m}'\n")

        full_output = os.path.join(OUTPUT_DIR, "schneider-full-practice-v3.m4a")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", full_concat,
            "-c:a", "aac", "-b:a", "192k", full_output
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"\n  DONE: {full_output}")
        except subprocess.CalledProcessError as e:
            print(f"  !! Full concat failed: {e}")

    # Copy to Desktop
    desktop_copy = os.path.expanduser("~/Desktop/schneider-full-practice-v3.m4a")
    src = os.path.join(OUTPUT_DIR, "schneider-full-practice-v3.m4a")
    if os.path.exists(src):
        shutil.copy2(src, desktop_copy)
        print(f"  Copied to {desktop_copy}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
