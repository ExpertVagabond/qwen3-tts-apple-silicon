#!/usr/bin/env python3
"""Generate Coldstar hackathon demo narration using Matthew's cloned voice.

Uses Qwen3-TTS Base model with Approach C: punchy text + emotion instruct.
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
OUTPUT_DIR = "/Volumes/Virtual Server/projects/coldstar/video/demo-2026/tts-matthew-v2"

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Natural, confident delivery
INSTRUCT = "Speaking with confident authority about a product you built. Clear, direct, like a pitch to investors. Natural pacing with emphasis on key selling points."

# Tighter script — target 2:45 total narration
SEGMENTS = [
    ("s01-title", "This is Coldstar. An open-source, air-gapped cold wallet for Solana. Eighteen thousand lines of code. Zero hardware cost."),

    ("s02-hardware-wallets", "So what is a hardware wallet? It's a device that keeps your private keys completely offline. Your private key controls your crypto. Software wallets like Phantom? Your keys live right in the browser. One exploit and you're drained. Hardware wallets fix this by signing transactions on a separate, offline device."),

    ("s03-problem", "But here's the catch. A Ledger Nano S costs seventy nine dollars. A Ledger Stax? Almost four hundred. Closed-source firmware. Proprietary chips. And remember Ledger's data breach? Two hundred seventy thousand customer addresses leaked. Phishing attacks followed."),

    ("s04-comparison", "Coldstar takes a different approach. Instead of buying proprietary hardware? You use any USB drive you already own. The software is one hundred percent open source. Fully auditable. No vendor lock-in. No supply chain risk. And the cost? Free."),

    ("s05-install", "Getting started takes three steps. Go to coldstar dot dev slash build. Clone the repo. Run the install script. Rust compiles the secure signer in about two minutes. Then plug in any USB drive and you're ready."),

    ("s06-terminal", "The install process is dead simple. Git clone. Run install dot sh. It checks your Python version, installs dependencies, builds the Rust secure signer with AES two fifty six encryption, and verifies everything. Done."),

    ("s07-tui", "Here's the terminal interface. Clean menu system. USB drives auto-detected. You can flash a cold wallet, mount and unlock it, sign transactions via QR code, or manage your vault dashboard. Everything runs offline."),

    ("s08-signing", "The air-gapped signing flow works in five steps. Build the transaction on your online machine. Encode it as a QR code. Scan it with Coldstar on the offline USB. Coldstar signs it in locked RAM using mlock and zeroize. Then scan the signed QR back and broadcast from your online device. Zero network exposure."),

    ("s09-vault", "The vault dashboard gives you full portfolio management right in the terminal. Token balances. Price data. Transaction history. And a send panel with priority fee control and QR signing built in."),

    ("s10-monetization", "Revenue model. One percent transaction fee. Capped at ten dollars max. Send five hundred bucks? Five dollar fee. Send ten thousand? Still just ten dollars. The cap protects whales and everyday users alike. On top of that? Jito staking tips on every transaction for passive validator revenue."),

    ("s11-revenue", "Unit economics at just one thousand active users. Five transactions per week at an average fee of three fifty. That's seventy thousand monthly. Plus another five thousand from Jito tips. Seventy five thousand MRR. And the user's total cost? Less than any hardware wallet on the market."),

    ("s12-cta", "Coldstar. Cold signing for everyone. No hardware required. Try it at coldstar dot dev. The code is open source on GitHub. Built by ChainLabs Technologies."),
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
    print(f"Reference transcript: {ref_text[:80]}...\n")

    total = len(SEGMENTS)
    for i, (seg_name, text) in enumerate(SEGMENTS):
        print(f"[{i+1}/{total}] Generating {seg_name}...")
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
                # Get duration
                r = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", dest],
                    capture_output=True, text=True
                )
                dur = float(r.stdout.strip()) if r.stdout.strip() else 0
                print(f"  -> {dest} ({dur:.1f}s audio, {elapsed:.0f}s gen time)")
            else:
                print(f"  !! No audio generated for {seg_name}")

        except Exception as e:
            print(f"  !! Error: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        gc.collect()

    # Concatenate all segments
    print("\nConcatenating all segments...")
    concat_list = os.path.join(OUTPUT_DIR, "concat.txt")
    with open(concat_list, 'w') as f:
        for seg_name, _ in SEGMENTS:
            wav_path = os.path.join(OUTPUT_DIR, f"{seg_name}.wav")
            if os.path.exists(wav_path):
                f.write(f"file '{wav_path}'\n")

    full_output = os.path.join(OUTPUT_DIR, "coldstar-narration-full.wav")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c:a", "pcm_s16le", full_output
    ], capture_output=True)

    if os.path.exists(full_output):
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", full_output],
            capture_output=True, text=True
        )
        dur = float(r.stdout.strip())
        print(f"\nFull narration: {full_output}")
        print(f"Total duration: {dur:.1f}s ({dur/60:.1f} min)")

    print("\nDone!")


if __name__ == "__main__":
    main()
