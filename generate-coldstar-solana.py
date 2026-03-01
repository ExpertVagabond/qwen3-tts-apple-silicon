#!/usr/bin/env python3
"""Generate Coldstar Solana demo voiceover using Matthew's cloned voice.

~2 minute narration for Solana demo video.
Uses Approach C: punchy text + emotion instruct for natural delivery.
No FairScore references.
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
OUTPUT_DIR = "/Volumes/Virtual Server/projects/coldstar/demo-audio-solana"

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Emotion instruct
INSTRUCT = "Speaking with quiet confidence and technical authority, like a founder demoing their product to smart investors. Clear, measured, slightly intense. Not salesy. Genuine belief in what they built."

# Voiceover segments — ~2 minutes total, no FairScore
SEGMENTS = [
    ("s01-intro",
     "This is Coldstar. An air-gapped cold wallet for Solana."),

    ("s02-problem",
     "Here's the problem with how people store crypto today. Browser wallets? "
     "Your private key sits in browser memory. One bad extension. One phishing link. Gone. "
     "Hardware wallets? Better. But they still connect over USB or Bluetooth. "
     "Ledger's firmware is closed source. You're trusting a company. Not verifying security."),

    ("s03-airgap",
     "Coldstar takes a different approach. True air gap. "
     "Your private key lives on a device that never touches the internet. "
     "Not over USB. Not Bluetooth. Not WiFi. Ever. "
     "The only thing that crosses the gap? QR codes. Optical. Unidirectional. "
     "You can see exactly what's being transferred."),

    ("s04-how-it-works",
     "Here's how it works. Step one. Build your transaction on an online device. "
     "Amount. Recipient. All fetched from Solana's RPC. "
     "Step two. That unsigned transaction becomes a QR code. Scan it on your offline device. "
     "Step three. The offline device signs it with the Rust secure signer. "
     "Keys exist only in locked memory. Automatically wiped after signing. "
     "Step four. The signed transaction comes back as a QR code. "
     "Scan it on the online device. Broadcast to Solana. Done."),

    ("s05-rust-signer",
     "The signing core is written in Rust. Not Python. Not JavaScript. Rust. "
     "Memory-locked buffers. Argon2id key derivation with sixty-four megabytes of memory cost. "
     "AES 256 GCM encryption. Automatic zeroization. "
     "The private key never exists in readable form outside of a locked memory page. "
     "And when signing is done? That page is wiped. Not freed. Wiped."),

    ("s06-solana-native",
     "We built this specifically for Solana. Native Ed25519 signing. "
     "Full SPL token support. Staking operations. Jupiter swap integration. "
     "This isn't a generic wallet. It's built for Solana from the ground up."),

    ("s07-mcp-server",
     "Coldstar includes an MCP server with nine tools for Solana. "
     "Wallet balances. Real-time token prices via Pyth. Swap quotes through Jupiter. "
     "Full portfolio tracking with USD values. Fee estimation. Transaction validation. "
     "All read-only. No private keys over the wire."),

    ("s08-agent-ready",
     "Drop-in for Claude. ChatGPT. Any MCP-compatible AI agent. "
     "Designed for treasury agents managing cold plus hot wallet architecture. "
     "The agent reads. The human signs. That's the security model."),

    ("s09-multichain",
     "Coldstar started on Solana with Ed25519. "
     "Now we've added Base L2 with secp256k1. "
     "The encryption layer is shared. Argon2id plus AES 256 GCM works the same "
     "regardless of which curve you're signing with. "
     "One wallet architecture. Two chains. Same air gap."),

    ("s10-open-source",
     "Everything is open source. The Rust signer. The Python CLI. The QR protocol. "
     "Eighteen thousand lines of auditable code. "
     "No proprietary firmware. No closed-source enclaves. "
     "You don't trust us. You verify."),

    ("s11-cta",
     "Coldstar. Cold signing for Solana. Check it out at coldstar dot dev."),
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
    print(f"Model loaded. Using voice reference: {REF_AUDIO}")
    print(f"Instruct: {INSTRUCT}")
    print(f"Reference transcript: {ref_text[:80]}...\n")

    total = len(SEGMENTS)

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
            dest = os.path.join(OUTPUT_DIR, f"{seg_name}.wav")
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

    # Generate silence gap
    silence_path = os.path.join(OUTPUT_DIR, "silence.wav")
    print("\nGenerating silence gap...")
    generate_silence(0.8, silence_path)

    # Concatenate all segments
    print("Concatenating all segments...")
    concat_file = os.path.join(OUTPUT_DIR, "concat.txt")
    wavs = sorted([
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("s") and f.endswith(".wav") and not f.startswith("silence")
    ])

    with open(concat_file, "w") as f:
        for j, w in enumerate(wavs):
            f.write(f"file '{w}'\n")
            if j < len(wavs) - 1:
                f.write(f"file '{silence_path}'\n")

    output_wav = os.path.join(OUTPUT_DIR, "coldstar-solana-voiceover.wav")
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

    output_m4a = os.path.join(OUTPUT_DIR, "coldstar-solana-voiceover.m4a")
    cmd = [
        "ffmpeg", "-y", "-i", output_wav,
        "-c:a", "aac", "-b:a", "192k", output_m4a
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  -> M4A: {output_m4a}")
    except subprocess.CalledProcessError as e:
        print(f"  !! M4A conversion failed: {e}")

    desktop_copy = os.path.expanduser("~/Desktop/coldstar-solana-voiceover.m4a")
    if os.path.exists(output_m4a):
        shutil.copy2(output_m4a, desktop_copy)
        print(f"  -> Desktop: {desktop_copy}")

    print("\nVoiceover generation complete!")
    print(f"Total segments: {total}")

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", output_wav],
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
