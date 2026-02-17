#!/usr/bin/env python3
"""Generate Parrot TWAP narration only (standalone)."""

import os
import sys
import shutil
import time
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODELS_DIR = os.path.join(os.getcwd(), "models")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs", "Hackathon")
SPEAKER = "Ryan"
SPEED = 0.9

PARROT_SEGMENTS = [
    ("01-title", "This is Parrot TWAP, a time-weighted average price bot for Solana that we brought back from the dead for the Graveyard Hackathon. Originally built by Parrot Protocol back in 2021. Written in Go. About 1,300 lines of code."),

    ("02-problem", "Here's why it died. The entire swap backend was built on Serum DEX and Raydium AMM v4. When FTX collapsed in November 2022, Serum was sunset immediately. The order book just disappeared. Raydium v4 was deprecated shortly after. And Parrot Protocol itself shut down. So every single dependency this bot relied on was dead. It's been sitting on GitHub collecting dust for three years."),

    ("03-solution", "Our revival. We ripped out the dead Raydium and Serum swap backend and replaced it with Jupiter, the dominant DEX aggregator on Solana. Jupiter routes through every DEX automatically. Raydium's new CLMM, Orca Whirlpools, Meteora, 30 plus venues. One API call gets you the optimal route. Now the bot supports any token pair, not just PRT. And we preserved the original architecture. The scheduler, balance tracking, stop conditions, auto-transfers. Just replaced the swap engine."),

    ("04-code-diff", "Here's the actual code change. Left side, the old Raydium swap. Hard-coded pool configs, Serum market references, only supported PRT-to-USDC. Right side, our new Jupiter integration. Get a quote via HTTP, receive a serialized transaction, decode it from base64, sign with your wallet, and send. Clean, universal, works with any token. Two new files. jupiter dot go for the swap engine, and jupiter dot go in config for token symbol resolution."),

    ("05-live-demo", "This is hitting the real Jupiter API right now on Solana mainnet. 0.1 SOL gets you about 8.50 USDC, routed through whichever DEX has the best price at this moment. Zero price impact at this size. The TWAP bot uses this exact same API. Get a quote, build a swap, send it on-chain."),

    ("06-dca-economics", "Why build a DCA bot? Because timing the market is a losing game. DCA spreads your entry across dozens of swaps at different price points. Here's the fee breakdown. Each swap costs about 95,000 lamports, less than a penny. At 0.1 SOL per swap, that's under 0.1 percent overhead. At 1 SOL, it's basically free. And it's fully self-custodied. Your keys, your wallet. No third-party platform risk."),

    ("07-cli-usage", "The CLI is straightforward. Pick your token pair using symbols. SOL, USDC, BONK, JUP, whatever. Set the amount per swap, choose your interval. You can add stop conditions to halt when you've accumulated enough. Auto-transfer to a cold wallet when the balance exceeds a threshold. Or just use raw mint addresses for any token Jupiter supports."),

    ("08-architecture", "Clean Go module structure. CLI parses arguments, scheduler fires on your interval, balance check runs against the RPC, Jupiter gets the best quote, transaction gets signed and sent, everything gets logged. Ten Go files, two new ones. The rest is preserved from the original Parrot codebase."),

    ("09-closing", "And here's the proof. We executed a real swap on Solana mainnet. 0.01 SOL swapped for 0.85 USDC. Transaction verified on Solscan, slot 400 million. Dead protocol, dead DEX, dead infrastructure. Alive again. Check the repo, try the live demo. Thanks for watching."),
]


def get_model_path(folder_name):
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
    print("=" * 50)
    print(" Parrot TWAP Narration Generator")
    print(f" Speaker: {SPEAKER} | Speed: {SPEED}x")
    print(f" Segments: {len(PARROT_SEGMENTS)}")
    print("=" * 50)

    model_path = get_model_path("Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
    if not model_path:
        print("ERROR: CustomVoice model not found in models/")
        sys.exit(1)

    print("\nLoading model...")
    model = load_model(model_path)
    print("Model loaded.\n")

    project_dir = os.path.join(OUTPUT_DIR, "parrot-twap")
    os.makedirs(project_dir, exist_ok=True)

    for seg_name, text in PARROT_SEGMENTS:
        output_path = f"temp_parrot_{seg_name}"
        print(f"\n  [{seg_name}] Generating ({len(text)} chars)...")
        start = time.time()

        try:
            generate_audio(
                model=model,
                text=text,
                voice=SPEAKER,
                instruct="Professional narrator, clear and confident, moderate pace",
                speed=SPEED,
                output_path=output_path,
            )

            source = os.path.join(output_path, "audio_000.wav")
            dest = os.path.join(project_dir, f"{seg_name}.wav")
            if os.path.exists(source):
                shutil.move(source, dest)
                elapsed = time.time() - start
                print(f"  [{seg_name}] Done in {elapsed:.1f}s -> {dest}")
            else:
                print(f"  [{seg_name}] WARNING: No audio file generated")

        except Exception as e:
            print(f"  [{seg_name}] ERROR: {e}")

        finally:
            if os.path.exists(output_path):
                shutil.rmtree(output_path, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f" Done! Files at: {project_dir}")
    print(f"{'='*50}")

    files = sorted(os.listdir(project_dir))
    for f in files:
        size = os.path.getsize(os.path.join(project_dir, f))
        print(f"  {f} ({size // 1024}KB)")


if __name__ == "__main__":
    main()
