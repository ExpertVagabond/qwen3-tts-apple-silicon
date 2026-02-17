#!/usr/bin/env python3
"""Batch generate hackathon video narrations using Qwen3-TTS MLX."""

import os
import sys
import shutil
import time
import gc
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODELS_DIR = os.path.join(os.getcwd(), "models")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs", "Hackathon")
SPEAKER = "Ryan"  # Male English narrator
SPEED = 0.9  # Slightly slower for clarity

# ── Tribeca DAO narration segments ──────────────────────────────────

TRIBECA_SEGMENTS = [
    ("01-opening", "This is Tribeca DAO, an open-standard governance toolkit for Solana, originally built by the TribecaHQ team with 210 stars. It was abandoned in early 2022 when its dependencies on yanked crates and shutdown projects made it completely unbuildable."),

    ("02-problem", "Tribeca depended on Anchor 0.22, which was yanked from crates dot I O. It also relied on two abandoned ecosystems. Saber's vipers library for validation macros, and Goki's smart-wallet for multisig execution. No Rust version after 1.60 could compile it. The TypeScript SDK depended entirely on Saber HQ packages that no longer exist."),

    ("03-revival", "We completed 9 migration stories. First, we replaced all 75 vipers macro calls across 3 programs with Anchor 0.30 native equivalents. Then we created a smart-wallet stub crate matching Goki's account layout so the full proposal lifecycle, create, vote, queue, execute, still works. We upgraded Anchor from 0.22 to 0.30.1, pinned BPF toolchain dependencies, and regenerated all 4 IDLs."),

    ("04-sdk", "The TypeScript SDK was migrated from Saber HQ to Coral XYZ Anchor. All type exports are backward-compatible. We have 9 smoke tests verifying the program types, PDA derivations, and SDK construction."),

    ("05-demo", "Here's the full governance lifecycle. Creating a governor, submitting a proposal, casting votes, queueing, and executing. This validates the entire Tribeca governance flow works end-to-end on the revived code."),

    ("06-close", "Tribeca is alive again. Four programs build, tests pass, SDK works, and the full governance lifecycle is demonstrated. Check the repo at github dot com slash ExpertVagabond slash tribeca-dao."),
]

# ── Grape Art narration segments ────────────────────────────────────

GRAPE_SEGMENTS = [
    ("01-opening", "This is Grape Art, a full-featured NFT marketplace on Solana with Metaplex Auction House integration. It was abandoned in late 2022 with over 200 broken dependencies."),

    ("02-problem", "Grape Art had deep dependencies on 9 dead protocols. CyberConnect for social graphs, Shadow Drive for storage, Strata for bonding curves, Dialect for messaging, and more. It used project-serum anchor, Metaplex SDK v1, and wallet-adapter versions that are all deprecated. npm install failed with 200 plus conflicts."),

    ("03-dead-code", "We took an aggressive approach. Rather than stubbing 9 dead integrations, we removed them entirely. CyberConnect, Shadow Drive, Strata, Streamflow, Dialect, Grape Network, Crossmint, Fida Name Service, and TipLink. This cut the dependency surface by about 40 percent."),

    ("04-migration", "We migrated every major package. Anchor from project-serum to coral-xyz, SPL Token from v0.1 to v0.3, Metaplex from v1 to v3 plus, wallet-adapter to latest. The 400 plus TypeScript source files compile with zero errors. Parcel was upgraded from 2.8 to 2.16.4. We wrote shims for dead packages like mercurial-finance optimist and added 50 plus alias entries for packages using the exports field."),

    ("05-demo", "The 746-line marketplace demo validates all 19 auction house operations. Listing, buying, offers, cancellations, plus the DAO governance integration. Every function is validated with proper TypeScript types."),

    ("06-close", "Grape Art is alive again. The full NFT marketplace builds, all 400 plus files compile, and the auction house operations are validated. Check it out at github dot com slash ExpertVagabond slash grape-art."),
]

# ── Port Finance narration segments ─────────────────────────────────

PORT_SEGMENTS = [
    ("01-opening", "This is Port Finance, a variable-rate lending protocol on Solana. Deposit, borrow, and liquidate. It was abandoned in mid-2022 on Solana SDK 1.9."),

    ("02-problem", "Port Finance was stuck on Solana SDK 1.9 with Rust 2018 edition. It depended on Switchboard v1 for price oracles, which has been completely deprecated. The token-lending program and staking program couldn't compile on any modern Rust toolchain. The 66 integration tests were all broken."),

    ("03-migration", "We upgraded to Solana SDK 1.18 plus and Rust 2021 edition. SPL Token was updated to the 2022-compatible crate. The Switchboard v1 oracle was replaced with a deprecation stub that preserves the interface. A production deployment would swap in Pyth or Switchboard v2. Both programs BPF-build successfully."),

    ("04-tests", "All 66 tests pass. Borrow, deposit, flash loan, init lending market, init obligation, init reserve, liquidate, redeem, refresh, repay, withdraw. Every core lending operation is verified."),

    ("05-sdk", "We built a new TypeScript client SDK that didn't exist before. State decoders for reserves and obligations, instruction builders for all lending operations, and a demo script with 52 assertions validating the complete API surface."),

    ("06-close", "Port Finance lending is alive again. Both programs build, 66 tests pass, and a new TypeScript SDK makes it accessible to developers. Check it out at github dot com slash ExpertVagabond slash port-lending."),
]

# ── Parrot TWAP narration segments ─────────────────────────────────

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


def generate_segments(model, project_name, segments):
    project_dir = os.path.join(OUTPUT_DIR, project_name)
    os.makedirs(project_dir, exist_ok=True)

    for seg_name, text in segments:
        output_path = f"temp_{project_name}_{seg_name}"
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

            # Move generated file
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


def main():
    print("=" * 50)
    print(" Hackathon Video Narration Generator")
    print(" Model: Qwen3-TTS 1.7B CustomVoice (MLX)")
    print(f" Speaker: {SPEAKER} | Speed: {SPEED}x")
    print("=" * 50)

    model_path = get_model_path("Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
    if not model_path:
        print("ERROR: CustomVoice model not found in models/")
        print("Download: huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        sys.exit(1)

    print("\nLoading model...")
    model = load_model(model_path)
    print("Model loaded.\n")

    projects = [
        ("tribeca-dao", TRIBECA_SEGMENTS),
        ("grape-art", GRAPE_SEGMENTS),
        ("port-lending", PORT_SEGMENTS),
        ("parrot-twap", PARROT_SEGMENTS),
    ]

    for project_name, segments in projects:
        print(f"\n{'='*50}")
        print(f" Generating: {project_name} ({len(segments)} segments)")
        print(f"{'='*50}")
        generate_segments(model, project_name, segments)
        gc.collect()

    print(f"\n{'='*50}")
    print(f" All done! Audio files at: {OUTPUT_DIR}")
    print(f"{'='*50}")

    # List generated files
    for project_name, _ in projects:
        project_dir = os.path.join(OUTPUT_DIR, project_name)
        if os.path.exists(project_dir):
            files = sorted(os.listdir(project_dir))
            print(f"\n{project_name}/")
            for f in files:
                size = os.path.getsize(os.path.join(project_dir, f))
                print(f"  {f} ({size // 1024}KB)")


if __name__ == "__main__":
    main()
