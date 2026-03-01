#!/usr/bin/env python3
"""Generate Schneider deep interview answers v3 — Conversational format for live Ribbon AI interview.

Full 60-90 sec answers for technical/hiring manager round.
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
OUTPUT_DIR = "/Volumes/Virtual Server/projects/schneider-ops/audio/matthew-clone-v3"

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Emotion instruct for conversational delivery
INSTRUCT = "Speaking naturally in a relaxed conversation, like talking to a recruiter you respect. Confident and warm, with natural pauses between thoughts. Not presenting — just talking."

# ── Deep Interview Answers ──
# 60-90 second answers with full detail for technical/hiring manager round

ANSWERS = {
    "q0-greeting": [
        ("q0-part1", "I'm doing great, thanks for asking. Really excited to chat about the Head of Technology role. I've been looking forward to this one."),
    ],
    "q1-legal": [
        ("q1-part1", "Yes, full authorization to work in the US. No visa needed. And I can start quickly. I've been building full-time as a founder, so there's no notice period. Pretty flexible on timing."),
    ],
    "q2-why-schneider": [
        ("q2-part1", "A couple things, honestly. First. I actually grew up around horses. I worked at Dixie Stampede in Branson, Missouri when I was younger. Thirty-two horses. Full tack room. Nightly shows. So when I saw this role, it wasn't just another tech posting. I know what a well-fitted saddle means. I know how riders think about their gear."),
        ("q2-part2", "I've spent the last two years building AI automation systems. MCP servers. Development infrastructure. Fifty-seven repos. Almost two thousand commits. But all of it has been digital infrastructure talking to other digital infrastructure."),
        ("q2-part3", "What drew me to Schneider is that the work lands somewhere real. Four thousand saddles. A thousand pieces of gear. Customers who genuinely love their horses. Eric has three hundred fifty YouTube videos because he truly cares about this. That's not a CEO chasing a tech trend. That's a founder building something he believes in."),
        ("q2-part4", "I want my AI work to express itself in something physical. Faster fulfillment. Smarter inventory. A customer experience that matches the passion your buyers already have. Most Head of AI postings are at startups burning cash. Schneider is a profitable business with real operations. That's the foundation I want to build on."),
    ],
    "q3-background": [
        ("q3-part1", "Sure thing. I started as a travel content creator. Seventy countries. Audience of twenty-nine thousand. Brand partnerships with major outdoor companies. But I realized I was building on rented platforms. Someone else owned the algorithm. The data. The distribution."),
        ("q3-part2", "That pushed me into engineering. Founded Purple Squirrel Media. Went deep. Python. TypeScript. Rust. Go. Ruby. In the last two years I've shipped fifty-seven repos and close to two thousand commits. Twelve production sites."),
        ("q3-part3", "What separates me is the AI-native approach. Over sixty-eight hundred Claude Code sessions. I've built an entire autonomous development infrastructure. Custom hooks. Skills. Behavioral autopilot. Multi-agent orchestration. I don't just use AI tools. I've built the systems that make AI development reliable and repeatable."),
    ],
    "q4-ai-workflow": [
        ("q4-part1", "AI is my primary developer. I'm the architect and reviewer. I run Claude Code with five custom hooks. Pre-bash guards. Auto-context injection. Auto-diagnosis for build errors. Credential blocking before every commit. Session handoff across sessions."),
        ("q4-part2", "Seven reusable skills. One-command deploys. Build failure diagnosis. Parallel codebase exploration. And I run parallel sessions on the same codebase. Multiple agents. Same repo. Git-aware conflict prevention."),
        ("q4-part3", "The result? Almost two thousand AI-generated commits. All reviewed. All in production. Sixteen CLI tools. Five MCP servers. Twelve production sites."),
        ("q4-part4", "For Schneider specifically. I know Fulfil dot I O has native Claude MCP integration. I've built five MCP servers myself. Day one, I could wire up natural language queries against your BigQuery warehouse. What blanket SKUs are under thirty days of inventory? Without anyone logging into Fulfil or writing SQL. First week deliverable. Not a roadmap item."),
    ],
    "q4b-mcp-followup": [
        ("q4b-part1", "Absolutely. I actually built three MCP servers specifically for this application. A Fulfil integration with twenty-nine tools covering inventory, orders, shipments, purchasing. A Shopify integration with thirty-six tools. And a Schneider operations bridge that connects both systems."),
        ("q4b-part2", "Things like inventory reconciliation. Order pipeline monitoring. A unified daily dashboard that pulls from both Shopify and Fulfil simultaneously. Seventy-nine tools total. Built in an afternoon. They're live right now."),
    ],
    "q5-cro": [
        ("q5-part1", "I'll be honest. My background isn't traditional CRO. But I've built infrastructure that makes real conversion possible. Especially in a deep catalog business."),
        ("q5-part2", "Think about what kills conversion in a four thousand SKU equestrian catalog. Customers can't find the right saddle. Questions at ten PM with nobody there. Inventory mismatch. Slow site on mobile at a horse show."),
        ("q5-part3", "I audited your current site. Two hundred twelve kilobyte HTML pages. Four to seven times a modern Shopify storefront. Ten-second crawl delay. The Shopify Plus migration alone is a massive conversion lever."),
        ("q5-part4", "You've got the right tools. Kustomer. Bloomreach. Noibu. What's missing is the AI layer connecting them. An AI agent trained on your two hundred thirty-three YouTube videos and three thousand product descriptions could handle sixty to seventy percent of routine inquiries. And actually know the difference between VTEK Wither Relief and a standard Cutback fit."),
    ],
    "q6-ninety-day": [
        ("q6-part1", "First thing I'd do? Shut up and listen. For real. First two weeks. Audit the tech. Shopify Plus migration status. Fulfil integration. BigQuery. AWS. The legacy ASP platform too. Map every integration."),
        ("q6-part2", "But more importantly. Spend time with the people. Customer service fielding rider questions. Warehouse team packing orders. Marketing creating content. Technology serves them. Not the other way around."),
        ("q6-part3", "Days fifteen to forty-five. Quick wins people can feel. Wire up Fulfil's MCP integration for natural language ops queries. Train an AI agent on the catalog and YouTube transcripts for customer service. And if shipping is the pain point. Build an order exception system that flags delays before customers have to call."),
        ("q6-part4", "Days forty-five to ninety. Scale and systematize. At least two AI systems delivering measurable value. Platform architecture ready for productization. And most importantly. The team trusts that the tech is there to help them. Not replace them."),
    ],
    "q7-questions": [
        ("q7-part1", "Yeah, I do actually. I was wondering. What does the current tech team look like? Would I be building from scratch or inheriting a team?"),
    ],
    "q7b-shopify-question": [
        ("q7b-part1", "And one more. Where is the Shopify Plus migration at right now? Is there an expected go-live timeline?"),
    ],
    "q8-closing": [
        ("q8-part1", "Thank you. I really enjoyed this conversation. I've done deep research on the business and I'm genuinely excited about what you're building. Looking forward to the next step."),
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
                    model=model, text=text, ref_audio=REF_AUDIO,
                    ref_text=ref_text, instruct=INSTRUCT, output_path=temp_dir,
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

    print("\nConcatenating segments per question...")
    for question_key in ANSWERS:
        question_dir = os.path.join(OUTPUT_DIR, question_key)
        wavs = sorted([os.path.join(question_dir, f) for f in os.listdir(question_dir) if f.endswith(".wav")])
        if not wavs:
            continue
        concat_file = os.path.join(question_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for w in wavs:
                f.write(f"file '{w}'\n")
        output_path = os.path.join(OUTPUT_DIR, f"schneider-{question_key}.m4a")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c:a", "aac", "-b:a", "192k", output_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  -> {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  !! Concat failed for {question_key}: {e}")

    print("\nBuilding full practice file...")
    all_m4as = sorted([os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.startswith("schneider-q") and f.endswith(".m4a")])
    if all_m4as:
        full_concat = os.path.join(OUTPUT_DIR, "full-concat.txt")
        with open(full_concat, "w") as f:
            for m in all_m4as:
                f.write(f"file '{m}'\n")
        full_output = os.path.join(OUTPUT_DIR, "schneider-deep-interview-practice.m4a")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", full_concat, "-c:a", "aac", "-b:a", "192k", full_output]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"\n  DONE: {full_output}")
        except subprocess.CalledProcessError as e:
            print(f"  !! Full concat failed: {e}")

    desktop_copy = os.path.expanduser("~/Desktop/schneider-deep-interview-practice.m4a")
    src = os.path.join(OUTPUT_DIR, "schneider-deep-interview-practice.m4a")
    if os.path.exists(src):
        shutil.copy2(src, desktop_copy)
        print(f"  Copied to {desktop_copy}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
