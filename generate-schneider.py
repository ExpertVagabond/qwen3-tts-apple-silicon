#!/usr/bin/env python3
"""Generate Schneider interview answers using Matthew's cloned voice — Approach C.

Combines punchy rewritten text (fragments, pauses, rhetorical rhythm)
with emotion instruct for natural inflection.
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
OUTPUT_DIR = "/Volumes/Virtual Server/projects/schneider-ops/audio/matthew-clone-v2"

# Voice clone reference
REF_AUDIO = os.path.join(VOICES_DIR, "matthew.wav")
REF_TEXT_FILE = os.path.join(VOICES_DIR, "matthew.txt")

# Emotion instruct for natural delivery
INSTRUCT = "Speaking with conviction and natural emotion, like telling a personal story to someone you respect. Confident but genuine, with varied pacing and emphasis."

# ── Punchy Rewritten Answers ──
# Fragments, rhetorical pauses, natural speech rhythm

ANSWERS = {
    "q1-background": [
        ("q1-part1", "I started as a travel content creator. Seventy plus countries. Built an audience of twenty-nine thousand. Brand partnerships with major outdoor brands. But here's the thing. I was building on rented platforms. Someone else owned the algorithm. The distribution. The data. I had influence? Sure. But no equity in what I'd built."),
        ("q1-part2", "That's what pushed me into engineering. I founded Purple Squirrel Media and went deep. Python. TypeScript. Rust. Go. Ruby. In the last two years? Fifty-seven repositories. Eighteen hundred and forty-seven commits. AI automation systems. MCP servers. Mobile apps. Distributed infrastructure. All of it."),
        ("q1-part3", "The through-line is this. I build things that work in production. Not prototypes that demo well. And I do it almost entirely through AI-native development. Over sixty-eight hundred Claude Code sessions. I don't just use AI tools. I've built an entire autonomous development infrastructure around them. Custom hooks. Skills. Behavioral autopilot. Multi-agent orchestration."),
        ("q1-part4", "But what I'm looking for now? Somewhere to plant those skills in something real. Something physical. I've been building digital infrastructure that talks to other digital infrastructure. What drew me to Schneider is that the work actually lands somewhere. Four thousand saddles. A thousand pieces of gear. Customers who genuinely love their horses. A business that's been serving them for seventy-eight years. I want my engineering to express itself in something you can touch."),
    ],
    "q2-ai-workflow": [
        ("q2-part1", "I don't use AI as an assistant. AI is my primary developer. I'm the architect. The reviewer. The systems thinker."),
        ("q2-part2", "Here's what my setup actually looks like. I run Claude Code with five custom hooks. A pre-bash guard that catches dangerous commands. Auto-context injection that loads project state on session start. Auto-diagnose that reads errors and fixes them autonomously. Credential blocking that scans for nine secret patterns before every commit. And session handoff that preserves state across sessions."),
        ("q2-part3", "On top of that? Seven reusable skills. One-command deploy pipelines. Automated build failure diagnosis. Parallel codebase exploration with sub-agents. And behavioral autopilot rules. If I mention deploying? Pre-flight checks run automatically. Config file changes? Validation fires without me asking."),
        ("q2-part4", "I also run parallel Claude Code sessions on the same codebase. Multiple agents. Different parts of a problem. Simultaneously. With git-aware conflict prevention so they don't step on each other."),
        ("q2-part5", "The result. Eighteen hundred and forty-seven AI-generated commits across fifty-seven repos. Sixteen interconnected CLI tools with a hundred and sixty-three commands. Five MCP servers. Twelve plus production sites. All AI-generated. All reviewed. All in production."),
        ("q2-part6", "Now. Why that matters for Schneider specifically. You have a catalog of four thousand plus saddles and over a thousand accessories. Your customers are fanatics about their horses. The AI automation that serves them? It needs to be as reliable as the business they trust. My workflow isn't about writing code fast. It's about building systems that ship consistently. Validate automatically. And don't break in production. That's what your operations need."),
        ("q2-part7", "And here's something specific. I know Fulfil dot I O has native Claude MCP integration. I've built five MCP servers myself. Day one? I can wire up natural language operations queries against your BigQuery data warehouse. What blanket SKUs are under thirty days of inventory? No one logs into Fulfil. No one writes SQL. That's not a pitch. That's something I could ship in the first week."),
    ],
    "q3-cro": [
        ("q3-part1", "I'll be straight with you. My background isn't traditional CRO. Not in the A B test every button sense. But I've built the kind of infrastructure that makes real conversion improvement possible. Especially for a catalog business like Schneider's."),
        ("q3-part2", "Think about what kills conversion in a four thousand SKU equestrian catalog. Customers can't find the right saddle for their horse. They have questions at ten PM and nobody's there. Inventory says in-stock but the warehouse says otherwise. The site is slow on mobile at a horse show. These aren't marketing problems. They're systems problems."),
        ("q3-part3", "I did my homework on your current site. It's serving two hundred and twelve kilobyte HTML pages. That's four to seven times what a modern Shopify storefront generates. Your robots dot txt has a ten-second crawl delay. That tells me the server struggles with traffic spikes. When a rider at a horse show tries to look up an ARMORFlex blanket on their phone? That page weight matters. The Shopify Plus migration alone is going to be a massive conversion lever."),
        ("q3-part4", "But beyond performance. You're running Kustomer for customer service. Bloomreach for personalization. Noibu to catch JavaScript errors that kill checkout. You have the tools. What's missing? The AI layer that connects them. A customer asks Kustomer chat about blanket fit. Right now a human has to answer. An AI agent trained on your two hundred and thirty-three YouTube videos and thirty-three hundred product descriptions? It could handle sixty to seventy percent of those inquiries. And it would know the difference between VTEK Wither Relief and a standard Cutback fit."),
        ("q3-part5", "What I'd build is that system layer. AI-powered product discovery for your deep catalog. Intelligent customer service that actually understands equestrian gear. Inventory reconciliation between Fulfil and Shopify so what the site says matches what the warehouse has. Those aren't CRO experiments. They're operational improvements that directly drive conversion."),
    ],
    "q4-90-day-plan": [
        ("q4-part1", "First thing I'd do? Shut up and listen. For real."),
        ("q4-part2", "Days one through fourteen. Understand the business. Not just the stack. Yes, I'd audit the tech. Shopify Plus migration status. Fulfil ERP integration points. The BigQuery data warehouse. Your AWS infrastructure. And honestly? The legacy ASP dot NET platform too. Because understanding what you're migrating away from matters as much as understanding where you're going. I'd map every integration. Bloomreach. Kustomer. Emarsys. Global-e. Noibu. What's carrying over. What needs rebuilding."),
        ("q4-part3", "But more importantly? I'd spend time with the people who run the operation. The customer service team fielding calls from riders who need the right bit for their horse. The warehouse team picking and packing orders. The marketing team creating content for people who live and breathe equestrian. Technology serves these people. Not the other way around. I'd also watch Eric's YouTube content. Two hundred and thirty-three videos tells me everything about what this brand stands for."),
        ("q4-part4", "Days fifteen through forty-five. Quick wins that people can feel. First quick win. Fulfil already has native Claude MCP integration. I'd wire up natural language operations queries against your BigQuery warehouse in the first week. What blanket SKUs are under thirty days of inventory? Show me orders delayed more than seven days. No one logs into Fulfil. No one writes SQL. The ops team gets immediate value from AI. No custom development needed."),
        ("q4-part5", "Second. Customer service automation. I'd train an AI agent on your product catalog and YouTube transcripts to handle routine product-matching questions. What blanket for a sixteen-hand Thoroughbred in Ohio? So your CS team can focus on the complex issues. Ship it narrow at first. Expand as it proves itself."),
        ("q4-part6", "Third. If shipping is the pain point the team confirms. And your Trustpilot reviews suggest it is. I'd build an order exception system. Shopify webhooks to AWS Lambda to Fulfil API. Flag delays before the customer has to call. Proactive outreach instead of reactive complaints."),
        ("q4-part7", "When the team sees AI handling the routine stuff so they can focus on what matters? That's when the organization starts believing in what we're building."),
        ("q4-part8", "Days forty-five through ninety. Scale. Systematize. Start thinking about productization. Expand the automation portfolio into reporting, analytics, operations optimization. Start documenting patterns with an eye toward the bigger play. If other e-commerce companies want these tools? The architecture needs to be tenant-aware and configurable from the start. I'd also lock down the dev workflow. CI/CD. Testing. Deployment automation. So we can ship reliably and fast."),
        ("q4-part9", "By day ninety. At least two AI systems delivering measurable value to the team. A platform architecture ready for external productization. A clear roadmap for the next six months. But more than the deliverables? The team trusts that the technology is here to help them. Not replace them."),
    ],
    "q5-why-schneider": [
        ("q5-part1", "I've built a lot of things in the last two years. Fifty-seven repos. Automation engines. MCP servers. Mobile apps. But all of it? Digital infrastructure talking to other digital infrastructure. What drew me to Schneider is that the work lands somewhere real."),
        ("q5-part2", "You have four thousand saddles in the catalog. Another thousand pieces of gear before a rider can even get on their horse. Your customers aren't casual shoppers. They're fanatics who love their animals. Eric has three hundred and fifty videos on YouTube because he genuinely cares about this. That's not a CEO chasing a tech trend. That's a founder building something he believes in. And you've been doing it for seventy-eight years."),
        ("q5-part3", "I want my AI work to express itself in something physical. Faster fulfillment. Smarter inventory. A customer experience that matches the passion your buyers already have. That's what makes this different from every other Head of AI posting on the market. Most of those? Startups burning cash. Hoping the AI narrative carries them to the next funding round. Schneider is a profitable business with real products. Real customers. Real operations. The AI automation that comes out of a business like this will be better than anything a pure-play SaaS company builds. Because it's forged in actual operations."),
        ("q5-part4", "That's the foundation I want to build on. The equity path is exciting. I'm a founder. I think in terms of ownership. But honestly? Even without equity. Building AI systems that make a seventy-eight-year-old business run better and serve its customers better? That's meaningful work. The equity just means I get to go all-in."),
    ],
    "q6-wildcard": [
        ("q6-part1", "One thing I want to emphasize. I've done real homework on this business. Third-generation family company. USEA official horse wear sponsor. Multi-channel. DTC. Amazon. eBay. Physical retail. Wholesale. Chagrin Valley Farms. Eighty-five acres. Two hundred stalls. This isn't a company playing at e-commerce. This is a company that's been earning trust from riders for nearly eight decades."),
        ("q6-part2", "The AI automation tools that come out of a business like this? Built by people who actually run the operations. Tested against real seasonal demand. Refined by real customer interactions. Those tools will be better than anything a SaaS startup builds from the outside looking in. That's the moat. I want to help build that moat."),
        ("q6-part3", "And I'll just say this. I'm not here because I need a job. I've been building full-time as a founder for two years. I'm here because this is the right thing to build next. With the right people. On the right foundation."),
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
            if f.endswith(".wav") and f.startswith(question_key.split("-")[0])
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

        full_output = os.path.join(OUTPUT_DIR, "schneider-full-practice-matthew-v2.m4a")
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

    # Copy full practice to Desktop
    desktop_copy = os.path.expanduser("~/Desktop/schneider-full-practice-matthew-v2.m4a")
    src = os.path.join(OUTPUT_DIR, "schneider-full-practice-matthew-v2.m4a")
    if os.path.exists(src):
        shutil.copy2(src, desktop_copy)
        print(f"  Copied to {desktop_copy}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
