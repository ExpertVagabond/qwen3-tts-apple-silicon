#!/usr/bin/env python3
"""Generate Schneider interview answers using VoiceDesign model."""

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
OUTPUT_DIR = "/Volumes/Virtual Server/projects/schneider-ops/audio/voice-design"

VOICE_DESCRIPTION = "A young American male in his mid-30s with a confident, conversational tone. He speaks clearly with a natural West Coast accent, moderate pace, warm baritone voice. He sounds like a tech founder pitching to investors - articulate, persuasive, and grounded."

ANSWERS = {
    "q1-background": [
        ("q1-part1", "I started as a travel content creator. 70 plus countries, built an audience of 29,000, brand partnerships with major outdoor brands. But I realized I was building on rented platforms. Someone else owned the algorithm, the distribution, the data. I had influence but no equity in what I'd built."),
        ("q1-part2", "That's what pushed me into engineering. I founded Purple Squirrel Media and went deep. Python, TypeScript, Rust, Go, Ruby. In the last two years I've shipped 57 repositories and 1,847 commits. I've built AI automation systems, MCP servers, mobile apps, distributed infrastructure."),
        ("q1-part3", "The through-line is: I build things that work in production, not prototypes that demo well. And I do it almost entirely through AI-native development. Over 6,800 Claude Code sessions. I don't just use AI tools. I've built an entire autonomous development infrastructure around them. Custom hooks, skills, behavioral autopilot, multi-agent orchestration."),
        ("q1-part4", "But what I'm looking for now is somewhere to plant those skills in something real. Something physical. I've been building digital infrastructure that talks to other digital infrastructure. What drew me to Schneider is that the work actually lands somewhere. 4,000 saddles, a thousand pieces of gear, customers who genuinely love their horses, and a business that's been serving them for 78 years. I want my engineering to express itself in something you can touch."),
    ],
    "q2-ai-workflow": [
        ("q2-part1", "I don't use AI as an assistant. AI is my primary developer. I'm the architect, reviewer, and systems thinker."),
        ("q2-part2", "Here's what my setup actually looks like. I run Claude Code with 5 custom hooks. A pre-bash guard that catches dangerous commands, auto-context injection that loads project state on session start, auto-diagnose that reads errors and fixes them autonomously, credential blocking that scans for 9 secret patterns before every commit, and session handoff that preserves state across sessions."),
        ("q2-part3", "On top of that, I have 7 reusable skills. One-command deploy pipelines, automated build failure diagnosis, parallel codebase exploration with sub-agents. And behavioral autopilot rules. If I mention deploying, pre-flight checks run automatically. If a config file changes, validation fires without me asking."),
        ("q2-part4", "I also run parallel Claude Code sessions on the same codebase. Multiple agents working on different parts of a problem simultaneously, with git-aware conflict prevention so they don't step on each other."),
        ("q2-part5", "The result: 1,847 AI-generated commits across 57 repos. 16 interconnected CLI tools with 163 commands. 5 MCP servers. 12 plus production sites. All AI-generated, all reviewed, all in production."),
        ("q2-part6", "Now, why that matters for Schneider specifically. You have a catalog of 4,000 plus saddles and over a thousand accessories. You have customers who are fanatics about their horses. The AI automation that serves them needs to be as reliable as the business they trust. My workflow isn't just about writing code fast. It's about building systems that ship consistently, validate automatically, and don't break in production. That's what your operations need."),
        ("q2-part7", "And here's something specific. I know Fulfil dot I O has native Claude MCP integration. I've built 5 MCP servers myself. On day one, I can wire up natural language operations queries against your BigQuery data warehouse. What blanket SKUs are under 30 days of inventory? Without anyone logging into Fulfil or writing SQL. That's not a pitch. That's something I could ship in the first week."),
    ],
    "q3-cro": [
        ("q3-part1", "I'll be straight with you. My background isn't traditional CRO in the A B test every button sense. But I've built the kind of infrastructure that makes real conversion improvement possible, especially for a catalog business like Schneider's."),
        ("q3-part2", "Think about what kills conversion in a 4,000 SKU equestrian catalog. Customers can't find the right saddle for their horse. They have questions at 10pm and nobody's there. Inventory says in-stock but the warehouse says otherwise. The site is slow on mobile at a horse show. These are systems problems, not marketing problems."),
        ("q3-part3", "I did my homework on your current site. It's serving 212 kilobyte HTML pages. That's 4 to 7 times what a modern Shopify storefront generates. Your robots dot txt has a 10-second crawl delay, which tells me the server struggles with traffic spikes. When a rider at a horse show tries to look up an ARMORFlex blanket on their phone, that page weight matters. The Shopify Plus migration alone is going to be a massive conversion lever."),
        ("q3-part4", "But beyond performance, you're running Kustomer for customer service, Bloomreach for personalization, Noibu to catch JavaScript errors that kill checkout. You have the tools. What's missing is the AI layer that connects them. A customer asks Kustomer chat about blanket fit, and right now a human has to answer. An AI agent trained on your 233 YouTube videos and 3,334 product descriptions could handle 60 to 70 percent of those inquiries. And it would know the difference between VTEK Wither Relief and a standard Cutback fit."),
        ("q3-part5", "What I'd build is that system layer. AI-powered product discovery for your deep catalog, intelligent customer service that actually understands equestrian gear, and inventory reconciliation between Fulfil and Shopify that means what the site says matches what the warehouse has. Those aren't CRO experiments. They're operational improvements that directly drive conversion."),
    ],
    "q4-90-day-plan": [
        ("q4-part1", "First thing I'd do is shut up and listen. For real."),
        ("q4-part2", "Days 1 through 14. Understand the business, not just the stack. Yes, I'd audit the tech. The Shopify Plus migration status, Fulfil ERP integration points, the BigQuery data warehouse, your AWS infrastructure, and honestly the legacy ASP dot NET platform too. Because understanding what you're migrating away from matters as much as understanding where you're going. I'd map every integration. Bloomreach for personalization, Kustomer for CS, Emarsys for email, Global-e for international, Noibu for error monitoring. And understand what's carrying over and what needs rebuilding."),
        ("q4-part3", "But more importantly, I'd spend time with the people who run the operation. The customer service team fielding calls from riders who need the right bit for their horse. The warehouse team picking and packing orders. The marketing team creating content for people who live and breathe equestrian. Technology serves these people, not the other way around. I'd also watch Eric's YouTube content. 233 videos tells me everything about what this brand stands for and how the technology should reflect that."),
        ("q4-part4", "Days 15 through 45. Quick wins that people can feel. First quick win: Fulfil already has native Claude MCP integration. I'd wire up natural language operations queries against your BigQuery warehouse in the first week. What blanket SKUs are under 30 days of inventory? Or show me orders delayed more than 7 days. Without anyone logging into Fulfil or writing SQL. The ops team gets immediate value from AI without any custom development."),
        ("q4-part5", "Second, customer service automation. I'd train an AI agent on your product catalog and YouTube transcripts to handle the routine product-matching questions. What blanket for a 16-hand Thoroughbred in Ohio? So your CS team can focus on the complex issues. Ship it narrow at first, expand as it proves itself."),
        ("q4-part6", "Third, if shipping is the pain point the team confirms, and your Trustpilot reviews suggest it is, I'd build an order exception system. Shopify webhooks to AWS Lambda to Fulfil API. Flag delays before the customer has to call. Proactive outreach instead of reactive complaints."),
        ("q4-part7", "When the team sees AI handling the routine stuff so they can focus on what matters, that's when the organization starts believing in what we're building."),
        ("q4-part8", "Days 45 through 90. Scale, systematize, and start thinking about productization. Expand the automation portfolio into reporting, analytics, and operations optimization. Start documenting patterns with an eye toward the bigger play. If other e-commerce companies want these tools, the architecture needs to be tenant-aware and configurable from the start. I'd also lock down the development workflow. CI/CD, testing, deployment automation. So we can ship reliably and fast."),
        ("q4-part9", "By day 90: at least 2 AI systems delivering measurable value to the team, a platform architecture ready for external productization, and a clear roadmap for the next 6 months. But more than the deliverables, the team trusts that the technology is here to help them, not replace them."),
    ],
    "q5-why-schneider": [
        ("q5-part1", "I've built a lot of things in the last two years. 57 repos, automation engines, MCP servers, mobile apps. But all of it has been digital infrastructure talking to other digital infrastructure. What drew me to Schneider is that the work lands somewhere real."),
        ("q5-part2", "You have 4,000 saddles in the catalog. Another thousand pieces of gear before a rider can even get on their horse. Your customers aren't casual shoppers. They're fanatics who love their animals. Eric has 350 videos on YouTube because he genuinely cares about this. That's not a CEO chasing a tech trend. That's a founder building something he believes in. And you've been doing it for 78 years."),
        ("q5-part3", "I want my AI work to express itself in something physical. Faster fulfillment, smarter inventory, a customer experience that matches the passion your buyers already have. That's what makes this different from every other Head of AI posting on the market. Most of those are at startups burning cash, hoping the AI narrative carries them to the next funding round. Schneider is a profitable business with real products, real customers, and real operations. The AI automation that comes out of a business like this will be better than anything a pure-play SaaS company builds, because it's forged in actual operations."),
        ("q5-part4", "That's the foundation I want to build on. The equity path is exciting. I'm a founder, I think in terms of ownership. But honestly, even without equity, building AI systems that make a 78-year-old business run better and serve its customers better? That's meaningful work. The equity just means I get to go all-in."),
    ],
    "q6-wildcard": [
        ("q6-part1", "One thing I want to emphasize. I've done real homework on this business. Third-generation family company. USEA official horse wear sponsor. Multi-channel. DTC, Amazon, eBay, physical retail, wholesale. Chagrin Valley Farms, 85 acres, 200 stalls. This isn't a company playing at e-commerce. This is a company that's been earning trust from riders for nearly eight decades."),
        ("q6-part2", "The AI automation tools that come out of a business like this, built by people who actually run the operations, tested against real seasonal demand, refined by real customer interactions, those tools will be better than anything a SaaS startup builds from the outside looking in. That's the moat. I want to help build that moat."),
        ("q6-part3", "And I'll just say. I'm not here because I need a job. I've been building full-time as a founder for two years. I'm here because this is the right thing to build next, with the right people, on the right foundation."),
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

    model_folder = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
    model_path = get_smart_path(model_folder)
    if not model_path:
        print(f"Error: Model not found at {os.path.join(MODELS_DIR, model_folder)}")
        sys.exit(1)

    print(f"Loading VoiceDesign model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded.\n")

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
                    instruct=VOICE_DESCRIPTION,
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

        full_output = os.path.join(OUTPUT_DIR, "schneider-full-practice-voicedesign.m4a")
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

    print("\nAll done!")


if __name__ == "__main__":
    main()
