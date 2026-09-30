"""Scripted doctor-patient conversation, built from real INCLUDE clips.

Produces the videos and the transcript used in the presentation, and doubles as a
test that whole conversations survive the pipeline.

    python conversation_demo.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# What the deaf patient signs, turn by turn, and what the doctor replies.
CONVERSATION = [
    (["hello", "how_are_you"], "Hello. Please tell me what is wrong."),
    (["i", "sick", "today"], "Since when have you been feeling sick?"),
    (["i", "hot", "night"], "A fever at night. Anything else?"),
    (["i", "weak"], "I will examine you now."),
    (["i", "medicine"], "Yes, I will give you medicine."),
    (["thank_you"], "You are welcome. Get well soon."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()
    out_dir = HERE / "demo" / "conversation"
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    from make_demo_video import build

    transcript, exact = [], 0
    for i, (glosses, reply) in enumerate(CONVERSATION, 1):
        video = out_dir / f"turn{i}_{'_'.join(glosses)}.mp4"
        build(glosses, video, seed=11 + i)
        run = subprocess.run([sys.executable, str(HERE / "live_demo.py"),
                              "--video", str(video), "--headless", "--no-llm",
                              "--threshold", str(args.threshold)],
                             capture_output=True, text=True)
        raw = [l.split()[1] for l in run.stdout.splitlines() if l.startswith("sign:")]
        detected = [w for i, w in enumerate(raw) if i == 0 or w != raw[i - 1]]
        sentence = next((l.split(":", 1)[1].strip() for l in run.stdout.splitlines()
                         if l.startswith("SENTENCE:")), "(none)")
        ok = detected == glosses
        exact += ok
        print(f"\nturn {i}  patient signs: {' '.join(g.upper() for g in glosses)}")
        print(f"        detected     : {' '.join(detected) or '(nothing)'}  {'OK' if ok else 'MISMATCH'}")
        print(f"        shown to doctor: {sentence}")
        print(f"        doctor replies : {reply}")
        transcript.append({"signed": glosses, "detected": detected,
                           "sentence": sentence, "reply": reply})

    print(f"\n{exact}/{len(CONVERSATION)} turns recognised exactly")
    lines = ["# Demo conversation transcript\n"]
    for i, t in enumerate(transcript, 1):
        lines.append(f"**Turn {i}** — patient signs `{' '.join(t['signed'])}`\n")
        lines.append(f"- screen shows: **{t['sentence']}**")
        lines.append(f"- doctor replies: {t['reply']}\n")
    (out_dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    print("transcript written to", out_dir / "transcript.md")


if __name__ == "__main__":
    main()
