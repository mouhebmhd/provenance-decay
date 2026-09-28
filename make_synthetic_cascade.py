"""
Generates a SYNTHETIC cascade_output.jsonl with the same schema as the real one,
so estimate_decay.py can be smoke-tested offline. Source texts carry (i) source-
specific tokens, (ii) credibility-style tokens, and each hop randomly replaces a
fraction of tokens, with a different retention rate per transformation.
This is NOT real data -- it only checks that the pipeline produces sensible curves.

  python make_synthetic_cascade.py --out synth_chain.jsonl                # one fixed chain of 5 different hops
  python make_synthetic_cascade.py --out synth_uniform.jsonl --mode uniform   # per-source transformation type
"""
import argparse, json, random
from datetime import datetime, timezone

CHAIN = [("paraphrase", 0.85), ("summarize", 0.70), ("translate_roundtrip", 0.80),
         ("paraphrase", 0.85), ("style_tweet", 0.55)]
UNIFORM = {"paraphrase": 0.85, "summarize": 0.70, "style_tweet": 0.55}
TRUSTED = ["randomized", "cohort", "peer-reviewed", "confidence", "placebo", "mechanism", "guideline", "replicated"]
SENSATIONAL = ["miracle", "secret", "doctors-hate", "instantly", "cure-all", "shocking", "banned", "hidden-truth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_cascade.jsonl")
    ap.add_argument("--mode", choices=["chain", "uniform"], default="chain")
    ap.add_argument("--n_sources", type=int, default=80)
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    vocab = [f"w{i:04d}" for i in range(5000)]
    filler = [f"f{i:03d}" for i in range(300)]

    sources = []
    for i in range(a.n_sources):
        if i % 4 != 3:  # PLOS, trusted
            jn = rng.choice(["pone", "pone", "pone", "pmed", "pbio"])
            sid, doi, origin, cred = f"plos_{i:06d}", f"10.1371/journal.{jn}.{rng.randint(100000, 999999):07d}", "PLOS", 1
        else:           # HealthNewsReview, mixed credibility
            sid, doi, origin = f"hnr_{i:06d}", f"hnr_{i:06d}", "HealthNewsReview"
            cred = rng.choice([0, 0, 1])
        style = rng.sample(TRUSTED if cred else SENSATIONAL, 5)
        toks = rng.sample(vocab, 25) + style + rng.sample(filler, 12)
        rng.shuffle(toks)
        sources.append(dict(sid=sid, doi=doi, origin=origin, cred=cred, toks=toks, style=set(style),
                            ttype=rng.choice(list(UNIFORM))))

    def hop(tokens, style, r):
        out = []
        for t in tokens:
            keep = r + (1 - r) * 0.5 if t in style else r      # style tokens are stickier
            out.append(t if rng.random() < keep else rng.choice(vocab + filler))
        return out

    n = 0
    with open(a.out, "w") as f:
        for s in sources:
            for smp in range(a.n_samples):
                cur = list(s["toks"])
                for h in range(1, 6):
                    name, r = CHAIN[h - 1] if a.mode == "chain" else (s["ttype"], UNIFORM[s["ttype"]])
                    nxt = hop(cur, s["style"], r)
                    f.write(json.dumps(dict(
                        cascade_id=f"{s['sid']}_{smp}", source_record_id=s["sid"], hop_index=h, hop_name=name,
                        sample_index=smp, model="synthetic", input_text=" ".join(cur), output_text=" ".join(nxt),
                        provenance_signal=s["doi"], credibility_label=s["cred"], origin=s["origin"],
                        timestamp=datetime.now(timezone.utc).isoformat(), temperature=0.7, top_p=0.9,
                        max_new_tokens=512, seed=smp)) + "\n")
                    cur = nxt
                    n += 1
    print(f"wrote {n} rows to {a.out}")


if __name__ == "__main__":
    main()
