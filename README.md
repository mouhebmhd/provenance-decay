# Provenance-decay estimation (I(S; Y_k) vs hop k)

    pip install -r requirements.txt
    python estimate_decay.py --cascade cascade_output.jsonl --out decay_results \
        --embedder sbert --embed_model sentence-transformers/all-MiniLM-L6-v2 \
        --stratify credibility transform

Smoke test with no data / no downloads:

    python make_synthetic_cascade.py --out synth.jsonl --mode uniform
    python estimate_decay.py --cascade synth.jsonl --out res --embedder tfidf --min_sources_per_stratum 15

## What it computes
| Spec step | Where |
|---|---|
| 1 signals S | `source` (provenance_signal/DOI), `journal` (from DOI, or `--metadata`), `credibility`, `origin` |
| 2 estimator | L2-normalised embeddings -> logistic regression (few classes) / calibrated cosine-centroid softmax (source identity) |
| 3 MI, retention, accuracy | `I_hat = H(S) - held-out cross-entropy`; `R_k = I_k/H(S)`; also `retention_rel_to_hop0`, Fano lower bound, `mi_best_lb_bits` |
| 4 bias | cross-fitting (group folds by source for non-source signals), permutation null (subtracted only if positive), cluster bootstrap over sources for all CIs |
| 6 Gaussian | `noise_estimates.json` (across-sample variance per hop, sigma_N^2 slope); `gaussian_fit.json` |
| 7 k* | `k_star.json`: absolute (`I_k > H-delta`) and relative-to-hop-0 (`I_k > I_0-delta`), integer + interpolated + bootstrap CI |
| 8 strata | `--stratify credibility transform origin` |
| 9 validation | `validation.json` (DPI, Gaussian usable?, k* observed vs theory) |
| 10 artifacts | `decay_curves.csv`, `noise_estimates.json`, `k_star.json`, `per_transform_drop.csv`, `figures/` |

## Things you must check on real data
1. **Gaussian bound form.** `gaussian_bound()` uses an ASSUMED form (additive Gaussian, noise = floor + k*sigma_N^2). Replace it with your theorem's form. On synthetic data this assumed form fit poorly (embedding-space noise saturates instead of growing linearly), and `validation.json -> gaussian_form_usable` says so; do not trust theory k* unless R2 > 0.
2. **R_0 is not 1 for class-level signals.** For credibility/journal/origin the classifier must generalise to held-out sources, so I_0 < H(S) is normal. Use `retention_rel_to_hop0` and `k_star_rel` for those; `k_star_abs` is only meaningful when I_0 > H(S)-delta (source identity).
3. **DPI "violations" are estimator artifacts.** True MI is monotone, but a fixed-capacity probe can improve at later hops when nuisance variation shrinks. Use `mi_isotonic` for a monotone curve; `validation.json` lists hops with significant increases.
4. **Transformation types.** If `hop_name` is constant within a cascade the script stratifies by it; if it varies along the chain (`paraphrase>summarize>...`) it stratifies by the whole chain and `per_transform_drop.csv` gives the MI lost at each hop attributed to that hop's transformation (the RQ2 comparison).
5. **Only complete cascades are used** (all hops present) so every hop is measured on the same population.
6. Raw MI is not comparable across strata with different H(S) (different numbers of sources); compare `retention`.
7. If any hop translates into another language, use a multilingual embedder (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
