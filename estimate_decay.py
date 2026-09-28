#!/usr/bin/env python3
"""
Estimate provenance decay I(S; Y_k) from a generative cascade.

Input : cascade_output.jsonl  (one row per cascade x hop; fields as in your
        example: cascade_id, source_record_id, hop_index, hop_name,
        sample_index, input_text, output_text, provenance_signal,
        credibility_label, origin, ...)
Output: <out>/decay_curves.csv, noise_estimates.json, k_star.json,
        gaussian_fit.json, validation.json, per_transform_drop.csv,
        figures/*.png

Pipeline (mirrors the 10 steps of the spec)
  1. Signals S:   source (provenance_signal / DOI), journal, credibility, origin
  2. Estimator:   embeddings -> logistic regression (few classes) or
                  cosine-centroid softmax (many classes, e.g. DOI)
  3. MI:          I_hat = H(S) - CE_heldout   (Barber-Agakov lower bound on
                  I(S;Y_k), evaluated on out-of-fold predictions), plus a
                  Fano lower bound from accuracy as a calibration-free check
  4. Bias:        cross-fitting, permutation null, cluster bootstrap CIs
  5. Curves:      MI, retention R_k = I_k / H(S), accuracy vs hop
  6. Gaussian:    per-hop noise sigma_N^2 from across-sample variance;
                  predicted curve vs observed (R^2, RMSE, KL)
  7. k*:          largest k with I_k > H(S) - delta (bootstrap CI)
  8. Strata:      by credibility class and/or transformation type
  9. Validation:  DPI monotonicity, Gaussian-bound check, Fano/theory k*
 10. Artifacts:   csv / json / figures

Usage
  python estimate_decay.py --cascade cascade_output.jsonl --out decay_results \
      --embedder sbert --embed_model sentence-transformers/all-MiniLM-L6-v2 \
      --stratify credibility transform
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-9
SIGNAL_COL = {"source": "provenance_signal", "journal": "journal",
              "credibility": "credibility_label", "origin": "origin"}
REQUIRED = ["cascade_id", "source_record_id", "hop_index", "hop_name", "input_text",
            "output_text", "provenance_signal", "credibility_label", "origin"]


def log(*a):
    print(*a, flush=True)


# =============================================================================
# 1. Loading + hop table
# =============================================================================
PLOS_JOURNALS = {"pone": "PLOS ONE", "pmed": "PLOS Medicine", "pbio": "PLOS Biology",
                 "pcbi": "PLOS Comput Biol", "ppat": "PLOS Pathogens",
                 "pgen": "PLOS Genetics", "pntd": "PLOS NTD"}


def derive_journal(doi: str, origin: str) -> str:
    """Journal/outlet from the DOI when possible (PLOS DOIs encode the journal),
    else the DOI registrant prefix, else the origin. Override with --metadata."""
    d = str(doi)
    m = re.match(r"^10\.1371/journal\.([a-z]+)\.", d, re.I)
    if m:
        return PLOS_JOURNALS.get(m.group(1).lower(), "PLOS-" + m.group(1).lower())
    m = re.match(r"^(10\.\d{4,9})/", d)
    if m:
        return "doi:" + m.group(1)
    return str(origin)


def load_cascade(path:a str, metadata: Optional[str], max_cascades: Optional[int]):
    df = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"cascade file is missing columns: {missing}")
    df = df.drop_duplicates(subset=["cascade_id", "hop_index"], keep="last")
    df["hop_index"] = df["hop_index"].astype(int)
    for c in ("input_text", "output_text"):
        df[c] = df[c].fillna("").astype(str)

    # journal column: metadata file > existing column > derived from DOI
    if metadata:
        md = pd.read_json(metadata, lines=True) if metadata.endswith(".jsonl") else pd.read_csv(metadata)
        df = df.drop(columns=[c for c in ["journal"] if c in df.columns]).merge(
            md[["source_record_id", "journal"]], on="source_record_id", how="left")
    if "journal" not in df.columns:
        df["journal"] = [derive_journal(d, o) for d, o in zip(df["provenance_signal"], df["origin"])]
    df["journal"] = df["journal"].fillna("unknown")

    # keep only COMPLETE cascades so every hop is computed on the same population
    # (otherwise survivorship changes the sample from hop to hop and fakes a decay)
    hops = sorted(df["hop_index"].unique())
    n_hops = df.groupby("cascade_id")["hop_index"].nunique()
    complete = n_hops[n_hops == len(hops)].index
    dropped = df["cascade_id"].nunique() - len(complete)
    if dropped:
        log(f"[data] dropping {dropped} incomplete cascades (missing hops)")
    df = df[df["cascade_id"].isin(complete)]
    if max_cascades:
        keep = df["cascade_id"].drop_duplicates().sample(min(max_cascades, len(complete)), random_state=0)
        df = df[df["cascade_id"].isin(keep)]
    return df, hops


def build_tables(df: pd.DataFrame, hops: List[int]):
    """Return casc (one row/cascade), long hop table with k = 0..K, and hop-name matrix."""
    meta_cols = ["cascade_id", "source_record_id", "provenance_signal", "credibility_label", "origin", "journal"]
    if "sample_index" in df.columns:
        meta_cols.append("sample_index")
    first_hop = min(hops)
    if first_hop == 0:  # hop 0 already present -> Y_0 is its output
        long = df.assign(k=df["hop_index"], text=df["output_text"])
    else:               # Y_0 = input_text of the first hop; Y_k = output_text of hop k
        y0 = df[df["hop_index"] == first_hop].assign(k=0, text=lambda d: d["input_text"], hop_name="source")
        yk = df.assign(k=df["hop_index"], text=df["output_text"])
        long = pd.concat([y0, yk], ignore_index=True)
    long = long[meta_cols + ["k", "hop_name", "text"]]
    # after this, hop indices are renumbered 0..K in order
    ks = sorted(long["k"].unique())
    remap = {k: i for i, k in enumerate(ks)}
    long["k"] = long["k"].map(remap)

    casc = (long[long["k"] == 0][meta_cols].drop_duplicates("cascade_id")
            .sort_values("cascade_id").reset_index(drop=True))
    return casc, long


def add_transform_type(casc, long, mode: str, max_groups: int = 12):
    later = long[long["k"] > 0].sort_values(["cascade_id", "k"])
    chain = later.groupby("cascade_id")["hop_name"].apply(lambda s: ">".join(s))
    uniform = later.groupby("cascade_id")["hop_name"].nunique().eq(1).all()
    if mode == "auto":
        mode = "cascade" if uniform else "chain"
    if mode == "cascade" and not uniform:
        log("[data] hop_name varies within cascades -> using full chain as transformation type")
        mode = "chain"
    if mode == "cascade":
        tt = later.groupby("cascade_id")["hop_name"].first()
    elif mode == "chain":
        tt = chain
    else:
        tt = pd.Series("all", index=casc["cascade_id"])
    casc["chain"] = casc["cascade_id"].map(chain)
    casc["transform_type"] = casc["cascade_id"].map(tt)
    if casc["transform_type"].nunique() > max_groups:
        log(f"[data] {casc['transform_type'].nunique()} distinct transformation types (> {max_groups}); "
            "collapsing to 'all' for stratification (per-hop-name drops are still reported)")
        casc["transform_type"] = "all"
    log(f"[data] transformation mode = {mode}; types = {sorted(casc['transform_type'].unique())[:6]}")
    return casc


# =============================================================================
# 2. Embeddings
# =============================================================================
def embed_texts(texts: List[str], backend: str, model_name: str, cache_dir: Path,
                batch_size: int = 64, device: Optional[str] = None) -> np.ndarray:
    key = hashlib.sha1((backend + model_name + "\x1e" + "\x1f".join(texts)).encode()).hexdigest()[:16]
    cache = cache_dir / f"emb_{backend}_{key}.npy"
    if cache.exists():
        log(f"[embed] loaded cache {cache.name}")
        return np.load(cache)
    if backend == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        X = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)
    elif backend == "tfidf":  # offline fallback, no downloads (used for the synthetic test)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=60000).fit_transform(texts)
        X = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=0).fit_transform(X)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    else:
        sys.exit(f"unknown embedder {backend}")
    X = X.astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


# =============================================================================
# 3. Probability estimators (out-of-fold)
# =============================================================================
def _normed(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)


def predict_logreg(Xtr, ytr, Xte, M, C):
    clf = LogisticRegression(C=C, max_iter=500)
    clf.fit(Xtr, ytr)
    P = np.full((len(Xte), M), 1e-6)
    P[:, clf.classes_] = clf.predict_proba(Xte)
    return P / P.sum(1, keepdims=True)


def predict_centroid(Xtr, ytr, Xte, M, default_tau=0.05):
    """Cosine-to-class-centroid softmax. Temperature chosen by leave-one-out NLL on
    the TRAIN fold only (no test leakage). Good for high-cardinality S (e.g. DOI)."""
    d = Xtr.shape[1]
    S = np.zeros((M, d), dtype=np.float64)
    np.add.at(S, ytr, Xtr)
    n = np.bincount(ytr, minlength=M)
    C = _normed(S)
    present = n > 0
    tau = default_tau
    idx = np.where(n[ytr] >= 2)[0]
    if len(idx) >= 20:
        Xc, yc = Xtr[idx], ytr[idx]
        sims = Xc @ C.T
        loo = (S[yc] - Xc) / (n[yc] - 1)[:, None]
        sims[np.arange(len(idx)), yc] = np.einsum("ij,ij->i", Xc, _normed(loo))
        sims[:, ~present] = -np.inf
        best = (np.inf, tau)
        for t in np.logspace(-2.5, 0, 16):
            lg = sims / t
            nll = -(lg[np.arange(len(idx)), yc] - logsumexp(lg, axis=1)).mean()
            if nll < best[0]:
                best = (nll, t)
        tau = best[1]
    sims = Xte @ C.T
    sims[:, ~present] = -np.inf
    lg = sims / tau
    P = np.exp(lg - logsumexp(lg, axis=1, keepdims=True))
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def oof_proba(X, y, fold, n_folds, M, method, C):
    P = np.zeros((len(y), M))
    for f in range(n_folds):
        te = fold == f
        fn = predict_logreg if method == "logreg" else predict_centroid
        P[te] = fn(X[~te], y[~te], X[te], M, C) if method == "logreg" else fn(X[~te], y[~te], X[te], M)
    return P


def make_folds(y, src, signal, n_splits, seed):
    """source-identity: stratified over cascades (class = the source, so the source must
    appear in train). every other signal: GROUP folds by source, so a classifier can't win
    by memorising a source it saw in training."""
    if signal == "source":
        n = min(n_splits, int(np.bincount(y).min()))
        if n < 2:
            return None, 0
        fold = np.full(len(y), -1)
        for i, (_, te) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            fold[te] = i
        return fold, n
    first = pd.DataFrame({"g": src, "y": y}).drop_duplicates("g")
    n = min(n_splits, int(first["y"].value_counts().min()))
    if n < 2:
        return None, 0
    fold = np.full(len(y), -1)
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, src)):
        fold[te] = i
    return fold, n


def permute_codes(y, src, signal, rng):
    if signal == "source":
        return rng.permutation(y)
    first = pd.DataFrame({"g": src, "y": y}).drop_duplicates("g")
    perm = dict(zip(first["g"], rng.permutation(first["y"].to_numpy())))
    return np.array([perm[g] for g in src])


# =============================================================================
# 4. Information quantities
# =============================================================================
def entropy_bits(y) -> float:
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def fano_lower_bound(H, M, acc):
    pe = 1.0 - acc
    hb = 0.0 if pe <= 0 or pe >= 1 else -(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe))
    return float(np.clip(H - hb - pe * np.log2(max(M - 1, 1)), 0, H))


def k_star(mi, thr):
    """Largest k such that I_k > thr contiguously from k=0. Returns (k, k_continuous, censored)."""
    if mi[0] <= thr:
        return None, None, False
    k = 0
    while k + 1 < len(mi) and mi[k + 1] > thr:
        k += 1
    if k == len(mi) - 1:
        return k, float(k), True
    return k, float(k + (mi[k] - thr) / max(mi[k] - mi[k + 1], 1e-12)), False


# =============================================================================
# 5. Run one (stratum, signal)
# =============================================================================
def run_signal(casc, IDX, E, sub_idx, signal, cfg, rng):
    c = casc.iloc[sub_idx].reset_index(drop=True)
    pos = np.asarray(sub_idx)
    y_str = c[SIGNAL_COL[signal]].astype(str).to_numpy(dtype=object)

    if signal == "source":
        vc = pd.Series(y_str).value_counts()
        keep = (pd.Series(y_str).map(vc).to_numpy() >= 2)      # need >=2 cascades to train and test
    else:
        keep = np.ones(len(c), bool)
        if signal != "credibility":                            # merge rare journals/origins
            first = c.assign(_y=y_str).drop_duplicates("source_record_id")
            vc = first["_y"].value_counts()
            rare = set(vc[vc < cfg.min_class_sources].index)
            if rare:
                y_str = np.array(["OTHER" if v in rare else v for v in y_str], dtype=object)
    c, pos, y_str = c[keep].reset_index(drop=True), pos[keep], y_str[keep]
    if len(c) < 20:
        return None
    classes, y = np.unique(y_str, return_inverse=True)
    M, H = len(classes), entropy_bits(y)
    if M < 2 or H < cfg.min_entropy:
        return None
    src = c["source_record_id"].to_numpy()
    fold, n_folds = make_folds(y, src, signal, cfg.n_splits, cfg.seed)
    if fold is None:
        return None
    # source identity ALWAYS uses the calibrated centroid estimator (same estimator in every stratum,
    # so curves are comparable); low-cardinality signals use regularised logistic regression
    method = cfg.estimator if cfg.estimator != "auto" else ("centroid" if (signal == "source" or M > 50) else "logreg")
    K1 = IDX.shape[0]
    Xs = [E[IDX[k, pos]] for k in range(K1)]

    logp, correct = [], []
    for k in range(K1):
        P = oof_proba(Xs[k], y, fold, n_folds, M, method, cfg.C)
        pt = np.clip(P[np.arange(len(y)), y], 1e-9, 1.0)
        logp.append(np.log2(pt))
        correct.append(P.argmax(1) == y)

    null = []
    for r in range(cfg.n_perm):
        yp = permute_codes(y, src, signal, rng)
        fp, nf = make_folds(yp, src, signal, cfg.n_splits, cfg.seed + 1 + r)
        if fp is None:
            continue
        Hp = entropy_bits(yp)
        row = []
        for k in range(K1):
            P = oof_proba(Xs[k], yp, fp, nf, M, method, cfg.C)
            ce = -np.log2(np.clip(P[np.arange(len(yp)), yp], 1e-9, 1)).mean()
            row.append(Hp - ce)
        null.append(row)

    return dict(signal=signal, classes=classes, M=M, H=H, y=y, src=src, pos=pos, method=method,
                n_folds=n_folds, logp=np.stack(logp), correct=np.stack(correct),
                null=np.array(null) if null else np.zeros((0, K1)), n=len(y))


# =============================================================================
# 6. Noise estimation + Gaussian hop-decay bound
# =============================================================================
def between_class_var(X, y):
    m = X.mean(0)
    tot = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        tot += len(Xc) * float(((Xc.mean(0) - m) ** 2).sum())
    return tot / len(X)


def noise_stats_for(E, IDX, pos, src_ids):
    K1 = IDX.shape[0]
    codes, _ = pd.factorize(src_ids)
    cnt = np.bincount(codes)
    m = cnt[codes] >= 2
    codes_m, _ = pd.factorize(codes[m])
    S = codes_m.max() + 1 if m.any() else 0
    within, between = [], []
    for k in range(K1):
        X = E[IDX[k, pos]]
        Xm = X[m]
        if S >= 2:
            means = pd.DataFrame(Xm).groupby(codes_m).transform("mean").to_numpy()
            within.append(float(((Xm - means) ** 2).sum() / max(len(Xm) - S, 1)))
            between.append(between_class_var(X, codes))
        else:
            within.append(float("nan")); between.append(float("nan"))
    w = np.array(within)
    ks = np.arange(K1)
    slope = float((ks[1:] * w[1:]).sum() / (ks[1:] ** 2).sum()) if np.isfinite(w).all() else float("nan")
    return dict(within_var=within, between_source_var=between,
                increments=[None] + [float(w[i] - w[i - 1]) for i in range(1, K1)],
                sigma_n2_slope=slope,
                sigma_n2_mean_increment=float(np.nanmean(np.diff(w))) if K1 > 1 else float("nan"),
                n_sources_used=int(S), n_cascades_used=int(m.sum()))


def gaussian_bound(k, P, floor, s2, H):
    """Assumed form (REPLACE with your paper's bound if it differs):
    additive-Gaussian cascade, noise variance floor + k*sigma_N^2 against signal power P,
    I(k) = 0.5*log2(1 + P / (floor + k*sigma_N^2)), capped at H(S) for discrete S."""
    return np.minimum(H, 0.5 * np.log2(1.0 + P / (floor + np.asarray(k) * s2 + EPS)))


def gaussian_analysis(ks, mi, lo, hi, H, P, s2_measured, delta):
    out = {"P_signal_power": P, "sigma_n2_measured": s2_measured}
    # (a) zero-free-parameter prediction: floor calibrated so the curve matches I_0,
    #     slope = measured sigma_N^2
    I0 = float(np.clip(mi[0], 1e-6, H - 1e-6))
    floor = P / (2 ** (2 * I0) - 1)
    pred = gaussian_bound(ks, P, floor, s2_measured, H) if np.isfinite(s2_measured) else np.full(len(ks), np.nan)
    # (b) two-parameter least-squares fit (floor, sigma_N^2) for comparison
    def resid(theta):
        return gaussian_bound(ks, P, np.exp(theta[0]), np.exp(theta[1]), H) - mi
    try:
        sol = least_squares(resid, x0=[np.log(max(floor, 1e-9)), np.log(max(abs(s2_measured) if np.isfinite(s2_measured) else 1e-3, 1e-9))])
        f_fit, s_fit = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
        fitc = gaussian_bound(ks, P, f_fit, s_fit, H)
    except Exception:
        f_fit = s_fit = float("nan"); fitc = np.full(len(ks), np.nan)

    def r2(a, b):
        ss = ((a - a.mean()) ** 2).sum()
        return float(1 - ((a - b) ** 2).sum() / ss) if ss > 0 else float("nan")

    def kl(a, b):
        p = np.clip(a, EPS, None); q = np.clip(b, EPS, None)
        p, q = p / p.sum(), q / q.sum()
        return float((p * np.log(p / q)).sum())

    def block(curve):
        if not np.isfinite(curve).all():
            return None
        return dict(curve=[float(v) for v in curve], R2=r2(mi, curve),
                    RMSE=float(np.sqrt(((mi - curve) ** 2).mean())), KL_normalized=kl(mi, curve),
                    mean_signed_error=float((curve - mi).mean()),
                    frac_hops_observed_CI_below_bound=float((lo <= curve).mean()),
                    verdict=("form does NOT fit the observed curve (R2<0): treat predicted curve / theory k* as unreliable"
                             if r2(mi, curve) < 0 else
                             "predicted curve is ABOVE observed on average (bound holds, conservative)" if (curve - mi).mean() > 0.02 * H
                             else "predicted curve is BELOW observed on average (bound violated / noise overestimated)" if (curve - mi).mean() < -0.02 * H
                             else "predicted ~ observed (tight)"))
    out["predicted_measured_noise"] = block(pred)
    out["fitted"] = block(fitc)
    out["fitted_params"] = dict(floor=f_fit, sigma_n2=s_fit)
    # theoretical k* from the closed form:  0.5 log2(1+P/(floor+k s2)) > H - delta
    def kth(fl, s2):
        if not np.isfinite(s2) or s2 <= 0:
            return None
        thr = 2 ** (2 * (H - delta)) - 1
        return float(max((P / thr - fl) / s2, 0.0))
    out["k_star_theory_measured_noise"] = kth(floor, s2_measured)
    out["k_star_theory_fitted"] = kth(f_fit, s_fit)
    return out


# =============================================================================
# 7. Summaries: curves, bootstrap CI, k*, DPI, per-hop drops
# =============================================================================
def summarize(res, HN, stratum, cfg, rng, E, IDX):
    K1 = res["logp"].shape[0]
    ks = np.arange(K1)
    H, M, y = res["H"], res["M"], res["y"]
    ce = -res["logp"].mean(1)
    mi_raw = H - ce
    null_mean = np.maximum(res["null"].mean(0), 0) if len(res["null"]) else np.zeros(K1)
    mi = np.clip(mi_raw - null_mean, 0, H)
    acc = res["correct"].mean(1)
    fano = np.array([fano_lower_bound(H, M, a) for a in acc])
    iso = IsotonicRegression(increasing=False).fit_transform(ks, mi)
    chance = float(np.bincount(y).max() / len(y))

    # cluster bootstrap over source records (resample sources, keep all hops together)
    usrc, sinv = np.unique(res["src"], return_inverse=True)
    groups = [np.where(sinv == i)[0] for i in range(len(usrc))]
    B = cfg.n_boot
    bmi, bacc = np.zeros((B, K1)), np.zeros((B, K1))
    bH = np.zeros(B)
    for b in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(usrc), len(usrc))])
        Hb = entropy_bits(y[idx])
        bH[b] = Hb
        bmi[b] = np.clip(Hb + res["logp"][:, idx].mean(1) - null_mean, 0, Hb)
        bacc[b] = res["correct"][:, idx].mean(1)
    q = lambda a: (np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0))
    mi_lo, mi_hi = q(bmi)
    ret_b = bmi / np.maximum(bH, EPS)[:, None]
    ret_lo, ret_hi = q(ret_b)
    acc_lo, acc_hi = q(bacc)

    rows = []
    for k in range(K1):
        rows.append(dict(stratum=stratum, signal=res["signal"], hop=k, n_records=res["n"], n_classes=M,
                         H_S_bits=H, estimator=res["method"], mi_bits=mi[k], mi_ci_lo=mi_lo[k], mi_ci_hi=mi_hi[k],
                         mi_raw_bits=mi_raw[k], mi_null_bits=null_mean[k], mi_isotonic=iso[k],
                         mi_fano_bits=fano[k], mi_best_lb_bits=max(mi[k], fano[k]), retention=mi[k] / H, retention_ci_lo=ret_lo[k],
                         retention_ci_hi=ret_hi[k], retention_rel_to_hop0=(mi[k] / mi[0]) if mi[0] > EPS else np.nan,
                         accuracy=acc[k], acc_ci_lo=acc_lo[k], acc_ci_hi=acc_hi[k], chance_accuracy=chance,
                         cross_entropy_bits=ce[k]))

    # k*
    ks_out = {"H_S_bits": H, "n_classes": M, "delta_bits": cfg.delta, "mi_hop0_bits": float(mi[0]),
              "estimator_reaches_H_minus_delta_at_hop0": bool(mi[0] > H - cfg.delta)}
    for name, thr, thr_b in [("k_star_abs", H - cfg.delta, bH - cfg.delta),
                             ("k_star_rel", mi[0] - cfg.delta, bmi[:, 0] - cfg.delta)]:
        kk, kc, cens = k_star(mi, thr)
        reps = [k_star(bmi[b], thr_b[b] if hasattr(thr_b, "__len__") else thr_b)[0] for b in range(B)]
        ok = np.array([r for r in reps if r is not None], dtype=float)
        ks_out[name] = dict(value=kk, continuous=kc, right_censored=cens,
                            ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else None,
                            frac_bootstrap_defined=float(len(ok) / B))

    # DPI
    p_inc = [None] + [float((bmi[:, k] > bmi[:, k - 1]).mean()) for k in range(1, K1)]
    d = np.diff(mi)
    dpi = dict(n_increases=int((d > 0).sum()), max_increase_bits=float(max(d.max(), 0)),
               bootstrap_p_increase_per_hop=p_inc,
               significant_violations=[k for k in range(1, K1) if p_inc[k] is not None and p_inc[k] > 0.975],
               monotone_after_isotonic=True)

    # marginal drop attributed to the transformation used at each hop
    drops = []
    for k in range(1, K1):
        names = HN[k - 1, res["pos"]]
        for nm in np.unique(names):
            sel = np.where(names == nm)[0]
            if len(sel) < 10:
                continue
            Hs = entropy_bits(y[sel])
            mprev = float(np.clip(Hs + res["logp"][k - 1, sel].mean() - null_mean[k - 1], 0, Hs))
            mcur = float(np.clip(Hs + res["logp"][k, sel].mean() - null_mean[k], 0, Hs))
            row = dict(stratum=stratum, signal=res["signal"], hop=k, hop_name=nm, n=len(sel),
                       mi_prev=mprev, mi_curr=mcur, drop_bits=mprev - mcur,
                       drop_fraction_of_H=(mprev - mcur) / Hs if Hs > 0 else np.nan)
            if len(sel) == res["n"]:
                dd = bmi[:, k - 1] - bmi[:, k]
                row["drop_ci_lo"], row["drop_ci_hi"] = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            drops.append(row)
    return rows, ks_out, dpi, drops, mi, mi_lo, mi_hi


# =============================================================================
# 8. Plots
# =============================================================================
def plot_metric(df, metric, lo, hi, ylabel, path, hue, title, extra=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, g in df.groupby(hue):
        g = g.sort_values("hop")
        ax.plot(g["hop"], g[metric], marker="o", label=str(name))
        if lo in g and hi in g:
            ax.fill_between(g["hop"], g[lo], g[hi], alpha=0.15)
    if extra:
        extra(ax)
    ax.set_xlabel("hop k"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_xticks(sorted(df["hop"].unique())); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_gaussian(ks, mi, lo, hi, H, ga, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(ks, mi, yerr=[np.maximum(mi - lo, 0), np.maximum(hi - mi, 0)], fmt="o-", label="observed (CI95)", capsize=3)
    if ga.get("predicted_measured_noise"):
        ax.plot(ks, ga["predicted_measured_noise"]["curve"], "s--", label="Gaussian, measured $\\sigma_N^2$")
    if ga.get("fitted"):
        ax.plot(ks, ga["fitted"]["curve"], "^:", label="Gaussian, 2-param fit")
    ax.axhline(H, color="gray", ls=":", label="H(S)")
    ax.set_xlabel("hop k"); ax.set_ylabel("I(S;Y_k) [bits]"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xticks(ks)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================================
# 9. Main
# =============================================================================
def build_strata(casc, stratify):
    strata = {"all": np.arange(len(casc))}
    if "credibility" in stratify:
        for v, g in casc.groupby("credibility_label"):
            strata[f"cred={v}"] = g.index.to_numpy()
    if "transform" in stratify:
        for v, g in casc.groupby("transform_type"):
            if v != "all" or len(casc["transform_type"].unique()) > 1:
                strata[f"transform={v}"] = g.index.to_numpy()
    if "origin" in stratify:
        for v, g in casc.groupby("origin"):
            strata[f"origin={v}"] = g.index.to_numpy()
    if "credibility" in stratify and "transform" in stratify:
        for (t, v), g in casc.groupby(["transform_type", "credibility_label"]):
            strata[f"transform={t}|cred={v}"] = g.index.to_numpy()
    # drop duplicates of 'all' (e.g. a single transform type)
    seen, out = [], {}
    for k, v in strata.items():
        key = tuple(v)
        if k != "all" and key in seen:
            continue
        seen.append(key); out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cascade", required=True)
    ap.add_argument("--out", default="decay_results")
    ap.add_argument("--metadata", help="optional csv/jsonl with source_record_id,journal")
    ap.add_argument("--embedder", choices=["sbert", "tfidf"], default="sbert")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--signals", nargs="+", default=["source", "journal", "credibility", "origin"], choices=list(SIGNAL_COL))
    ap.add_argument("--stratify", nargs="*", default=["credibility", "transform"], choices=["credibility", "transform", "origin"])
    ap.add_argument("--transform_mode", choices=["auto", "cascade", "chain", "none"], default="auto")
    ap.add_argument("--estimator", choices=["auto", "logreg", "centroid"], default="auto")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--n_perm", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.1, help="tolerated MI gap in bits for k*")
    ap.add_argument("--min_class_sources", type=int, default=5)
    ap.add_argument("--min_entropy", type=float, default=0.05)
    ap.add_argument("--min_sources_per_stratum", type=int, default=20)
    ap.add_argument("--max_cascades", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    df, hops = load_cascade(cfg.cascade, cfg.metadata, cfg.max_cascades)
    casc, long = build_tables(df, hops)
    casc = add_transform_type(casc, long, cfg.transform_mode)
    K1 = long["k"].nunique()
    log(f"[data] {len(casc)} cascades, {casc['source_record_id'].nunique()} sources, hops 0..{K1 - 1}")

    codes, uniq = pd.factorize(long["text"])
    long["emb_idx"] = codes
    E = embed_texts(list(uniq), cfg.embedder, cfg.embed_model, out / "cache", device=cfg.device)
    IDX = long.pivot(index="cascade_id", columns="k", values="emb_idx").loc[casc["cascade_id"]].to_numpy().astype(int).T
    HN = (long[long["k"] > 0].pivot(index="cascade_id", columns="k", values="hop_name")
          .loc[casc["cascade_id"]].to_numpy().T)

    strata = build_strata(casc, cfg.stratify)
    all_rows, drops_all, kstar_all, dpi_all, gauss_all, noise_all = [], [], {}, {}, {}, {}
    for sname, sidx in strata.items():
        n_src = casc.iloc[sidx]["source_record_id"].nunique()
        if n_src < cfg.min_sources_per_stratum:
            log(f"[skip] stratum {sname}: only {n_src} sources"); continue
        log(f"\n=== stratum {sname} ({len(sidx)} cascades, {n_src} sources) ===")
        noise = noise_stats_for(E, IDX, sidx, casc.iloc[sidx]["source_record_id"].to_numpy())
        noise_all[sname] = noise
        for sig in cfg.signals:
            res = run_signal(casc, IDX, E, sidx, sig, cfg, rng)
            if res is None:
                log(f"  [skip] signal={sig}: degenerate (single class / too few samples / H(S) < {cfg.min_entropy})")
                continue
            rows, ks_out, dpi, drops, mi, lo, hi = summarize(res, HN, sname, cfg, rng, E, IDX)
            all_rows += rows; drops_all += drops
            kstar_all.setdefault(sname, {})[sig] = ks_out
            dpi_all.setdefault(sname, {})[sig] = dpi
            P = between_class_var(E[IDX[0, res["pos"]]], res["y"])
            ga = gaussian_analysis(np.arange(K1), mi, lo, hi, res["H"], P, noise["sigma_n2_slope"], cfg.delta)
            gauss_all.setdefault(sname, {})[sig] = ga
            plot_gaussian(np.arange(K1), mi, lo, hi, res["H"], ga,
                          out / "figures" / f"gaussian__{sname.replace('|', '_')}__{sig}.png", f"{sig} | {sname}")
            ka = ks_out["k_star_abs"]["value"]
            log(f"  signal={sig:<11} M={res['M']:<5} H={res['H']:.2f}b  est={res['method']:<8} "
                f"MI: " + " ".join(f"{v:.2f}" for v in mi) + f" | acc: " + " ".join(f"{r['accuracy']:.2f}" for r in rows) +
                f" | k*={ka}")

    curves = pd.DataFrame(all_rows)
    curves.to_csv(out / "decay_curves.csv", index=False)
    pd.DataFrame(drops_all).to_csv(out / "per_transform_drop.csv", index=False)
    json.dump(noise_all, open(out / "noise_estimates.json", "w"), indent=2)
    json.dump(kstar_all, open(out / "k_star.json", "w"), indent=2)
    json.dump(gauss_all, open(out / "gaussian_fit.json", "w"), indent=2)

    # validation summary (Step 9)
    val = {}
    for s, d in dpi_all.items():
        for sig, dp in d.items():
            ga = gauss_all[s][sig]
            pm = ga.get("predicted_measured_noise")
            val.setdefault(s, {})[sig] = dict(
                dpi_n_increases=dp["n_increases"], dpi_max_increase_bits=dp["max_increase_bits"],
                dpi_significant_violation_hops=dp["significant_violations"],
                gaussian_verdict=(pm or {}).get("verdict"), gaussian_R2=(pm or {}).get("R2"),
                k_star_observed=kstar_all[s][sig]["k_star_abs"]["value"],
                k_star_theory_measured_noise=ga["k_star_theory_measured_noise"],
                k_star_theory_fitted=ga["k_star_theory_fitted"],
                hops_where_fano_bound_is_tighter_than_ce=[int(h) for h in curves[(curves.stratum == s) & (curves.signal == sig)]
                                                          .query("mi_fano_bits > mi_bits + 0.05 * H_S_bits")["hop"]],
                gaussian_form_usable=bool(pm is not None and pm["R2"] is not None and pm["R2"] > 0))
    json.dump(val, open(out / "validation.json", "w"), indent=2)

    # figures: metric vs hop; per signal across strata, and all signals in stratum 'all'
    if len(curves):
        metrics = [("mi_bits", "mi_ci_lo", "mi_ci_hi", "I(S;Y_k) [bits]"),
                   ("retention", "retention_ci_lo", "retention_ci_hi", "retention $R_k$"),
                   ("accuracy", "acc_ci_lo", "acc_ci_hi", "accuracy")]
        for m, lo_c, hi_c, yl in metrics:
            base = curves[curves.stratum == "all"]
            if len(base):
                plot_metric(base, m, lo_c, hi_c, yl, out / "figures" / f"{m}__all__by_signal.png", "signal", f"{yl}: all records")
            for sig, g in curves.groupby("signal"):
                if g["stratum"].nunique() > 1:
                    def chance(ax, g=g, m=m):
                        if m == "accuracy":
                            ax.axhline(g["chance_accuracy"].mean(), color="gray", ls=":")
                    plot_metric(g, m, lo_c, hi_c, yl, out / "figures" / f"{m}__{sig}__by_stratum.png", "stratum",
                                f"{yl}: signal={sig}", extra=chance)
    log(f"\nSaved results to {out}/ (decay_curves.csv, noise_estimates.json, k_star.json, gaussian_fit.json, "
        f"validation.json, per_transform_drop.csv, figures/)")


if __name__ == "__main__":
    main()
