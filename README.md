# Provenance Decay

**Fundamental Limits of Traceability in Multi-Hop Generative AI for Health Communication**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

---

## Overview

This repository accompanies the paper:

> **Provenance Decay: Fundamental Limits of Traceability in Multi-Hop Generative AI for Health Communication**

Generative AI has made it possible for health content to be summarized, translated, paraphrased, and re-styled by different large language models (LLMs) within hours. Each of these operations is a **generative hop**. At each hop, the ability to trace content back to its source—its **provenance**—degrades.

This project provides:

- A formal **channel model** for provenance under multi-hop generative transformations.
- A **data processing inequality** for provenance, proving that traceability cannot increase across hops.
- A **Gaussian hop decay bound** quantifying how fast provenance is lost.
- A **maximum detectable hops theorem** via Fano's inequality.
- An **optimal Bayesian governance threshold** for escalation, audit, and abstention decisions.
- A **simulation framework** and empirical illustration on social media health texts.

The goal is not to propose a new watermarking algorithm or detector, but to derive the **fundamental limits** that any provenance method must respect—just as Shannon's channel capacity defines the limits of communication regardless of the code used.

---

## Key Concepts

- **Generative hop**: One stochastic transformation of content by an LLM, such as summarization, translation, paraphrasing, or style transfer.
- **Provenance signal** \(S\): A latent variable representing source identity, claim origin, or credibility.
- **Provenance decay**: The monotonic loss of mutual information \(I(S; Y_k)\) across successive generative hops.
- **Maximum detectable hops** \(k^*\): The critical hop count beyond which no detector can reliably recover provenance.
- **Optimal governance threshold**: The Bayesian decision boundary that minimizes expected harm subject to review costs.

---

## Research Questions

1. What is the maximum number of generative hops a provenance signal can survive before detection becomes information-theoretically impossible?
2. How do different transformation types—summarization, paraphrasing, translation, and style transfer—affect the rate of provenance decay?
3. What is the optimal governance intervention point that minimizes expected harm subject to review costs?
4. How do these theoretical bounds manifest empirically on real social media health texts?

---

## Repository Structure

```text
provenance-decay/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── src/
│   ├── data/
│   │   ├── build_corpus.py
│   │   ├── preprocess.py
│   │   └── provenance_signal.py
│   ├── channels/
│   │   ├── channel_model.py
│   │   ├── noise_estimation.py
│   │   └── transformations.py
│   ├── theory/
│   │   ├── dpi.py
│   │   ├── gaussian_decay.py
│   │   ├── fano_bound.py
│   │   └── bayesian_governance.py
│   ├── simulation/
│   │   ├── cascade.py
│   │   ├── decay_curves.py
│   │   └── governance_policy.py
│   └── utils/
│       ├── metrics.py
│       └── plotting.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_channel_estimation.ipynb
│   ├── 03_provenance_decay.ipynb
│   └── 04_governance_threshold.ipynb
├── experiments/
│   ├── configs/
│   └── results/
└── paper/
    ├── main.tex
    └── figures/
