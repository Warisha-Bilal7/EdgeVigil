# EdgeVigil

**Domain-adversarial anomaly detection for on-premise infrastructure, with local multi-agent root-cause diagnosis.**

No telemetry leaves the network. No cloud inference. No black-box thresholds.

## Table of Contents
- [The Problem](#the-problem)
- [What EdgeVigil Does](#what-edgevigil-does)
- [Why This Is Different](#why-this-is-different)
- [Architecture](#architecture)
- [Core ML: Domain-Adversarial Anomaly Detection](#core-ml-domain-adversarial-anomaly-detection)
- [Diagnostic Agent Layer](#diagnostic-agent-layer)
- [Tech Stack](#tech-stack)
- [Datasets & Evaluation](#datasets--evaluation)
- [Project Structure](#project-structure)
- [Build Roadmap](#build-roadmap)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Status](#status)
- [License](#license)

## The Problem

Banks, hospitals, telecoms, and government networks run fleets of on-prem servers, workstations, and IoT sensors that can't send telemetry to a cloud AIOps vendor — data sovereignty regulation (SBP, PTA, HIPAA-equivalent frameworks) rules it out. So these orgs fall back to static threshold monitoring: alert if CPU > 90%, alert if a heartbeat is missed. That catches failures after they've started, not before, and it can't tell a real anomaly from normal load variance per device type.

EdgeVigil predicts failures before they happen, entirely on-premise, by learning what "normal" looks like per device instead of relying on fixed thresholds.

## What EdgeVigil Does

- Continuously ingests telemetry (CPU, memory, disk I/O, network latency, temperature) from heterogeneous endpoints — servers, workstations, IoT sensors
- Learns a normal-behavior baseline per device type using an unsupervised reconstruction model, so it doesn't need labeled failure data (which barely exists in practice)
- Flags anomalies as early-warning signals, with a measurable lead time before actual failure
- On anomaly, triggers a local multi-agent pipeline that correlates logs/metrics, generates a root-cause hypothesis against a runbook knowledge base, and writes a human-readable incident report — without a single API call leaving the network

## Why This Is Different

| | Threshold tools (Zabbix, Nagios, Uptime Kuma) | Cloud AIOps (Datadog, Dynatrace) | SentinelEdge |
|---|---|---|---|
| Data leaves network | No | Yes | No |
| Adapts per device type | No | Partial | Yes (domain-adversarial) |
| Predicts before failure | No | Yes | Yes |
| Runs on CPU-only edge hardware | Yes | N/A (cloud) | Yes (quantized) |
| Root-cause diagnosis | Manual | Manual/limited | Automated, local |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Endpoint Fleet                         │
│   Servers      Workstations      IoT Sensors              │
└───────────────────────┬─────────────────────────────────┘
                         │ telemetry stream
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Anomaly Detection Core (quantized)              │
│   Shared Encoder ── GRL ── Domain Classifier (aux)         │
│        │                                                    │
│   Reconstruction Head ── Anomaly Score                    │
└───────────────────────┬─────────────────────────────────┘
                         │ anomaly event
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Diagnostic Agent Layer (LangGraph)              │
│   Correlator ──▶ Diagnostician (local RAG) ──▶ Reporter   │
└───────────────────────┬─────────────────────────────────┘
                         │
                         ▼
                 Dashboard / Alert / Incident Report
```

## Core ML: Domain-Adversarial Anomaly Detection

Different device types have fundamentally different "normal" telemetry distributions — a server's baseline CPU pattern looks nothing like an IoT sensor's. Training a separate model per device type doesn't scale and doesn't generalize to new device types added later. Training one model on pooled data without correction lets device-type identity leak into the "normal" representation, which causes the model to flag legitimate device-type variance as anomalies.

**Approach:**
- Shared encoder ingests windowed multivariate telemetry per device (sliding window, ~5-10 min)
- A gradient reversal layer (GRL) feeds a device-type domain classifier, forcing the encoder to learn device-type-invariant representations of "normal" — the same technique used in HalluProbe's domain-adversarial hallucination detection, applied here to device fleets instead of text domains
- A reconstruction head (LSTM-autoencoder or TCN) computes reconstruction error as the anomaly score — high error means "this doesn't look like normal behavior for any device type," not just "this device type is different"
- Quantize the trained model to INT8/NF4 post-training so it runs inference on CPU-only on-prem hardware with no GPU dependency

This means the model generalizes to new device types added to the fleet later without retraining from scratch — the practical failure mode that kills most per-device threshold systems.

## Diagnostic Agent Layer

When the anomaly core flags a device, a LangGraph supervisor routes to:

1. **Correlator agent** — pulls the relevant metrics and log window around the anomaly timestamp, structures it for downstream reasoning
2. **Diagnostician agent** — retrieves similar past incidents and runbook entries from a local vector store (no cloud embedding API), generates a root-cause hypothesis grounded in retrieved context
3. **Reporter agent** — writes a human-readable incident summary with recommended next steps

Statistics and correlation are computed in Python before any LLM call — the LLM's job is natural-language synthesis of pre-computed structured results, not raw number-crunching. This keeps the agent layer fast, deterministic where it needs to be, and cheap to run on local hardware.

**Important constraint:** for the "no data leaves network" claim to hold in production, the LLM backend and vector store both need to be local — Ollama or llama.cpp serving a small local model (not a cloud API like Groq), and a local vector index (FAISS, Chroma, or local Postgres+pgvector) rather than a hosted service. Cloud APIs are fine for early prototyping speed, but the production deployment story depends on swapping them out.

## Tech Stack

- **Anomaly core:** PyTorch (LSTM-autoencoder / TCN + GRL domain classifier), bitsandbytes for NF4/INT8 quantization
- **Serving:** FastAPI, async telemetry ingestion
- **Agent layer:** LangGraph, local LLM via Ollama (production) / Groq (prototyping)
- **Vector store:** FAISS or local Postgres + pgvector (no hosted/cloud instance)
- **Dashboard:** lightweight frontend (Next.js or plain HTML/Chart.js) — not the focus of the technical story, keep it minimal
- **Telemetry collection:** lightweight Python/Go agent on each endpoint, pushes to ingestion API

## Datasets & Evaluation

Real labeled enterprise failure data is scarce by nature — that's the whole reason thresholding dominates this space. Evaluation plan:

- **Server Machine Dataset (SMD)** and **NAB (Numenta Anomaly Benchmark)** for baseline time-series anomaly detection validation
- **Synthetic fleet simulator** — generate telemetry for simulated servers/workstations/IoT sensors with injected failure patterns (gradual drift, sudden spike, slow leak) to test domain-adversarial generalization across device types
- **Metrics to report:**
  - F1 on injected failure detection
  - False positive rate vs. a Nagios-style static threshold baseline (this comparison is the headline number)
  - Detection lead time — minutes of early warning before actual failure, averaged across failure types
  - Generalization gap — F1 on a held-out device type never seen during training, with vs. without the domain-adversarial component

## Project Structure

```
edgevigil/
├── core/
│   ├── data/              # telemetry simulators, SMD/NAB loaders
│   ├── models/             # encoder, GRL, domain classifier, reconstruction head
│   ├── train.py
│   ├── quantize.py
│   └── eval.py
├── agents/
│   ├── correlator.py
│   ├── diagnostician.py
│   ├── reporter.py
│   └── graph.py             # LangGraph supervisor/router
├── api/
│   ├── main.py              # FastAPI app
│   └── ingestion.py
├── dashboard/
├── runbooks/                 # local knowledge base for RAG
├── tests/
├── .env.example
└── README.md
```

## Build Roadmap

**Phase 1 — Data foundation**
Telemetry simulator for 3 device types (server, workstation, IoT sensor), failure injection framework, SMD/NAB loaders.

**Phase 2 — Baseline anomaly model**
Per-device-type LSTM-autoencoder, no domain adaptation yet. Establish baseline F1/FPR — this is the number everything else needs to beat.

**Phase 3 — Domain-adversarial training**
Add shared encoder + GRL + domain classifier. Compare generalization on held-out device type vs. Phase 2 baseline. This comparison is the core technical result of the project.

**Phase 4 — Quantization & edge benchmarking**
INT8/NF4 quantize the trained model, benchmark inference latency and memory on CPU-only hardware.

**Phase 5 — Diagnostic agent layer**
LangGraph supervisor, Correlator/Diagnostician/Reporter agents, local vector store + runbook RAG.

**Phase 6 — API & dashboard**
FastAPI ingestion + alerting endpoints, minimal dashboard for live status and incident reports.

## Getting Started

```bash
git clone https://github.com/yourusername/sentineledge.git
cd sentineledge
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt --break-system-packages
cp .env.example .env
```

Run the telemetry simulator and start training the Phase 2 baseline:

```bash
python core/data/simulate.py
python core/train.py --phase baseline
```

## Configuration

- `DEVICE_TYPES` — list of simulated/real device types to monitor
- `WINDOW_SIZE` — telemetry window length for the encoder (minutes)
- `LLM_BACKEND` — `ollama` (local, production) or `groq` (prototyping only)
- `VECTOR_STORE` — `faiss`, `chroma`, or local Postgres connection string

## Status

Phase 1 in progress. This README defines the target architecture; implementation details may shift as the domain-adversarial training results come in.

## License

MIT