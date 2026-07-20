# 🛡️ Machine Learning-Driven Cyber Resilience Engine

An enterprise-grade, hybrid ML + LLM cybersecurity pipeline designed to detect behavioral anomalies, map threat indicators to the **MITRE ATT&CK Framework** (Enterprise & ICS), and automatically generate grounded triage playbooks for Tier-2 SOC analysts.

---

## 🌟 Key Features & Architecture

Most automated threat detection tools rely solely on thin LLM wrappers that process raw, noisy log dumps. The **Cyber Resilience Engine** implements a **two-tier architecture** to optimize performance, accuracy, and operational cost:

+---------------------+      +------------------------+      +-------------------------+
| Raw Event Telemetry | ---> | ML Anomaly Detector    | ---> | High-Risk Filtering     |
| (entity_events.csv) |      | (IsolationForest+Drift)|      | (risk_score >= 0.60)    |
+---------------------+      +------------------------+      +-------------------------+
|
v
+---------------------+      +------------------------+      +-------------------------+
| Structured JSON     | <--- | Layer 2: LLM Enricher  | <--- | Layer 1: Rule TTP Map   |
| (ttp_mappings.json) |      | (Claude API + Playbook)|      | (Feature Z-Score Engine)|
+---------------------+      +------------------------+      +-------------------------+


### 1. 🤖 Behavioral Anomaly Detection (`anomaly_detector.py`)
* **Per-Entity IsolationForest:** Fits baseline models per entity type (`workstation`, `ics_device`, `user`) to identify multi-feature deviations.
* **Global Temporal Drift Scoring:** Computes population-wide feature drift to identify subtle, long-term state shifts.
* **Weighted Ensemble Risk Score:** Combines isolation and drift scores into a normalized `risk_score` (0.0 – 1.0).

### 2. 🎯 Graceful Fallback TTP Mapper (`ttp_mapping.py`)
* **Layer 1: Deterministic Z-Score Shortlisting:** Calculates key feature drivers ($\vert{}z\text{-score}\vert{}$) and maps them to candidate MITRE ATT&CK techniques across Enterprise and ICS matrices (e.g., `T1110` Brute Force, `T0831` Manipulation of Control).
* **Layer 2: LLM Enrichment & Triage:** Feeds feature deltas and rule candidates to Claude (`claude-3-5-sonnet-20241022`) to generate structured JSON outputs containing severity, kill-chain narratives, and 4–7 step triage playbooks.
* **ICS Safety Guardrails:** Strict prompt constraints prevent automated containment actions on Industrial Control Systems without human sign-off.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Install the required Python packages:

```bash
pip install pandas numpy scikit-learn anthropic
