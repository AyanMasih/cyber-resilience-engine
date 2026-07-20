"""
ttp_mapping.py
---------------------------------
Maps behavioral risk-engine output (anomaly_detector.py) to MITRE ATT&CK
technique(s) and generates a triage playbook per high-risk (entity, day).

Two-layer design, mirroring the graceful-degradation philosophy already
used for the LSTM/torch fallback in anomaly_detector.py:

1. Rule-based feature -> ATT&CK lookup (deterministic, always available,
   zero external dependencies). Looks at WHICH raw features drove the
   anomaly (via per-entity-type z-scores) and maps those features to a
   short list of candidate techniques, covering both Enterprise ATT&CK
   and ATT&CK for ICS (since ot_write_command_count implies an OT/ICS
   asset population alongside IT hosts).

2. LLM enrichment (optional - requires `anthropic` package + an
   ANTHROPIC_API_KEY env var). Feeds the rule-based candidates plus the
   entity's actual feature deltas to Claude, which picks the most likely
   technique(s), grounds the rationale in the specific numbers, and drafts
   a step-by-step triage playbook. If the package/key is missing or a
   call fails, the system logs a warning and falls back to the rule-based
   mapping only - it never crashes the pipeline.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from anomaly_detector import FEATURE_NAMES

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Layer 1: deterministic feature -> ATT&CK candidate lookup
# ---------------------------------------------------------------------------
FEATURE_TTP_MAP: dict[str, list[tuple[str, str, str, str]]] = {
    "failed_login_count": [
        ("T1110", "Brute Force", "Credential Access", "enterprise"),
        ("T1078", "Valid Accounts", "Initial Access / Persistence", "enterprise"),
    ],
    "login_count": [
        ("T1078", "Valid Accounts", "Initial Access / Persistence", "enterprise"),
        ("T1021", "Remote Services", "Lateral Movement", "enterprise"),
    ],
    "off_hours_ratio": [
        ("T1078", "Valid Accounts", "Defense Evasion", "enterprise"),
        ("T1499", "Endpoint Denial of Service (staging)", "Impact", "enterprise"),
    ],
    "bytes_out": [
        ("T1041", "Exfiltration Over C2 Channel", "Exfiltration", "enterprise"),
        ("T1567", "Exfiltration Over Web Service", "Exfiltration", "enterprise"),
    ],
    "bytes_in": [
        ("T1105", "Ingress Tool Transfer", "Command and Control", "enterprise"),
    ],
    "unique_dst_count": [
        ("T1046", "Network Service Discovery", "Discovery", "enterprise"),
        ("T1021", "Remote Services", "Lateral Movement", "enterprise"),
    ],
    "new_process_count": [
        ("T1059", "Command and Scripting Interpreter", "Execution", "enterprise"),
        ("T1204", "User Execution", "Execution", "enterprise"),
    ],
    "privilege_change_count": [
        ("T1548", "Abuse Elevation Control Mechanism", "Privilege Escalation", "enterprise"),
        ("T1136", "Create Account", "Persistence", "enterprise"),
    ],
    "ot_write_command_count": [
        ("T0831", "Manipulation of Control", "Impair Process Control", "ics"),
        ("T0836", "Modify Parameter", "Impair Process Control", "ics"),
        ("T0855", "Unauthorized Command Message", "Inhibit Response Function", "ics"),
    ],
    "session_duration_std": [
        ("T1078", "Valid Accounts", "Defense Evasion", "enterprise"),
    ],
}


@dataclass
class ContributingFeature:
    name: str
    value: float
    baseline_mean: float
    z_score: float


@dataclass
class EntityDayFinding:
    entity_id: str
    entity_type: str
    day: object
    risk_score: float
    if_score: float
    drift_score: float
    contributing_features: list
    candidate_ttps: list
    llm_enrichment: Optional[dict] = None


class RuleBasedTTPMapper:
    """
    Computes per-entity-type z-scores for each feature and shortlists
    ATT&CK candidates from the top-|z| contributing features.
    """

    def __init__(self, top_k_features: int = 3):
        self.top_k_features = top_k_features
        self._baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame):
        for etype, group in df.groupby("entity_type"):
            X = group[FEATURE_NAMES].values.astype(float)
            self._baselines[etype] = (X.mean(axis=0), X.std(axis=0) + 1e-9)
        return self

    def contributing_features(self, row: pd.Series) -> list[ContributingFeature]:
        etype = row["entity_type"]
        if etype not in self._baselines:
            return []
        mean, std = self._baselines[etype]
        x = row[FEATURE_NAMES].values.astype(float)
        z = (x - mean) / std
        order = np.argsort(-np.abs(z))[: self.top_k_features]
        return [
            ContributingFeature(
                name=FEATURE_NAMES[i],
                value=float(x[i]),
                baseline_mean=float(mean[i]),
                z_score=float(z[i]),
            )
            for i in order
        ]

    def candidate_ttps(self, features: list[ContributingFeature]) -> list[dict]:
        seen = {}
        for f in features:
            for tid, name, tactic, matrix in FEATURE_TTP_MAP.get(f.name, []):
                seen.setdefault(
                    tid,
                    {
                        "technique_id": tid,
                        "technique_name": name,
                        "tactic": tactic,
                        "matrix": matrix,
                        "triggered_by": f.name,
                    },
                )
        return list(seen.values())


# ---------------------------------------------------------------------------
# Layer 2: optional LLM enrichment
# ---------------------------------------------------------------------------
_ENRICHMENT_SYSTEM_PROMPT = """You are a SOC Tier-2 triage assistant mapping \
behavioral anomalies to MITRE ATT&CK techniques and drafting a first-response \
playbook. You are given: entity metadata, a risk score already computed by an \
upstream ML pipeline (IsolationForest + LSTM-autoencoder drift), the specific \
features that deviated most from that entity's baseline, and a rule-based \
shortlist of candidate ATT&CK techniques (Enterprise and/or ICS matrix).

Respond with ONLY a JSON object (no markdown fences, no preamble) matching \
this schema:
{
  "techniques": [
    {"technique_id": str, "technique_name": str, "tactic": str,
     "matrix": "enterprise" | "ics", "confidence": "low"|"medium"|"high",
     "rationale": str}
  ],
  "kill_chain_narrative": str,
  "severity": "low" | "medium" | "high" | "critical",
  "playbook": [
    {"step": int, "action": str, "owner": str, "priority": "immediate"|"high"|"normal"}
  ]
}

Ground every rationale in the actual numbers you were given. Prefer the \
provided candidate techniques when they fit; you may add or drop one if the \
feature pattern clearly points elsewhere. Keep the playbook to 4-7 concrete, \
entity-type-appropriate steps (containment/isolation steps for OT entities \
must never include actions that would interrupt safe process operation \
without human sign-off)."""


class LLMEnricher:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 1200):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            self._client = anthropic.Anthropic()

    @property
    def available(self) -> bool:
        return self._client is not None

    def enrich(self, finding: EntityDayFinding) -> Optional[dict]:
        if not self.available:
            return None

        user_payload = {
            "entity_id": finding.entity_id,
            "entity_type": finding.entity_type,
            "day": finding.day,
            "risk_score": round(finding.risk_score, 3),
            "isolation_forest_score": round(finding.if_score, 3),
            "temporal_drift_score": round(finding.drift_score, 3),
            "contributing_features": [asdict(f) for f in finding.contributing_features],
            "candidate_ttps": finding.candidate_ttps,
        }

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_ENRICHMENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(user_payload)}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"LLM enrichment failed for {finding.entity_id}/{finding.day}: {exc}")
            return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
class TTPMappingEngine:
    """
    Ties the rule-based mapper and (optional) LLM enricher together.
    """

    def __init__(self, risk_threshold: float = 0.6, top_k_features: int = 3,
                 use_llm: bool = True, model: str = "claude-3-5-sonnet-20241022"):
        self.risk_threshold = risk_threshold
        self.rule_mapper = RuleBasedTTPMapper(top_k_features=top_k_features)
        self.enricher = LLMEnricher(model=model) if use_llm else None

        if use_llm and not ANTHROPIC_AVAILABLE:
            warnings.warn(
                "anthropic package not installed - falling back to rule-based TTP mapping."
            )
        elif use_llm and ANTHROPIC_AVAILABLE and not os.environ.get("ANTHROPIC_API_KEY"):
            warnings.warn(
                "ANTHROPIC_API_KEY not set - falling back to rule-based TTP mapping."
            )

    def run(self, raw_df: pd.DataFrame, result_df: pd.DataFrame) -> list[EntityDayFinding]:
        self.rule_mapper.fit(raw_df)

        merged = result_df[result_df["risk_score"] >= self.risk_threshold].merge(
            raw_df[["entity_id", "day"] + FEATURE_NAMES],
            on=["entity_id", "day"],
            how="left",
        )

        findings: list[EntityDayFinding] = []
        for _, row in merged.iterrows():
            features = self.rule_mapper.contributing_features(row)
            candidates = self.rule_mapper.candidate_ttps(features)

            finding = EntityDayFinding(
                entity_id=row["entity_id"],
                entity_type=row["entity_type"],
                day=row["day"],
                risk_score=float(row["risk_score"]),
                if_score=float(row["if_score"]),
                drift_score=float(row["drift_score"]),
                contributing_features=features,
                candidate_ttps=candidates,
            )

            if self.enricher is not None and self.enricher.available:
                finding.llm_enrichment = self.enricher.enrich(finding)

            findings.append(finding)

        findings.sort(key=lambda f: f.risk_score, reverse=True)
        return findings

    @staticmethod
    def to_json_records(findings: list[EntityDayFinding]) -> list[dict]:
        records = []
        for f in findings:
            rec = asdict(f)
            rec["contributing_features"] = [asdict(cf) for cf in f.contributing_features]
            records.append(rec)
        return records

    def save(self, findings: list[EntityDayFinding], path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_json_records(findings), fh, indent=2, default=str)


if __name__ == "__main__":
    import pathlib

    module_dir = pathlib.Path(__file__).parent
    data_dir = module_dir.parent / "data"

    # Fallback to current directory if files exist locally
    raw_path = data_dir / "entity_events.csv" if (data_dir / "entity_events.csv").exists() else module_dir / "entity_events.csv"
    risk_path = module_dir / "risk_scores.csv"

    if not raw_path.exists() or not risk_path.exists():
        print("❌ Error: Missing required input files (`entity_events.csv` or `risk_scores.csv`).")
        print("Run `python anomaly_detector.py` first to generate them!")
        exit(1)

    raw_df = pd.read_csv(raw_path)
    result_df = pd.read_csv(risk_path)

    engine = TTPMappingEngine(risk_threshold=0.6, use_llm=True)
    findings = engine.run(raw_df, result_df)
    engine.save(findings, module_dir / "ttp_mappings.json")

    print("\n" + "=" * 60)
    print("🚀 CYBER RESILIENCE ENGINE - TTP MAPPER RUN COMPLETE")
    print("=" * 60)
    print(f"Mapped {len(findings)} high-risk entity-days to MITRE ATT&CK techniques.")
    print(f"LLM Enrichment Status: {'✅ Active' if engine.enricher and engine.enricher.available else '⚠️ Fallback (Rule-Based Only)'}\n")

    # DEMO DISPLAY
    if findings:
        top_f = findings[0]
        print("------------------------------------------------------------")
        print(f"🔥 DEMO HIGHLIGHT: Top Flagged Threat [{top_f.entity_id}]")
        print("------------------------------------------------------------")
        print(f"• Entity Type : {top_f.entity_type}")
        print(f"• Day         : {top_f.day}")
        print(f"• Risk Score  : {top_f.risk_score:.3f} (IF: {top_f.if_score:.2f} | Drift: {top_f.drift_score:.2f})")
        
        print("\n🔍 Candidate MITRE ATT&CK Techniques:")
        for c in top_f.candidate_ttps:
            print(f"  - [{c['matrix'].upper()}] {c['technique_id']} - {c['technique_name']} (Trigger: {c['triggered_by']})")

        if top_f.llm_enrichment:
            enrichment = top_f.llm_enrichment
            print(f"\n🧠 LLM Enricher Analysis:")
            print(f"• Severity   : {enrichment.get('severity', 'N/A').upper()}")
            print(f"• Narrative  : {enrichment.get('kill_chain_narrative', 'N/A')}")
            
            print("\n📋 Automated Triage Playbook:")
            for step in enrichment.get("playbook", []):
                print(f"  Step {step.get('step')}: [{step.get('priority').upper()}] {step.get('action')} (Owner: {step.get('owner')})")
        print("=" * 60 + "\n")
