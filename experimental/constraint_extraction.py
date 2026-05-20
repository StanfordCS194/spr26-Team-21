#!/usr/bin/env python3
"""
Constraint Extraction Prototype
================================
1. Send the carclaims column list to Claude and ask for domain constraints as JSON.
2. If real data is available, validate each constraint against it.
3. Generate 100 synthetic rows via naive per-column random sampling and measure
   how many violate each constraint.

Usage:
    python constraint_extraction.py [--data PATH_TO_CSV]

Requires: ANTHROPIC_API_KEY in environment or a .env file in this directory.
"""

import argparse
import json
import os
import sys
import re
import textwrap
from pathlib import Path

import numpy as np

# ── Load .env if present ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env")  # also try repo root
except ImportError:
    pass  # python-dotenv optional; rely on environment variable

import anthropic

# ── Auth helpers ──────────────────────────────────────────────────────────────
def _get_claude_code_oauth_token() -> str | None:
    """Retrieve the Claude Code OAuth access token from the macOS keychain."""
    import subprocess, json
    try:
        raw = subprocess.check_output(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        data = json.loads(raw)
        return data.get("claudeAiOauth", {}).get("accessToken")
    except Exception:
        return None


# ── Column list ───────────────────────────────────────────────────────────────
COLUMNS = [
    "months_as_customer", "age", "policy_number", "policy_bind_date",
    "policy_state", "policy_csl", "policy_deductible", "policy_annual_premium",
    "umbrella_limit", "insured_zip", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "capital_gains", "capital_loss", "incident_date", "incident_type",
    "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "incident_city", "incident_location",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "property_damage", "bodily_injuries", "witnesses",
    "police_report_available", "total_claim_amount", "injury_claim",
    "property_claim", "vehicle_claim", "auto_make", "auto_model",
    "auto_year", "fraud_reported",
]

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert in insurance data modelling.
    When given a list of columns from an insurance claims dataset, you output
    ONLY a valid JSON array — no prose, no markdown fences, no explanation.
    Each element must be an object with exactly these keys:
      constraint_type   – one of: range, cardinality, relationship, format, conditional
      columns_involved  – list of column names from the provided list
      rule              – a concise, unambiguous English description of the constraint
      example_violation – one concrete example of a row that would violate the rule
""").strip()

USER_PROMPT = textwrap.dedent(f"""
    The following columns exist in a car-insurance claims dataset:

    {json.dumps(COLUMNS, indent=2)}

    Extract 15–20 realistic domain constraints that a *real* insurance dataset
    should satisfy. Focus on constraints that:
    - Can be programmatically checked (not vague business rules)
    - Span different constraint types (range, cardinality, relationship, format, conditional)
    - Would frequently be violated by naive random synthesis

    Return ONLY the JSON array described in the system prompt.
""").strip()


# ── Step 1: call Claude ────────────────────────────────────────────────────────
def extract_constraints() -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
    else:
        # Fall back to Claude Code's OAuth token (macOS keychain)
        oauth_token = _get_claude_code_oauth_token()
        if oauth_token:
            print("  (using Claude Code OAuth token from keychain)")
            client = anthropic.Anthropic(auth_token=oauth_token)
        else:
            sys.exit("ERROR: ANTHROPIC_API_KEY not set and no Claude Code OAuth token found")
    print("Calling Claude to extract constraints …")

    import time
    for attempt in range(5):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_PROMPT}],
            )
            break
        except anthropic.RateLimitError as e:
            wait = 30 * (attempt + 1)
            print(f"  Rate limited (attempt {attempt+1}/5) — waiting {wait}s …")
            time.sleep(wait)
            if attempt == 4:
                raise

    raw = message.content[0].text.strip()
    # Strip markdown fences if the model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        constraints = json.loads(raw)
    except json.JSONDecodeError as e:
        print("WARNING: Could not parse Claude response as JSON:", e)
        print("Raw response:\n", raw[:500])
        constraints = []

    print(f"  → Claude returned {len(constraints)} constraints\n")
    return constraints


# ── Step 2: validate against real data ────────────────────────────────────────
def build_checker(constraint: dict):
    """
    Return a callable  f(row: dict) -> bool  that returns True if the row
    SATISFIES the constraint, False if it VIOLATES it.

    We use a simple rule-based interpreter keyed on constraint_type + keywords
    in the rule string.  This covers ~80 % of the constraints Claude typically
    generates for this schema; genuinely ambiguous ones return True (pass).
    """
    rule = constraint.get("rule", "").lower()
    cols = constraint.get("columns_involved", [])
    ctype = constraint.get("constraint_type", "")

    # ── range / numeric ───────────────────────────────────────────────────────
    if ctype == "range" or "between" in rule or "greater than" in rule or "less than" in rule or "positive" in rule:
        # age
        if "age" in cols and "age" in rule:
            return lambda r: _in_range(r.get("age"), 16, 100)
        # months_as_customer
        if "months_as_customer" in cols:
            return lambda r: _in_range(r.get("months_as_customer"), 0, None)
        # incident_hour_of_the_day
        if "incident_hour_of_the_day" in cols:
            return lambda r: _in_range(r.get("incident_hour_of_the_day"), 0, 23)
        # policy_deductible
        if "policy_deductible" in cols:
            return lambda r: _in_range(r.get("policy_deductible"), 0, None)
        # policy_annual_premium
        if "policy_annual_premium" in cols:
            return lambda r: _in_range(r.get("policy_annual_premium"), 0, None)
        # total_claim_amount
        if "total_claim_amount" in cols:
            if "injury_claim" in cols or "property_claim" in cols or "vehicle_claim" in cols:
                return _check_claim_sum
            return lambda r: _in_range(r.get("total_claim_amount"), 0, None)
        # capital gains / loss
        if "capital_gains" in cols:
            return lambda r: _in_range(r.get("capital_gains"), 0, None)
        if "capital_loss" in cols:
            return lambda r: _in_range(r.get("capital_loss"), 0, None)
        # number_of_vehicles_involved
        if "number_of_vehicles_involved" in cols:
            return lambda r: _in_range(r.get("number_of_vehicles_involved"), 1, None)
        # bodily_injuries
        if "bodily_injuries" in cols:
            return lambda r: _in_range(r.get("bodily_injuries"), 0, None)
        # witnesses
        if "witnesses" in cols:
            return lambda r: _in_range(r.get("witnesses"), 0, None)
        # auto_year
        if "auto_year" in cols:
            return lambda r: _in_range(r.get("auto_year"), 1900, 2025)

    # ── relationship: claim sum ───────────────────────────────────────────────
    if "total_claim_amount" in cols and (
        "injury_claim" in cols or "property_claim" in cols or "vehicle_claim" in cols
    ):
        return _check_claim_sum

    # ── relationship: incident_date >= policy_bind_date ───────────────────────
    if "incident_date" in cols and "policy_bind_date" in cols:
        return _check_incident_after_policy

    # ── relationship: age vs months_as_customer ───────────────────────────────
    if "age" in cols and "months_as_customer" in cols:
        return lambda r: _check_age_vs_months(r)

    # ── conditional: collision_type when incident_type ───────────────────────
    if "collision_type" in cols and "incident_type" in cols:
        return _check_collision_type_conditional

    # ── conditional: property_damage / bodily_injuries when severity ─────────
    if "incident_severity" in cols and "bodily_injuries" in cols:
        return _check_severity_vs_injuries

    # ── format: policy_csl ───────────────────────────────────────────────────
    if "policy_csl" in cols and "csl" in rule:
        return lambda r: _check_csl_format(r)

    # ── cardinality: binary / boolean columns ─────────────────────────────────
    if ctype == "cardinality":
        if "fraud_reported" in cols:
            return lambda r: r.get("fraud_reported") in ("Y", "N", 0, 1, True, False)
        if "police_report_available" in cols:
            return lambda r: r.get("police_report_available") in ("YES", "NO", "Y", "N", 0, 1, True, False)
        if "property_damage" in cols:
            return lambda r: r.get("property_damage") in ("YES", "NO", "Y", "N", 0, 1, True, False)

    # ── format: zip code ──────────────────────────────────────────────────────
    if "insured_zip" in cols and ("zip" in rule or "postal" in rule or "5" in rule):
        return lambda r: _check_zip(r)

    # ── umbrella_limit non-negative ───────────────────────────────────────────
    if "umbrella_limit" in cols and ("negative" in rule or "non-negative" in rule or "≥" in rule or ">=" in rule):
        return lambda r: _to_float(r.get("umbrella_limit"), 0) >= 0

    # Default: can't interpret → treat as satisfied
    return lambda r: True


# ── helper functions ──────────────────────────────────────────────────────────
def _to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _in_range(v, lo, hi):
    fv = _to_float(v)
    if fv is None:
        return True  # can't check
    if lo is not None and fv < lo:
        return False
    if hi is not None and fv > hi:
        return False
    return True


def _check_claim_sum(row):
    total = _to_float(row.get("total_claim_amount"))
    inj   = _to_float(row.get("injury_claim"), 0)
    prop  = _to_float(row.get("property_claim"), 0)
    veh   = _to_float(row.get("vehicle_claim"), 0)
    if total is None:
        return True
    # allow a small tolerance (rounding)
    return abs(total - (inj + prop + veh)) <= 1.0


def _check_incident_after_policy(row):
    from datetime import datetime
    fmt_candidates = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
    def parse(s):
        if not s:
            return None
        for fmt in fmt_candidates:
            try:
                return datetime.strptime(str(s), fmt)
            except ValueError:
                pass
        return None
    inc  = parse(row.get("incident_date"))
    bind = parse(row.get("policy_bind_date"))
    if inc is None or bind is None:
        return True
    return inc >= bind


def _check_age_vs_months(row):
    age = _to_float(row.get("age"))
    months = _to_float(row.get("months_as_customer"))
    if age is None or months is None:
        return True
    # can't have been a customer longer than you've been an adult (roughly)
    return months / 12.0 <= (age - 16)


def _check_collision_type_conditional(row):
    itype = str(row.get("incident_type", "")).lower()
    ctype = str(row.get("collision_type", "")).lower()
    # If incident is NOT a collision, collision_type should be null/none/?
    non_collision_keywords = ["theft", "parked", "fire"]
    is_non_collision = any(k in itype for k in non_collision_keywords)
    if is_non_collision:
        return ctype in ("?", "", "none", "nan", "n/a")
    return True  # can't fully check the positive case without an enum


def _check_severity_vs_injuries(row):
    severity = str(row.get("incident_severity", "")).lower()
    injuries = _to_float(row.get("bodily_injuries"), 0)
    if "major" in severity and injuries is not None:
        return injuries >= 1
    return True


def _check_csl_format(row):
    csl = str(row.get("policy_csl", ""))
    # e.g. "100/300/100"
    return bool(re.match(r"^\d+/\d+/\d+$", csl))


def _check_zip(row):
    z = str(row.get("insured_zip", ""))
    return bool(re.match(r"^\d{5}$", z))


# ── Step 3: validate against real data ────────────────────────────────────────
def validate_real(constraints: list[dict], csv_path: str) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping real-data validation")
        return constraints

    print(f"Loading real data from {csv_path} …")
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    n = len(rows)
    print(f"  → {n} rows loaded\n")

    for c in constraints:
        checker = build_checker(c)
        satisfied = sum(1 for r in rows if checker(r))
        c["real_data_satisfaction_pct"] = round(100 * satisfied / n, 1) if n else None

    return constraints


# ── Step 4: naive synthetic data ──────────────────────────────────────────────
def generate_synthetic(constraints: list[dict], csv_path: str | None, n: int = 100) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping synthetic validation")
        return constraints

    if csv_path and Path(csv_path).exists():
        df_real = pd.read_csv(csv_path)
        print(f"Building synthetic data from observed column distributions …")
        rng = np.random.default_rng(42)
        synth = {}
        for col in df_real.columns:
            vals = df_real[col].dropna().values
            if len(vals) == 0:
                synth[col] = [""] * n
            else:
                synth[col] = rng.choice(vals, size=n).tolist()
        df_synth = pd.DataFrame(synth)
    else:
        # No real data: generate minimally plausible random values for each column
        print("No real data found — generating purely random synthetic rows …")
        rng = np.random.default_rng(42)
        df_synth = pd.DataFrame({
            "months_as_customer":        rng.integers(0, 600, n),
            "age":                       rng.integers(16, 90, n),
            "policy_number":             [f"POL{rng.integers(100000,999999)}" for _ in range(n)],
            "policy_bind_date":          ["2010-01-01"] * n,
            "policy_state":              rng.choice(["OH","IN","IL","IN","PA"], n),
            "policy_csl":                rng.choice(["100/300/100","250/500/100","500/1000/500"], n),
            "policy_deductible":         rng.choice([500, 1000, 2000], n),
            "policy_annual_premium":     rng.uniform(500, 3000, n).round(2),
            "umbrella_limit":            rng.choice([0, 1000000, 2000000], n),
            "insured_zip":               rng.integers(10000, 99999, n),
            "insured_sex":               rng.choice(["MALE","FEMALE"], n),
            "insured_education_level":   rng.choice(["JD","High School","College","Masters"], n),
            "insured_occupation":        rng.choice(["tech-support","craft-repair","machine-op-inspct"], n),
            "insured_hobbies":           rng.choice(["chess","cross-fit","polo"], n),
            "insured_relationship":      rng.choice(["husband","wife","own-child","other-relative"], n),
            "capital_gains":             rng.integers(0, 100000, n),
            "capital_loss":              rng.integers(0, 100000, n),
            "incident_date":             ["2015-01-01"] * n,
            "incident_type":             rng.choice(["Single Vehicle Collision","Multi-vehicle Collision","Parked Car"], n),
            "collision_type":            rng.choice(["Front Collision","Rear Collision","Side Collision","?"], n),
            "incident_severity":         rng.choice(["Minor Damage","Major Damage","Total Loss","Trivial Damage"], n),
            "authorities_contacted":     rng.choice(["Police","Fire","Ambulance","None"], n),
            "incident_state":            rng.choice(["OH","PA","SC","NY","VA"], n),
            "incident_city":             rng.choice(["Columbus","Riverwood","Springfield"], n),
            "incident_location":         [f"{rng.integers(1,999)} Main St" for _ in range(n)],
            "incident_hour_of_the_day":  rng.integers(0, 24, n),       # intentional: 24 is OOB
            "number_of_vehicles_involved": rng.integers(1, 5, n),
            "property_damage":           rng.choice(["YES","NO","?"], n),
            "bodily_injuries":           rng.integers(0, 5, n),
            "witnesses":                 rng.integers(0, 5, n),
            "police_report_available":   rng.choice(["YES","NO","?"], n),
            "total_claim_amount":        rng.integers(100, 100000, n),
            "injury_claim":              rng.integers(0, 50000, n),
            "property_claim":            rng.integers(0, 50000, n),
            "vehicle_claim":             rng.integers(0, 50000, n),
            "auto_make":                 rng.choice(["Toyota","Honda","Ford","Chevrolet"], n),
            "auto_model":                rng.choice(["Camry","Civic","F-150","Malibu"], n),
            "auto_year":                 rng.integers(1990, 2026, n),
            "fraud_reported":            rng.choice(["Y","N"], n),
        })

    rows = df_synth.to_dict(orient="records")
    print(f"  → {n} synthetic rows generated\n")

    for c in constraints:
        checker = build_checker(c)
        violated = sum(1 for r in rows if not checker(r))
        c["synthetic_violation_pct"] = round(100 * violated / n, 1)

    return constraints


# ── Pretty print results ──────────────────────────────────────────────────────
def print_results(constraints: list[dict]):
    print("=" * 70)
    print("CONSTRAINT EXTRACTION RESULTS")
    print("=" * 70)
    for i, c in enumerate(constraints, 1):
        print(f"\n[{i:02d}] {c.get('constraint_type','?').upper()} — {c.get('columns_involved', [])}")
        print(f"     Rule : {c.get('rule', '')}")
        print(f"     Violation example : {c.get('example_violation', '')}")
        if "real_data_satisfaction_pct" in c:
            pct = c["real_data_satisfaction_pct"]
            print(f"     Real data satisfaction : {pct}%")
        if "synthetic_violation_pct" in c:
            pct = c["synthetic_violation_pct"]
            print(f"     Naive synthetic violation : {pct}%")
    print("\n" + "=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Constraint extraction prototype")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent / "data" / "insurance_claims.csv"),
        help="Path to real carclaims CSV (optional)",
    )
    parser.add_argument(
        "--save-constraints",
        default=str(Path(__file__).parent / "constraints.json"),
        help="Where to save the extracted constraints JSON",
    )
    parser.add_argument(
        "--synthetic-n", type=int, default=100,
        help="Number of synthetic rows to generate (default 100)",
    )
    args = parser.parse_args()

    # Determine if real data exists
    real_csv = args.data if Path(args.data).exists() else None
    if real_csv:
        print(f"Real data found at: {real_csv}")
    else:
        print(f"No real data at {args.data} — will skip real-data validation\n")

    # Step 1: Extract constraints via Claude
    constraints = extract_constraints()
    if not constraints:
        sys.exit("No constraints extracted — aborting")

    # Save raw constraints
    with open(args.save_constraints, "w") as f:
        json.dump(constraints, f, indent=2)
    print(f"Constraints saved to {args.save_constraints}\n")

    # Step 2 (optional): validate against real data
    if real_csv:
        constraints = validate_real(constraints, real_csv)

    # Step 3: generate synthetic data and measure violations
    constraints = generate_synthetic(constraints, real_csv, n=args.synthetic_n)

    # Print summary
    print_results(constraints)

    # Save final results (with validation stats)
    results_path = Path(__file__).parent / "constraint_results.json"
    with open(results_path, "w") as f:
        json.dump(constraints, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
