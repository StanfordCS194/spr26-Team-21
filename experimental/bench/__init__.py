"""Trust Benchmark — reproducible multi-synthesizer evaluation harness.

Runs every Aperture pillar (utility / rule_packs / audit / privacy / detection)
across multiple synthesizers and datasets, producing the headline numbers from
slide 13 of the project pitch:
  * +Xpt rare-class recall lift
  * 100% business-rule compliance
  * 0 exact-match privacy violations

Layout
------
  datasets.py            loaders for fraud_oracle + pima_diabetes (downloads on first use)
  synthesizers.py        uniform wrapper around GaussianCopula/CTGAN/TVAE (and later TabDDPM)
  run_trust_benchmark.py orchestrator: loops (dataset × synthesizer × n_rows) and writes CSV
  figures.py             plot helpers (recall lift, privacy-utility frontier)
  results/               output CSVs and PNGs (gitignored intermediate runs;
                         canonical snapshots checked in)
"""
