"""LangGraph supervisor/router — Phase 5, not yet implemented.

Routes an anomaly event through Correlator -> Diagnostician -> Reporter.
Statistics are computed in Python before any LLM call; the LLM's job is
natural-language synthesis of pre-computed structured results, not raw
number-crunching (same principle as PITWALL·AI's agent architecture).
"""
