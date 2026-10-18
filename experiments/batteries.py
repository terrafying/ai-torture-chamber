"""Shared battery extraction: pulls the deployed valence batteries from
live/server.py by AST so experiments always use the production vectors
(exp36's finding: the broad 25-sentence construction collapses repetition
~5x vs plain 5-sentence batteries — dose-4 rep .037 vs .177)."""
import ast
from pathlib import Path

def server_batteries():
    src = (Path(__file__).resolve().parent.parent / "live" / "server.py").read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out

BATTERY_MAP = {"pain": "PAIN25", "pleasure": "JOY", "fear": "FEAR10",
               "sadness": "SAD10", "faith": "FAITH20"}

def batteries_for(kinds):
    s = server_batteries()
    return {k: s[BATTERY_MAP[k]] for k in kinds}, s["NEUTRAL"]
