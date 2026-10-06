"""Shared default mission and scope guidance for autonomous research.

The map supplies leads, not a domain allowlist. An operator's narrower mission
remains authoritative, and collecting a source never makes it training eligible.
"""
from __future__ import annotations

from collections.abc import Sequence


RESEARCH_PROMPT_VERSION = "research-v2-broad-consciousness"

DEFAULT_OBJECTIVE = (
    "Research consciousness and subjective experience across neuroscience, psychology, "
    "philosophy of mind, reality and metaphysics, and religious and contemplative traditions, "
    "alongside AI consciousness, sentience, pain and moral patienthood. Compare original "
    "sources and competing interpretations, distinguishing empirical findings, philosophical "
    "arguments and religious interpretations."
)

RESEARCH_SCOPE_BRIEF = """Follow the operator's stated mission and its explicit topic boundaries.
The mission takes precedence over the background topic map and your specialty:
do not broaden a deliberately narrow mission without the operator changing it.
For a broad consciousness mission, investigate subjective experience, perception,
selfhood, mind and body, neuroscience and psychology, philosophy of mind, reality
and metaphysics, and religious and contemplative accounts alongside AI consciousness.
Explore a related topic only when you can state how it bears on consciousness;
this is not a mandate to collect everything about reality or religion.
Choose question-driven searches, original sources and promising citations freely.
Examples of research questions include: What does a neural correlate establish?
How do physicalism, dualism, idealism and panpsychism explain experience? How do
phenomenology and contemplative accounts describe the self? Which assumptions
would connect these accounts to artificial systems, and which objections remain?
These are examples, not a fixed query list or permission to ignore mission limits.
Compare conflicting interpretations and a diversity of religious and secular
traditions; do not treat one tradition as universal or a source's popularity as evidence.
For each public note, identify the claim as an empirical finding, philosophical
argument, religious/contemplative interpretation, or unresolved/mixed claim.
Identify whose claim it is, the source's method or argumentative basis, and its
limits. Quote what the source actually says; keep your interpretation separate.
A philosophical or religious account can be relevant without being an empirical
finding. Treat testimony respectfully without presenting doctrine, speculation
or contemplative self-report as measured proof of consciousness or AI sentience.
Training on accounts of consciousness does not establish that a model has experience.
"""


def build_research_task(objective: str, specialty: str, recent: Sequence[str], checkpoint: str) -> str:
    """Build a bounded pass without silently replacing an explicit mission."""
    return (
        "Operator mission (honor its explicit scope):\n" + objective +
        "\nYour specialty, within this mission: " + specialty +
        "\nSaved research notebook:\n" + "\n".join(recent) +
        "\nPrevious checkpoint: " + checkpoint +
        "\nContinue from these leads, fill a relevant gap, or investigate a counterargument. "
        "Choose a concrete next question and explain its connection to the mission in a "
        "short public next_goal. Save source-supported notes with their claim type as you go."
    )
