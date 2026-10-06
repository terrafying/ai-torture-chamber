from observatory.research import ROLES, SYSTEM, save_document, save_note
from observatory.research_scope import DEFAULT_OBJECTIVE, RESEARCH_PROMPT_VERSION, build_research_task
from observatory.store import Store


def test_default_research_pass_retains_broad_mission_and_individual_memory():
    specialty = dict((role, brief) for role, _, brief in ROLES)["cartographer"]
    task = build_research_task(
        DEFAULT_OBJECTIVE, specialty,
        ["Philosophical argument: compare the author's account with objections."],
        "Next lead: a source on contemplative accounts of selfhood.",
    )
    assert DEFAULT_OBJECTIVE in task
    assert specialty in task
    assert "the author's account with objections" in task
    assert "contemplative accounts of selfhood" in task
    assert "claim type" in task
    assert "empirical finding" in SYSTEM and "religious/contemplative interpretation" in SYSTEM


def test_explicit_narrow_mission_is_preserved_without_appending_default_objective():
    objective = "Only compare visual perception studies in humans; exclude AI and theology."
    task = build_research_task(objective, ROLES[0][2], [], "No previous sources.")
    assert objective in task
    assert DEFAULT_OBJECTIVE not in task
    assert "within this mission" in task
    assert "do not broaden a deliberately narrow mission" in SYSTEM


def test_philosophical_source_note_records_new_brief_version_without_admitting_training(tmp_path):
    passage = (
        "The argument describes experience through an account of mind and reality, "
        "rather than a controlled empirical measurement."
    )
    store = Store(tmp_path / "state.db")
    try:
        source = save_document(
            store, url="https://example.org/philosophy-of-mind", title="A philosophical argument",
            text=(passage + "\n") * 4, agent_id="cartographer", scope="article",
        )
        note = save_note(
            store, source, "cartographer",
            "Philosophical argument: the author offers a metaphysical account; it does not measure AI sentience.",
            passage,
        )
        assert note["support_verified"]
        assert note["prompt_version"] == RESEARCH_PROMPT_VERSION
        assert note["review_status"] == "pending"
        assert not source["curation"]["eligible"]
        assert "rights_not_verified" in source["curation"]["reasons"]
        assert "source_quality_review_required" in source["curation"]["reasons"]
    finally:
        store.close()
