import pytest

from observatory.store import Store, secret_field


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "settings.sqlite3")
    yield value
    value.close()


@pytest.mark.parametrize("updates", [
    {"agent_count": "not-a-number"}, {"agent_count": True}, {"agent_count": 7},
    {"research_provider": "unknown"}, {"research_protocol": "rpc"},
    {"research_provider": ["x402"]}, {"browser_provider": {}},
    {"browser_provider": "my-personal-browser"}, {"research_model": ""},
    {"research_max_output_tokens": -1},
    {"auto_curation_enabled": "true"}, {"auto_curation_enabled": True},
    {"auto_curation_policy_ack": {}}, {"auto_curation_policy_ack": "any-document"},
])
def test_invalid_control_settings_do_not_replace_saved_configuration(store, updates):
    store.save_settings({"agent_count": 2, "research_provider": "x402"})
    with pytest.raises(ValueError):
        store.save_settings(updates)
    assert store.get_settings(private=True) == {"agent_count": 2, "research_provider": "x402"}


def test_automatic_curation_requires_an_explicit_policy_choice(store):
    assert not store.get_settings().get("auto_curation_enabled", False)
    store.save_settings({"auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v1"})
    assert store.get_settings()["auto_curation_enabled"] is True
    store.save_settings({"auto_curation_enabled": False, "auto_curation_policy_ack": ""})
    assert store.get_settings()["auto_curation_enabled"] is False


def test_legacy_ack_is_not_silently_upgraded_to_broad_scope(store):
    from observatory.automatic_curation import enabled
    store.save_settings({"auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v1"})
    assert not enabled(store.get_settings(private=True))
    store.save_settings({"auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v2"})
    assert enabled(store.get_settings(private=True))


def test_x402_settings_and_environment_signer_isolation(store, monkeypatch):
    store.save_settings({"research_provider": "x402", "research_protocol": "responses", "research_model": "openai/gpt-6-astra", "agent_count": "3"})
    assert store.get_settings()["agent_count"] == 3
    for key in ("x402_solana_private_key", "payment_broker_token", "wallet_seed_phrase"):
        with pytest.raises(ValueError):
            store.save_settings({key: "fixture-not-a-real-key"})
    monkeypatch.setenv("OBSERVATORY_PAYMENT_BROKER_TOKEN", "fixture-broker-token")
    monkeypatch.setenv("X402_SOLANA_PRIVATE_KEY", "fixture-spending-key")
    result = store.sanitize({"message": "fixture-broker-token fixture-spending-key", "private_key": "fixture-spending-key", "signed_transaction": "opaque", "wallet_address": "public-address"})
    assert result == {"message": "[REDACTED] [REDACTED]", "private_key": "[REDACTED]", "wallet_address": "public-address"}
    assert secret_field("payment_broker_token")
