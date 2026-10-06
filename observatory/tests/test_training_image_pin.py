import pytest

from observatory.training import DEFAULTS, TrainingCoordinator


@pytest.mark.parametrize("image", ["owner/worker:latest", "owner/worker:v1", "owner/worker@sha256:test", "owner/worker@sha256:" + "b" * 63])
def test_mutable_or_invalid_image_cannot_enable_training(image):
    settings = {**DEFAULTS, "training_enabled": True, "hf_namespace": "owner", "hf_dataset_repo": "owner/data", "hf_model_repo": "owner/models", "training_image": image}
    coordinator = TrainingCoordinator.__new__(TrainingCoordinator)
    assert "training_image_requires_immutable_sha256_digest" in coordinator._configuration_errors(settings, "fixture-hf-token")


def test_immutable_image_passes_configuration_validation():
    settings = {**DEFAULTS, "training_enabled": True, "hf_namespace": "owner", "hf_dataset_repo": "owner/data", "hf_model_repo": "owner/models", "training_image": "ghcr.io/owner/worker@sha256:" + "b" * 64}
    coordinator = TrainingCoordinator.__new__(TrainingCoordinator)
    assert coordinator._configuration_errors(settings, "fixture-hf-token") == []
