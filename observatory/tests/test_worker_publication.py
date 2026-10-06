from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from observatory.train_worker import publish_result


class WorkerPublicationTests(unittest.TestCase):
    def publish_candidate(self, base_model: str):
        """Capture the exact folder the mocked publisher receives, not a cloud upload."""
        manifest = {
            "base_model": base_model,
            "base_revision": "base-pinned-revision",
            "stage": "cpt",
            "run_id": "candidate-one",
            "output_repo": "owner/research-adapters",
            "publish_policy": "private",
            "evaluation": {"sha256": "frozen-suite-hash"},
        }
        result = {"checks": {"passed": True, "independent_evaluation_passed": True}, "published": False,
                  "evaluation": {"passed": True, "status": "measured", "suite_sha256": "frozen-suite-hash"}}
        uploads, repositories, tokens = [], [], []

        class FakeHfApi:
            def __init__(self, token=None):
                tokens.append(token)

            def create_repo(self, **kwargs):
                repositories.append(kwargs)

            def model_info(self, repo_id):
                return SimpleNamespace(private=True)

            def upload_folder(self, **kwargs):
                folder = Path(kwargs["folder_path"])
                uploads.append({
                    **kwargs,
                    "files": {
                        path.relative_to(folder).as_posix(): path.read_bytes()
                        for path in folder.rglob("*") if path.is_file()
                    },
                })
                return SimpleNamespace(oid="published-pinned-revision")

        hub = ModuleType("huggingface_hub")
        hub.HfApi = FakeHfApi
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "adapter").mkdir()
            (output / "adapter" / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": base_model}), encoding="utf-8")
            with patch.dict("sys.modules", {"huggingface_hub": hub}), \
                    patch.dict("os.environ", {"HF_TOKEN": "test-owner-token"}):
                published = publish_result(manifest, result, output)
        self.assertEqual(tokens, ["test-owner-token"])
        self.assertEqual(repositories, [{
            "repo_id": "owner/research-adapters", "private": True, "exist_ok": True}])
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["path_in_repo"], "runs/candidate-one")
        self.assertTrue(published["published"])
        self.assertEqual(published["artifact_revision"], "published-pinned-revision")
        self.assertTrue(json.loads(uploads[0]["files"]["result.json"])["published"])
        return uploads[0]["files"]

    def test_llama_run_upload_contains_official_agreement_notice_and_attributed_card(self):
        files = self.publish_candidate("meta-llama/Llama-3.1-70B")
        license_dir = Path(__file__).resolve().parents[1] / "licenses" / "llama3.1"
        self.assertEqual(files["LICENSE"], (license_dir / "LICENSE").read_bytes())
        self.assertEqual(
            hashlib.sha256(files["LICENSE"]).hexdigest(),
            "bf2eac60b81e5c5f779fc3b4849d3bfacfd63f97f3eb68b44e9c6868c5801712")
        self.assertEqual(
            files["NOTICE"].decode("utf-8"),
            "Llama 3.1 is licensed under the Llama 3.1 Community License, "
            "Copyright © Meta Platforms, Inc. All Rights Reserved.\n")
        card = files["README.md"].decode("utf-8").replace("\r\n", "\n")
        self.assertTrue(card.startswith(
            "---\nbase_model: meta-llama/Llama-3.1-70B\nlicense: llama3.1\n---\n\n"
            "# Llama Consciousness Research Adapter\n"))
        self.assertIn("**Built with Llama**", card)
        self.assertIn("meta-llama/Llama-3.1-70B@base-pinned-revision", card)
        self.assertIn("This is a PEFT adapter, not a new foundation model.", card)
        self.assertIn("adapter/adapter_config.json", files)

    def test_non_llama_publication_keeps_existing_generic_card(self):
        files = self.publish_candidate("Qwen/Qwen3-4B-Base")
        self.assertEqual(
            files["README.md"].decode("utf-8").replace("\r\n", "\n"),
            "# Consciousness research adapter candidate\n\n"
            "Base: `Qwen/Qwen3-4B-Base@base-pinned-revision`. Stage: `cpt`.\n\n"
            "This is a PEFT adapter, not a new foundation model. Corpus lineage, held-out loss, "
            "frozen domain/general engineering evaluations against the unadapted base and incoming adapter, "
            "task-engagement checks and fresh intervention artifacts accompany this run. "
            "These measurements do not establish consciousness or pain.\n")
        self.assertNotIn("LICENSE", files)
        self.assertNotIn("NOTICE", files)


if __name__ == "__main__":
    unittest.main()
