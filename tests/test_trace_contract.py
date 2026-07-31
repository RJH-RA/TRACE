from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from trace_tfe3.data import CT_PHASES, TRACECaseDataset, trace_collate
from trace_tfe3.models import (
    CompactSliceBackbone,
    PathologyTeacher,
    TRACECTEncoder,
    TRACEModel,
)
from trace_tfe3.preprocessing.manifest import build_patient_manifest


ROOT = Path(__file__).resolve().parents[1]


class TRACEContractTests(unittest.TestCase):
    def test_config_and_examples_use_two_phase_ct(self) -> None:
        with open(ROOT / "configs" / "trace_default.yaml", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(tuple(config["data"]["phases"]), CT_PHASES)
        self.assertEqual(CT_PHASES, ("noncontrast", "arterial"))
        self.assertEqual(tuple(config["preprocessing"]["roi_shape_voxels"]), (16, 128, 128))
        with open(
            ROOT / "data" / "ct_cases.example.csv", newline="", encoding="utf-8"
        ) as stream:
            header = tuple(next(csv.reader(stream)))
        self.assertEqual(header, ("patient_id", "mask_path", "noncontrast", "arterial"))

    def test_dataset_constructs_enhancement_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "nc.npy", np.full((16, 8, 8), 0.25, dtype=np.float32))
            np.save(root / "ap.npy", np.full((16, 8, 8), 0.75, dtype=np.float32))
            torch.save(torch.randn(5, 1536), root / "he.pt")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "patient_id,split,label,noncontrast,arterial,he_token_embeddings\n"
                f"P1,train,1,{root / 'nc.npy'},{root / 'ap.npy'},{root / 'he.pt'}\n",
                encoding="utf-8",
            )
            item = TRACECaseDataset(manifest, "train", require_pathology=True)[0]
            self.assertEqual(item["ct"].shape, (3, 16, 8, 8))
            self.assertTrue(
                torch.allclose(item["ct"][2], torch.full((16, 8, 8), 0.5))
            )
            batch = trace_collate([item])
            self.assertEqual(batch["pathology_tokens"].shape, (1, 5, 1536))

    def test_training_rows_require_pathology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "ct.npy", np.zeros((16, 8, 8), dtype=np.float32))
            manifest = root / "manifest.csv"
            manifest.write_text(
                "patient_id,split,label,noncontrast,arterial,he_token_embeddings\n"
                f"P1,train,1,{root / 'ct.npy'},{root / 'ct.npy'},\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no H&E"):
                TRACECaseDataset(manifest, "train", require_pathology=True)[0]

    def test_trace_model_contract(self) -> None:
        encoder = TRACECTEncoder(
            backbone=CompactSliceBackbone(in_channels=3, output_dim=64),
            backbone_output_dim=64,
            shared_dim=32,
            token_count=16,
            transformer_layers=1,
            transformer_heads=4,
            freeze_backbone=False,
        )
        output = TRACEModel(encoder, shared_dim=32)(torch.randn(2, 3, 16, 32, 32))
        self.assertEqual(output["ct_tokens"].shape, (2, 16, 32))
        self.assertEqual(output["logit"].shape, (2,))

    def test_pathology_teacher_selects_representative_tokens(self) -> None:
        teacher = PathologyTeacher(
            token_embedding_dim=16,
            shared_dim=8,
            hidden_dim=8,
            representative_tokens=3,
        )
        mask = torch.tensor([[True, True, True, True, False]])
        output = teacher(torch.randn(1, 5, 16), mask)
        self.assertEqual(output["pathology_tokens"].shape, (1, 3, 8))
        self.assertTrue(output["pathology_mask"].all())

    def test_manifest_requires_he_for_every_training_patient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ct.csv").write_text(
                "patient_id,noncontrast,arterial\nP1,nc.npy,ap.npy\n",
                encoding="utf-8",
            )
            (root / "labels.csv").write_text(
                "patient_id,split,label\nP1,train,1\n", encoding="utf-8"
            )
            (root / "slides.csv").write_text(
                "patient_id,stain,embedding_path\nP1,cd34,cd34.pt\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Every training patient"):
                build_patient_manifest(
                    root / "ct.csv",
                    root / "labels.csv",
                    root / "slides.csv",
                    root / "out.csv",
                )


if __name__ == "__main__":
    unittest.main()
