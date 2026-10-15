#!/usr/bin/env python3
"""Run all-source output KD after all HI-MIA diagnostics are consumed."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.train_allsource_output_kd as experiment


def all_himia_ids(trial_path):
    with Path(trial_path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    utterances = {
        str(row[key]["utterance_stable_id"])
        for row in rows for key in ("enrollment", "test")
    }
    speakers = {
        str(row[key]["speaker_stable_id"])
        for row in rows for key in ("enrollment", "test")
    }
    if len(utterances) != 576 or len(speakers) != 36:
        raise ValueError("all-HI-MIA consumed set changed")
    return utterances, speakers


experiment.OUTPUT_DIR = experiment.ROOT / (
    "records/2026-09-04/026_distillation_allsource_output_kd_all_himia"
)
experiment.OUTPUT = experiment.OUTPUT_DIR / "DEVELOPMENT_RESULT.json"
experiment.ADAPTERS = experiment.OUTPUT_DIR / "distillation_allsource_all_himia_adapters.npz"
experiment.SEEDS = (825001, 825002, 825003)
experiment.himia_training_ids = all_himia_ids
experiment.DATASET_PRIORS = {
    "3dspeaker": 0.22,
    "himia_train": 0.35,
    "childmandarin": 0.16,
    "aishell4": 0.08,
    "seniortalk": 0.07,
    "commonvoice_beijing": 0.07,
    "magicdata_slr68": 0.05,
}


if __name__ == "__main__":
    experiment.main()
