#!/usr/bin/env python3
"""Three-seed search of seven-domain sampling priors for the distillation adapters.

The acoustic models, 832-D output contract, duration detector and safe runtime
are frozen.  Only the sampling priors used to train the four 192-D low-rank
short-to-full adapter matrices are varied.  Selection uses the external
AliMeeting Eval and ChildMandarin Dev speakers.  Coverage and the already
consumed AliMeeting Test are audits, never search objectives.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.train_allsource_output_kd as trainer
from tools.run_redim_output_kd_support import (
    DURATIONS, MODEL_DIRS, MODEL_NAMES, load_redim, load_acoustic,
)
from tools.train_allsource_all_himia import all_himia_ids
from tools.validate_redim_output_kd_3seed import (
    EXTERNALS, load_external_cache,
)
from tools.in_domain_residual_adapter import (
    score_cached, score_coverage, summarize,
)
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import atomic_json, load_distillation_adapters, sha256_file


OUTPUT_DIR = ROOT / "records/2026-09-05/053_adapter_prior_search"
RESULT_PATH = OUTPUT_DIR / "SEARCH_RESULT.json"
BEST_ADAPTER_PATH = OUTPUT_DIR / "BEST_ADAPTER.npz"
# Reuse the released distillation seeds so the anchor configuration is an exact
# reproduction check and sampling-prior changes are the only variable.
SEEDS = (825001, 825002, 825003)
TAIL_PRIORS = {
    "seniortalk": 0.07,
    "commonvoice_beijing": 0.07,
    "magicdata_slr68": 0.05,
}


def prior_grid():
    configs = []
    for three_d in (0.18, 0.22, 0.26):
        for child in (0.10, 0.14, 0.16, 0.18, 0.22, 0.26):
            for aishell in (0.04, 0.08, 0.12, 0.16):
                himia = 1.0 - sum(TAIL_PRIORS.values()) - three_d - child - aishell
                if not (0.15 <= himia <= 0.40):
                    continue
                priors = {
                    "3dspeaker": three_d,
                    "himia_train": himia,
                    "childmandarin": child,
                    "aishell4": aishell,
                }
                priors.update(TAIL_PRIORS)
                if abs(sum(priors.values()) - 1.0) > 1.0e-12:
                    raise ValueError("prior grid does not sum to one")
                configs.append({
                    "id": "d{:02d}_h{:02d}_c{:02d}_a{:02d}".format(
                        round(three_d * 100), round(himia * 100),
                        round(child * 100), round(aishell * 100),
                    ),
                    "priors": priors,
                })
    anchor = {
        "id": "released_distillation_anchor",
        "priors": {
            "3dspeaker": 0.22,
            "himia_train": 0.35,
            "childmandarin": 0.16,
            "aishell4": 0.08,
            "seniortalk": 0.07,
            "commonvoice_beijing": 0.07,
            "magicdata_slr68": 0.05,
        },
    }
    unique = {row["id"]: row for row in configs}
    unique[anchor["id"]] = anchor
    return [unique[key] for key in sorted(unique)]


def fit_candidate(redims, metadata, himia, himia_ids, config):
    trainer.DATASET_PRIORS = dict(config["priors"])
    seed_adapters = {}
    audits = []
    for seed in SEEDS:
        seed_adapters[seed] = {name: {} for name in MODEL_NAMES}
        for model_offset, name in enumerate(MODEL_NAMES):
            for duration_offset, duration in enumerate(DURATIONS):
                delta, audit = trainer.train(
                    redims[name], metadata, himia, himia_ids, name, duration,
                    seed + model_offset * 10 + duration_offset,
                )
                seed_adapters[seed][name][duration] = delta
                audits.append(audit)
    soup = {name: {} for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        for duration in DURATIONS:
            soup[name][duration] = np.mean(np.stack([
                seed_adapters[seed][name][duration] for seed in SEEDS
            ]).astype(np.float64), axis=0).astype(np.float32)
    return soup, audits


def adapter_arrays(candidate, config, status):
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "status": np.asarray(status),
        "config_id": np.asarray(config["id"]),
        "seed_ids": np.asarray(SEEDS, dtype=np.int64),
        "dataset_names": np.asarray(sorted(config["priors"])),
        "dataset_priors": np.asarray([
            config["priors"][key] for key in sorted(config["priors"])
        ], dtype=np.float64),
    }
    for name in MODEL_NAMES:
        for duration in DURATIONS:
            arrays[name + "_" + duration] = np.ascontiguousarray(
                candidate[name][duration], dtype=np.float32
            )
    return arrays


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def development_checks(metrics):
    return {
        "macro_positive": metrics["macro"] > 0.0,
        "short_positive": metrics["short_macro"] > 0.0,
        "worst_ge_minus_0p2": metrics["worst"] >= -0.2,
        "each_dataset_short_nonnegative": all(
            value["short_macro"] >= 0.0
            for value in metrics["datasets"].values()
        ),
    }


def main():
    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = prior_grid()
    current = load_distillation_adapters()
    _, _, metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    himia_spec = EXTERNALS["himia_dev"]
    himia = load_external_cache(
        himia_spec["cache"], himia_spec["expected_cache"]
    )
    himia_ids, _ = all_himia_ids(himia_spec["trials"])
    _, dev_trials, dev_cache = load_suite(SUITES[0])

    development = []
    candidate_cache = {}
    anchor_reproduction = None
    for index, config in enumerate(configs, 1):
        print("PRIOR {}/{} {}".format(index, len(configs), config["id"]), flush=True)
        candidate, audits = fit_candidate(
            redims, metadata, himia, himia_ids, config
        )
        candidate_cache[config["id"]] = candidate
        metrics = summarize(score_cached(
            dev_trials, dev_cache, current, candidate
        ))
        checks = development_checks(metrics)
        row = {
            "config": config,
            "metrics": metrics,
            "checks": checks,
            "eligible": all(checks.values()),
            "fit_elapsed_seconds": float(sum(
                item["elapsed_seconds"] for item in audits
            )),
            "adapter_frobenius": {
                name + "_" + duration: float(np.linalg.norm(candidate[name][duration]))
                for name in MODEL_NAMES for duration in DURATIONS
            },
        }
        development.append(row)
        if config["id"] == "released_distillation_anchor":
            maximum = max(
                float(np.max(np.abs(
                    candidate[name][duration] - current[name][duration]
                )))
                for name in MODEL_NAMES for duration in DURATIONS
            )
            anchor_reproduction = {
                "max_abs_difference_vs_released_adapter": maximum,
                "exact_float32_reproduction": bool(maximum == 0.0),
            }
        print("  macro={:.6f} short={:.6f} worst={:.6f} pass={}".format(
            metrics["macro"], metrics["short_macro"], metrics["worst"],
            row["eligible"],
        ), flush=True)

    development.sort(key=lambda row: (
        row["eligible"],
        min(value["short_macro"] for value in row["metrics"]["datasets"].values()),
        row["metrics"]["short_macro"], row["metrics"]["macro"],
        row["metrics"]["worst"],
    ), reverse=True)

    # Coverage is an audit only.  Evaluate the strongest eight development
    # candidates, then choose the most balanced one that does not regress.
    audited = []
    for row in development[:8]:
        candidate = candidate_cache[row["config"]["id"]]
        coverage = summarize(score_coverage(current, candidate))
        coverage_checks = {
            "macro_nonnegative": coverage["macro"] >= 0.0,
            "short_nonnegative": coverage["short_macro"] >= 0.0,
            "worst_ge_minus_0p2": coverage["worst"] >= -0.2,
            "each_dataset_short_nonnegative": all(
                value["short_macro"] >= 0.0
                for value in coverage["datasets"].values()
            ),
        }
        copied = json.loads(json.dumps(row, sort_keys=True))
        copied["coverage_audit"] = coverage
        copied["coverage_checks"] = coverage_checks
        copied["audit_eligible"] = bool(
            copied["eligible"] and all(coverage_checks.values())
        )
        audited.append(copied)
    audited.sort(key=lambda row: (
        row["audit_eligible"],
        min(row["metrics"]["short_macro"], row["coverage_audit"]["short_macro"]),
        row["metrics"]["short_macro"] + row["coverage_audit"]["short_macro"],
        row["metrics"]["macro"] + row["coverage_audit"]["macro"],
    ), reverse=True)
    selected = next((row for row in audited if row["audit_eligible"]), None)

    final = None
    status = "PRIOR_SEARCH_GATE_FAILED_DO_NOT_PACKAGE"
    if selected is not None:
        candidate = candidate_cache[selected["config"]["id"]]
        _, final_trials, final_cache = load_suite(SUITES[1])
        final_metrics = summarize(score_cached(
            final_trials, final_cache, current, candidate
        ))
        final_checks = development_checks(final_metrics)
        final = {"metrics": final_metrics, "checks": final_checks}
        if all(final_checks.values()):
            status = "PRIOR_SEARCH_PASS_REQUIRES_RUNTIME"
            atomic_npz(
                BEST_ADAPTER_PATH,
                adapter_arrays(candidate, selected["config"], status),
            )

    payload = {
        "schema_version": 1,
        "artifact_kind": "joint_three_seed_seven_domain_adapter_prior_search",
        "status": status,
        "distillation_online_anchor": 89.76798,
        "target_online_score": 91.0,
        "search_config_count": len(configs),
        "selection_data": ["alimeeting_eval_holdout", "childmandarin_dev_holdout"],
        "audit_data": ["coverage_lodo", "alimeeting_test_consumed"],
        "seeds": list(SEEDS),
        "anchor_reproduction": anchor_reproduction,
        "development_ranking": development,
        "coverage_audited_top8": audited,
        "selected": selected,
        "final_confirmation": final,
        "best_adapter": (
            {
                "path": str(BEST_ADAPTER_PATH.relative_to(ROOT)),
                "bytes": BEST_ADAPTER_PATH.stat().st_size,
                "sha256": sha256_file(BEST_ADAPTER_PATH),
            }
            if BEST_ADAPTER_PATH.is_file() else None
        ),
        "limitations": [
            "All public datasets are proxy evidence and cannot guarantee a hidden score of 91.",
            "AliMeeting Test was consumed by earlier work and is only a final audit here.",
            "No package is built unless every point-estimate gate passes.",
        ],
    }
    atomic_json(RESULT_PATH, payload)
    print(json.dumps({
        "status": status,
        "configs": len(configs),
        "development_best": development[0],
        "selected": selected,
        "final_confirmation": final,
        "best_adapter": payload["best_adapter"],
        "result": str(RESULT_PATH.relative_to(ROOT)),
        "sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
