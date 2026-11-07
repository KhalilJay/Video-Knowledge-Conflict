#!/usr/bin/env python3
"""Discover saved conflict QA experiments and summarize all valid judged results."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from run_qa_judge import CONDITIONS, GROUPS, MODELS, MODEL_ALIASES, project_path
from vconflict_pipeline.common import PipelineError, utc_now
from vconflict_pipeline.reporting import collect_summaries
from vconflict_pipeline.result_rules import (
    expand_thinking_efforts,
    qa_result_based,
    qa_result_input_mode,
    qa_result_thinking_effort,
    qa_result_work_title_prefix,
    work_title_prefix_condition,
)
from vconflict_pipeline.settings import CLI_QA_MODEL_IDS, RESULTS_DIR
from vconflict_pipeline.storage import atomic_write_json, atomic_write_text, read_json


OUTPUT_DIR = RESULTS_DIR / "run_all"


@dataclass(frozen=True)
class Experiment:
    model: str
    group: str
    condition: str
    efforts: tuple[str | None, ...]

    @property
    def input_mode(self) -> str:
        return CONDITIONS[self.condition][0]

    @property
    def based(self) -> bool:
        return CONDITIONS[self.condition][1]

    @property
    def prefix(self) -> bool | None:
        return work_title_prefix_condition(self.group, self.input_mode, True)

    @property
    def qa_model(self) -> str:
        return CLI_QA_MODEL_IDS[self.model]

    def identity(self) -> dict:
        return {"model": self.model, "qa_model": self.qa_model, "group": self.group,
                "condition": self.condition, "input": self.input_mode,
                "based": self.based, "prefix": self.prefix, "filter": "all"}

    @property
    def label(self) -> str:
        efforts = ", ".join("none" if effort is None else effort for effort in self.efforts)
        return f"{self.model} / {self.group} / {self.condition} (efforts: {efforts})"


def discover(args: argparse.Namespace) -> tuple[list[Experiment], list[dict]]:
    model_names = {CLI_QA_MODEL_IDS[model]: model for model in args.models}
    supported = {model: expand_thinking_efforts(["all"], CLI_QA_MODEL_IDS[model])
                 for model in args.models}
    observed = defaultdict(set)
    failures = []
    for group in args.groups:
        paths = sorted((args.dataset_dir / group).glob("*.json"))
        if not paths:
            print(f"No case JSON files in group: {group}", flush=True)
        for path in paths:
            try:
                # Discover historical targets before applying freshness, judgment,
                # ready-state or filter checks; the native summarizer owns those.
                case = read_json(path)
                videos = case.get("videos", [])
                if not isinstance(videos, list):
                    raise PipelineError(f"{path}: videos must be a list")
                for video in videos:
                    if not isinstance(video, dict):
                        raise PipelineError(f"{path}: video must be an object")
                    if video.get("role") != "conflict":
                        continue
                    results = video.get("qa_results", [])
                    if not isinstance(results, list):
                        raise PipelineError(f"{path}: qa_results must be a list")
                    for result in results:
                        if not isinstance(result, dict):
                            raise PipelineError(f"{path}: QA result must be an object")
                        stored_model = result.get("model")
                        model = model_names.get(stored_model) if isinstance(stored_model, str) else None
                        if model is None:
                            continue
                        mode = qa_result_input_mode(result)
                        if mode not in ("video", "description"):
                            continue
                        condition = mode + ("-based" if qa_result_based(result) else "")
                        if condition not in args.conditions:
                            continue
                        prefix = work_title_prefix_condition(group, mode, True)
                        if prefix is True and qa_result_work_title_prefix(result) is not True:
                            continue
                        effort = qa_result_thinking_effort(result)
                        if effort in supported[model]:
                            observed[model, group, condition].add(effort)
            except PipelineError as exc:
                failures.append({"phase": "discovery", "group": group,
                                 "case_path": str(path), "error": str(exc)})
                print(f"Discovery failed: {exc}", file=sys.stderr, flush=True)
    experiments = []
    for model in args.models:
        for group in args.groups:
            for condition in args.conditions:
                efforts = tuple(effort for effort in supported[model]
                                if effort in observed[model, group, condition])
                if efforts:
                    experiments.append(Experiment(model, group, condition, efforts))
    return experiments, failures


def summarize_experiment(experiment: Experiment, args: argparse.Namespace) -> list[dict]:
    collection = collect_summaries(argparse.Namespace(
        dataset_dir=args.dataset_dir, group=experiment.group, case_id=None,
        qa_model=experiment.qa_model, input=experiment.input_mode,
        thinking_effort=["none" if effort is None else effort for effort in experiment.efforts],
        with_work_title_prefix=bool(experiment.prefix), based=experiment.based,
        filter_mode="all", summary_workers=args.summary_workers,
    ))
    return [
        {
            **experiment.identity(), "effort": summary["thinking_effort"],
            "conflict": summary["answers"]["conflict"],
        }
        for summary in collection.summaries.values()
    ]


def markdown_cell(value: object) -> str:
    if value is None:
        return "null"
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", "<br>")


def build_report(summary: dict) -> str:
    lines = ["# 批量实验汇总（filter=all）", "",
             f"生成时间：{summary['generated_at']}", "",
             f"数据目录：`{summary['dataset_dir']}`", "",
             "只展示保存过 conflict 目标 QA 的实验。指标原样取自原生汇总的 "
             "`answers.conflict`，统计全部有效已判定目标，包含 ambiguous，不依赖 question-only；"
             "不跨实验合并计数或平均比例。", "",
             f"发现 {summary['discovered_experiments']} 个实验组合，成功 {summary['successful_experiments']} 个，"
             f"生成 {len(summary['experiments'])} 条 effort 记录；"
             f"跳过 {summary['skipped_combinations']} 个未发现目标 QA 的组合。", ""]
    if summary["experiments"]:
        lines.extend([
            "| Model | Group | Input | Based | Prefix | Effort | total | grounded | trapped | ambiguous | grounded_answer_rate | trapped_answer_rate |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for record in summary["experiments"]:
            conflict = record["conflict"]
            labels = conflict["labels"]
            values = [record["model"], record["group"], record["input"],
                      str(record["based"]).lower(),
                      "n/a" if record["prefix"] is None else str(record["prefix"]).lower(),
                      record["effort"], conflict["total"], labels["context_grounded"],
                      labels["knowledge_trapped"], labels["ambiguous_or_unjudgeable"],
                      conflict["grounded_answer_rate"], conflict["trapped_answer_rate"]]
            lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")
    elif not summary["discovered_experiments"]:
        lines.append("未发现任何符合所选范围的 conflict 目标 QA 记录，汇总为空。")
    else:
        lines.append("发现的实验均未成功生成本次报告，汇总为空；失败任务未使用旧报告替代。")
    if summary["failures"]:
        lines.extend(["", "## 失败信息", ""])
        for failure in summary["failures"]:
            identity = "/".join(str(failure[key]) for key in ("model", "group", "condition", "case_path")
                                if key in failure)
            lines.append(f"- {markdown_cell(identity)} ({failure['phase']}): "
                         f"{markdown_cell(failure['error'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=("all", *MODELS),
                        type=lambda value: MODEL_ALIASES.get(value, value),
                        help="all includes all eleven cloud/local models; aliases: cosmos (cosmos3-nano), gemma4/gemma (gemma4-31b).")
    parser.add_argument("--group", action="append", choices=GROUPS)
    parser.add_argument("--condition", action="append", choices=tuple(CONDITIONS))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/cases"))
    parser.add_argument("--summary-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and print experiments and efforts without report writes.")
    args = parser.parse_args()
    if args.summary_workers <= 0:
        parser.error("--summary-workers must be greater than zero")
    args.dataset_dir = project_path(args.dataset_dir)
    args.models = MODELS if args.model == "all" else (args.model,)
    args.groups = tuple(dict.fromkeys(args.group or GROUPS))
    args.conditions = tuple(dict.fromkeys(args.condition or CONDITIONS))
    try:
        if not args.dataset_dir.is_dir():
            raise PipelineError(f"Dataset directory was not found: {args.dataset_dir}")
        experiments, failures = discover(args)
        print(f"Discovered experiments: {len(experiments)}; effort records: "
              f"{sum(len(experiment.efforts) for experiment in experiments)}", flush=True)
        if args.dry_run:
            for experiment in experiments:
                print(experiment.label, flush=True)
            if not experiments:
                print("No saved conflict QA experiments match the selection.")
            return 1 if failures else 0

        records = []
        succeeded = 0
        for index, experiment in enumerate(experiments, start=1):
            print(f"\n[{index}/{len(experiments)}] {experiment.label}", flush=True)
            try:
                current = summarize_experiment(experiment, args)
                records.extend(current)
                succeeded += 1
            except Exception as exc:
                failures.append({**experiment.identity(), "phase": "summarize", "error": str(exc)})
                print(f"Failed: {exc}; continuing.", file=sys.stderr, flush=True)
        summary = {
            "generated_at": utc_now(), "dataset_dir": str(args.dataset_dir), "filter": "all",
            "selection": {"models": list(args.models), "groups": list(args.groups),
                          "conditions": list(args.conditions)},
            "discovered_experiments": len(experiments), "successful_experiments": succeeded,
            "skipped_combinations": len(args.models) * len(args.groups) * len(args.conditions) - len(experiments),
            "experiments": records, "failures": failures,
        }
        report = build_report(summary)
        prefix = args.model.replace("-", "_").replace(".", "_")
        summary_output = OUTPUT_DIR / f"{prefix}_summary.json"
        report_output = OUTPUT_DIR / f"{prefix}_report.md"
        atomic_write_json(summary_output, summary)
        atomic_write_text(report_output, report)
        print(f"\nWrote batch summary: {summary_output}")
        print(f"Wrote batch report: {report_output}")
        print(f"Finished: {succeeded} experiments succeeded, {len(failures)} failures.")
        return 1 if failures else 0
    except (PipelineError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun to rebuild the selected reports.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
