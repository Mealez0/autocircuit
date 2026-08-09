"""Validated causal-mechanism specifications for discovery-only hypothesis testing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


@dataclass(frozen=True)
class MechanismVariable:
    """A high-level causal variable in a candidate internal algorithm."""

    name: str
    role: str

    def __post_init__(self) -> None:
        _identifier(self.name, "variable name")
        _identifier(self.role, "variable role")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "role": self.role}


@dataclass(frozen=True)
class MechanismStep:
    """One directed computation in a candidate mechanism DAG."""

    step_id: str
    operation: str
    inputs: tuple[str, ...]
    output: str
    carrier_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.step_id, "step id")
        _identifier(self.operation, "operation")
        _identifier(self.output, "step output")
        if not self.inputs:
            raise ValueError("mechanism step requires at least one input")
        for value in self.inputs:
            _identifier(value, "step input")
        for carrier in self.carrier_hints:
            _text(carrier, "carrier hint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "output": self.output,
            "carrier_hints": list(self.carrier_hints),
        }


@dataclass(frozen=True)
class MechanismPrediction:
    """A falsifiable categorical prediction for one proposed experiment."""

    experiment_id: str
    outcome: str
    rationale: str

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "prediction experiment id")
        _identifier(self.outcome, "prediction outcome")
        _text(self.rationale, "prediction rationale")

    def to_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "outcome": self.outcome,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MechanismExperiment:
    """An abstract causal experiment that a later compiler may execute."""

    experiment_id: str
    intervention_kind: str
    target: str
    readout: str
    cost: float
    rationale: str = "distinguish competing causal mechanism hypotheses"

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment id")
        _identifier(self.intervention_kind, "intervention kind")
        _identifier(self.target, "experiment target")
        _identifier(self.readout, "experiment readout")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
            raise ValueError("experiment cost must be a finite positive number")
        if not math.isfinite(float(self.cost)) or float(self.cost) <= 0.0:
            raise ValueError("experiment cost must be a finite positive number")
        _text(self.rationale, "experiment rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "intervention_kind": self.intervention_kind,
            "target": self.target,
            "readout": self.readout,
            "cost": float(self.cost),
            "rationale": self.rationale,
            "evidence_status": "proposal_only_not_evidence",
        }


@dataclass(frozen=True)
class MechanismHypothesis:
    """A candidate causal program whose predictions can be falsified."""

    hypothesis_id: str
    description: str
    variables: tuple[MechanismVariable, ...]
    steps: tuple[MechanismStep, ...]
    predictions: tuple[MechanismPrediction, ...]

    def __post_init__(self) -> None:
        _identifier(self.hypothesis_id, "hypothesis id")
        _text(self.description, "hypothesis description")
        if not self.variables or not self.steps or not self.predictions:
            raise ValueError("mechanism hypothesis requires variables, steps, and predictions")

        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("mechanism hypothesis contains duplicate variables")
        variable_set = set(variable_names)

        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("mechanism hypothesis contains duplicate step ids")

        producers: dict[str, str] = {}
        for step in self.steps:
            if step.output not in variable_set or any(item not in variable_set for item in step.inputs):
                raise ValueError("mechanism step references an unknown variable")
            if step.output in producers:
                raise ValueError("mechanism variable has multiple producing steps")
            producers[step.output] = step.step_id

        prediction_ids = [prediction.experiment_id for prediction in self.predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("mechanism hypothesis contains duplicate experiment predictions")

        self.topological_steps()

    def topological_steps(self) -> tuple[str, ...]:
        """Return a deterministic topological order or reject cyclic mechanisms."""

        position = {step.step_id: index for index, step in enumerate(self.steps)}
        producer = {step.output: step.step_id for step in self.steps}
        dependencies: dict[str, set[str]] = {step.step_id: set() for step in self.steps}
        dependents: dict[str, set[str]] = {step.step_id: set() for step in self.steps}
        for step in self.steps:
            for value in step.inputs:
                upstream = producer.get(value)
                if upstream is not None and upstream != step.step_id:
                    dependencies[step.step_id].add(upstream)
                    dependents[upstream].add(step.step_id)
                elif upstream == step.step_id:
                    raise ValueError("mechanism contains a causal cycle")

        ready = sorted(
            (step_id for step_id, deps in dependencies.items() if not deps),
            key=position.__getitem__,
        )
        ordered: list[str] = []
        while ready:
            step_id = ready.pop(0)
            ordered.append(step_id)
            for dependent in sorted(dependents[step_id], key=position.__getitem__):
                dependencies[dependent].discard(step_id)
                if not dependencies[dependent] and dependent not in ready:
                    ready.append(dependent)
                    ready.sort(key=position.__getitem__)
        if len(ordered) != len(self.steps):
            raise ValueError("mechanism contains a causal cycle")
        return tuple(ordered)

    def prediction_map(self) -> dict[str, str]:
        return {prediction.experiment_id: prediction.outcome for prediction in self.predictions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "variables": [variable.to_dict() for variable in self.variables],
            "steps": [step.to_dict() for step in self.steps],
            "topological_steps": list(self.topological_steps()),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "evidence_status": "proposal_only_not_evidence",
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MechanismCampaign:
    """A complete competing-hypothesis matrix for active falsification."""

    campaign_id: str
    task: str
    hypotheses: tuple[MechanismHypothesis, ...]
    experiments: tuple[MechanismExperiment, ...]
    selected_layer: int | None = None
    candidate_carriers: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign id")
        _identifier(self.task, "campaign task")
        if not self.hypotheses or not self.experiments:
            raise ValueError("mechanism campaign requires hypotheses and experiments")
        hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in self.hypotheses]
        experiment_ids = [experiment.experiment_id for experiment in self.experiments]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("mechanism campaign contains duplicate hypotheses")
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("mechanism campaign contains duplicate experiments")
        required = set(experiment_ids)
        for hypothesis in self.hypotheses:
            predicted = set(hypothesis.prediction_map())
            if predicted != required:
                raise ValueError("mechanism campaign prediction matrix is incomplete")
        if self.selected_layer is not None and (
            isinstance(self.selected_layer, bool) or self.selected_layer < 0
        ):
            raise ValueError("selected layer must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "task": self.task,
            "interpretation_scope": "exploratory_discovery_only",
            "selected_layer": self.selected_layer,
            "candidate_carriers": dict(self.candidate_carriers or {}),
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "experiments": [experiment.to_dict() for experiment in self.experiments],
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "scientific_confirmation": False,
            "circuit_found": False,
        }
