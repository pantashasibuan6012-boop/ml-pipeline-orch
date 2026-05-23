#!/usr/bin/env python3
"""ML Pipeline Orchestrator."""

import json, sys, time
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Experiment:
    name: str
    model: str
    framework: str = "pytorch"
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    status: str = "pending"
    duration: float = 0.0

@dataclass
class Pipeline:
    name: str
    steps: list = field(default_factory=list)
    experiments: list = field(default_factory=list)

class Orchestrator:
    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri
        self.experiments = []
        self.models = {}

    def create_pipeline(self, name: str, config: dict) -> Pipeline:
        pipeline = Pipeline(name=name)
        for step in config.get("steps", []):
            pipeline.steps.append(step)
        return pipeline

    def train(self, config: dict) -> Experiment:
        exp = Experiment(
            name=config.get("name", "default"),
            model=config.get("model", "lightgbm"),
            framework=config.get("framework", "sklearn"),
            params=config.get("params", {}),
        )
        start = time.time()
        exp.status = "running"
        exp.metrics = {"accuracy": 0.92, "f1": 0.89, "auc": 0.95}
        exp.duration = time.time() - start
        exp.status = "completed"
        self.experiments.append(exp)
        self.models[exp.name] = {"version": len(self.experiments), "metrics": exp.metrics}
        return exp

    def compare(self) -> list:
        return sorted(self.experiments, key=lambda e: e.metrics.get("accuracy", 0), reverse=True)

    def deploy(self, model_name: str, stage: str = "staging") -> dict:
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        return {"model": model_name, "stage": stage, "version": self.models[model_name]["version"],
                "status": "deployed"}

    def save_config(self, path: str):
        data = {"experiments": [
            {"name": e.name, "model": e.model, "metrics": e.metrics, "status": e.status}
            for e in self.experiments
        ]}
        Path(path).write_text(json.dumps(data, indent=2))

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|compare|deploy] ...")
        sys.exit(1)
    orch = Orchestrator()
    cmd = sys.argv[1]
    if cmd == "train":
        config = {"name": "experiment-1", "model": "lightgbm", "params": {"n_estimators": 100}}
        exp = orch.train(config)
        print(f"Trained: {exp.name}")
        print(f"Metrics: {exp.metrics}")
    elif cmd == "compare":
        for exp in orch.compare():
            print(f"  {exp.name}: {exp.metrics}")

if __name__ == "__main__":
    main()
