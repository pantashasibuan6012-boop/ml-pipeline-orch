# ML Pipeline Orchestrator

Automated machine learning pipeline orchestration for training, tuning, and deployment.

## Features
- Multi-framework support (PyTorch, TensorFlow, scikit-learn)
- Automatic hyperparameter tuning
- Experiment tracking with W&B integration
- Model versioning and registry
- A/B testing infrastructure

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python main.py train --config pipeline.yaml
python main.py deploy --model v2 --stage production
```

## License
MIT