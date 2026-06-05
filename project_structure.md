# Project Structure

```text
FabricNIR/
|-- configs/
|   |-- base_config.yaml
|   |-- fabricnir_best.yaml
|   |-- experiment_matrix.yaml
|   `-- ablation/
|-- data/
|   |-- README.md
|   |-- train.xlsx
|   `-- valid.xlsx
|-- docs/
|   |-- assets/
|   `-- open_source_audit.md
|-- checkpoints/
|   |-- README.md
|   `-- fabricnir_best.pt
|-- experiments/
|-- fabric_nir/
|   |-- data/
|   |-- metrics/
|   |-- models/
|   |-- tokenizers/
|   |-- train/
|   |-- utils/
|   `-- visualization/
|-- scripts/
|-- main.py
|-- README.md
|-- README.en.md
`-- requirements.txt
```

Generated outputs are written to `results/` and ignored by version control.
Full datasets, intermediate checkpoints, installers, and local project files are
kept outside the publishable tree through `.gitignore`.
