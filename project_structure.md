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
|   |-- open_source_audit.md
|   `-- readme_assets/
|-- results/
|   |-- .gitkeep
|   |-- best/
|       |-- metrics/
|   |   `-- visualizations/
|   `-- multi_task_gru_random_multicov_nocbam/
|       `-- visualizations/
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

Generated outputs are written to `results/`. Only the selected public result
artifacts under `results/best/` and
`results/multi_task_gru_random_multicov_nocbam/` are tracked; other generated
outputs are ignored by version control.
Full datasets, intermediate checkpoints, installers, and local project files are
kept outside the publishable tree through `.gitignore`.
