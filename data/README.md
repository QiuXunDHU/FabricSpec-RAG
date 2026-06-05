# Data

This repository includes only small sample workbooks for reproducibility and
format inspection:

- `train.xlsx`: 12 sample rows
- `valid.xlsx`: 6 sample rows

The full raw dataset is intentionally not included. Each workbook must contain a
`Labels` column and numeric wavelength columns such as `950`, `955`, `960`, ...
Labels use compact component-percentage strings such as `C93P7` or `P52N48`.

To train with the full dataset, place the complete files at:

- `data/train.xlsx`
- `data/valid.xlsx`
