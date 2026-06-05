# Open-Source Readiness Audit

This document records issues found during the pre-release cleanup, their root
causes, and the fixes applied.

| Area | Issue | Root cause | Fix |
| --- | --- | --- | --- |
| Public data | Full raw train/valid workbooks were originally in `data/`. | Research workspace contained private/full artifacts in publishable paths. | Moved full workbooks to `_private/data_full/`; kept only small sample workbooks in `data/`; added `.gitignore` safeguards. |
| Public checkpoint | Multiple model files existed under generated results. | Training outputs and release assets were mixed. | Moved non-public checkpoints to `_private/`; kept only `checkpoints/fabricnir_best.pt`. |
| Checkpoint loading | Initial selected checkpoint failed strict loading. | Checkpoint contained historical `_fixed_decoder.*` keys from debugging/fallback modules. | Cleaned the public checkpoint by removing legacy keys; current `fabricnir_best.pt` strict-loads with `configs/fabricnir_best.yaml`. |
| Checkpoint config | Default config is multi-task Transformer, but public checkpoint is single-task GRU. | Default experiment config was not the architecture that produced the selected checkpoint. | Added `configs/fabricnir_best.yaml`; README now tells users to load the checkpoint with that config. |
| Pretrained ablation config | `pretrained.yaml` pointed to an unpublished `results/pretrained/...` file. | Private/generated pretraining artifacts were referenced from public config. | Set `weights_path: null` and documented that users must generate or provide pretraining weights. |
| CBAM dimensions | Checkpoint CBAM weights had shape based on `feature_map_dim=42`, while model initialized CBAM with `out_channels=128`. | CBAM forward permutes features and attends over feature-map dimension, but constructor used backbone channel count. | Initialize CBAM with `feature_map_dim` in both single-task and multi-task Seq2Seq models. |
| Transformer decoder | Default multi-task smoke test failed with embedding-dimension mismatch. | Transformer encoder received unprojected feature tensors and an unnecessary `permute` despite `batch_first=True`. | Project features to hidden size before the batch-first encoder. |
| Positional encoding | Transformer smoke test failed on batch-first tensors. | Positional encoding buffer used sequence-first shape. | Changed positional encoding to `[1, max_len, d_model]` and slice on sequence dimension. |
| Multi-task regression head | Regression head received vocabulary logits rather than hidden states. | Current decoder returns logits, but regression head expected `hidden_size`. | Set regression head input dimension to `vocab_size` for current decoder contract. |
| Embedding visualization | Training crashed during t-SNE visualization. | `get_embeddings()` added an extra dimension before CBAM. | Use the same 3D feature tensor shape as model forward. |
| Beam search | Checkpoint test crashed with 4D GRU input. | Beam search added `unsqueeze(1)` to already-3D feature tensors. | Pass feature tensors directly to the decoder. |
| Beam search return contract | Beam search assumed decoder returns `(output, hidden)`. | Decoder implementations return either a tensor or a tuple depending on variant. | Added tuple/tensor compatibility handling. |
| Legacy fixed decoder | Main single-task model still created a runtime `_fixed_decoder` and kept a duplicate decoder implementation. | Historical shape-debug workaround remained after the canonical decoder was fixed. | Removed the runtime fallback path and the unused `decoders_fixed.py`; public checkpoint was cleaned, while trainer loading remains permissive for older local checkpoints. |
| Tokenizer decoding | Decode failed for nested numpy arrays and ambiguous truth checks. | `decode()` expected flat Python lists and used `if percentages` on array-like values. | Flatten tensor/numpy/list inputs and use `percentages is not None`. |
| README scope | README implied the minimal code release included full KG/RAG components. | Paper method scope and public code release scope were conflated. | Added explicit scope note: this release focuses on NIR Seq2Seq components; full KG/RAG resources are not bundled. |
| README language | README was English-only after cleanup. | Initial rewrite optimized for release clarity but not bilingual presentation. | Added Chinese `README.md` and English `README.en.md`, both with figures and citation. |
| Citation | Repository lacked standard citation metadata. | No `CITATION.cff` existed. | Added `CITATION.cff` and BibTeX entries in both README files. |
| Generated artifacts | `results/`, `__pycache__/`, IDE files, and `.codegraph/` could be accidentally committed. | Local analysis/training outputs live in the workspace. | Added `.gitignore`; kept `.codegraph/` locally but ignored; moved prior results to `_private/`. |
| Visual assets | README did not show representative results in the expected result directory. | Existing best visualizations were under private results, while the first public copy used `docs/assets/`. | Copied the full selected best visualization set and metrics to `results/best/`; README files now reference those result artifacts directly. |

## Remaining Notes

- The public sample data is intentionally tiny; metrics from smoke tests are only
  functional checks and should not be interpreted as paper-level performance.
- Internal shape and generation debugging scripts were moved to `_private/scripts/`
  and are not part of the public release.
- The DOI should be added to `README.md`, `README.en.md`, and `CITATION.cff`
  once it is available.
