# Checkpoints

Only one selected checkpoint is intended for publication:

- `fabricnir_best.pt`

Other training checkpoints and intermediate models are intentionally excluded
from version control.

This checkpoint was selected from the available local experiment artifacts. When
loading it, use the matching model architecture from the experiment that produced
the checkpoint; do not assume it can be loaded into every ablation variant.

Release note: `fabricnir_best.pt` is the only model weight file intended for the
public repository. It was cleaned from the selected best local checkpoint by
removing obsolete `_fixed_decoder.*` compatibility keys; all current model
parameters are unchanged from the selected checkpoint.
