"""gridfm_graphkit CLI with this repository's model variants registered.

Same arguments as `gridfm_graphkit`:
  python -m gencodoe.graphkit_cli train --config configs/... --data_path data_out/datakit ...

Also fixes graphkit 0.9.0.3's SaveBestModelStateDict, which keeps the monitored
metric *tensor* as its best score. When Lightning updates that tensor in place,
the best score follows every new value, `current < best` is never true again,
and artifacts/model/best_model_state_dict.pt stays at an early epoch (seen in
two of our runs: export from epoch 0 although validation loss fell every
epoch). We store the best score as a float right after each save.
"""

from gencodoe import genco_model  # noqa: F401  (registers GNS_heterogeneous_lv)
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict

_on_validation_end = SaveBestModelStateDict.on_validation_end


def _on_validation_end_float_best(self, trainer, pl_module):
    if hasattr(self.best_score, "item"):
        self.best_score = float(self.best_score)
    _on_validation_end(self, trainer, pl_module)
    if hasattr(self.best_score, "item"):
        self.best_score = float(self.best_score)


SaveBestModelStateDict.on_validation_end = _on_validation_end_float_best

if __name__ == "__main__":
    from gridfm_graphkit.__main__ import main

    main()
