"""gridfm_graphkit CLI with this repository's model variants registered.

Same arguments as `gridfm_graphkit`:
  python -m gencodoe.graphkit_cli train --config configs/... --data_path data_out/datakit ...
"""

from gencodoe import genco_model  # noqa: F401  (registers GNS_heterogeneous_lv)
from gridfm_graphkit.__main__ import main

if __name__ == "__main__":
    main()
