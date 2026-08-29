### Data

This folder contains the AERMOD-generated training data used by RAPID. Each pollutant subfolder contains compressed NumPy (`.npz`) data shards organized by facility scenario, along with CSV metadata describing each shard’s scenario, date and time, variables, dimensions, and concentration range.

Load a shard with:

```python
import numpy as np

data = np.load("path/to/shard.npz")
print(data.files)
```
