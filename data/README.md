# Data

Raw data is downloaded locally and ignored by Git. The download script writes a manifest beside each split with the upstream dataset source and license information.

The current V1 uses:

- WikiText-2 raw, published under CC BY-SA 4.0.
- AG News, whose upstream card describes research and non-commercial use but does not state a standard license. Verify those terms before reusing the target-domain experiment for another purpose.

Validation documents are deduplicated against the training split before they are written. The raw files are not redistributed by this repository.
