# Simulation behind the clip

The curves in `data/disc_growth.csv` are written by `sim/simulate_dn.py` (repository root) with prfmodel 0.2.0
on the JAX backend of Keras. The response is the output of `DivNormGaussian2DPRFModel` without an impulse
response or scaling, for a disc centred on the pRF whose radius grows from 0 to 6 deg in 0.05 deg steps
inside a circular aperture of 8 deg radius. Numerator `a (G1 . S) + b` and denominator `c (G2 . S) + d`
come from the same model's two Gaussian profiles encoded with the stimulus; the script checks that
numerator / denominator - b / d equals the model's output.

The stimulus design is each grid cell's fraction inside the stimulus (0.05 deg cells) multiplied by the cell area, so `G . S`
is the integral of the volume-normalized Gaussian over the stimulus.

Parameters (`data/parameters.json`) are illustrative, not fitted: pRF centre (1.5, 1.0) deg, sigma_1 = 1 deg,
sigma_2 = 4 deg, a = 0.6, b = 0.05, c = 1, d = 0.25.

Besides the reference parameters (`sweep = base`), the table holds the curves for d from 0.25 to 1.5 in
steps of 0.05 with the other parameters at their reference values (`sweep = d`), and for sigma_2 / sigma_1
from 1 to 4 in steps of 0.1, changing sigma_2 with sigma_1 fixed (`sweep = ratio`). The clip interpolates
linearly between neighbouring simulated values.
