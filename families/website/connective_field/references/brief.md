# Brief: a connective field (website clip)

For the Knapen Lab website: the section "From brain to brain" of the page on computational models, linked
from the page on topographic maps. Proposed in the website review of 13 November 2026 and approved by the
site's owner.

Every step of pRF modelling has a clip; the method behind the lab's findings in the cerebellum, the
default network, the hippocampus and the body maps is one paragraph of text. Show it: a connective field
model explains a voxel's activity from a small Gaussian patch of activity on a source map, here V1, and the
voxel inherits the visual field position of that patch. No description of the stimulus is needed, only
activity on the source map.

Left: the model sheet of V1 from the clip `fields_to_map`, one square per voxel, coloured by distance from
fixation, with the candidate patch on it. Right: the target voxel's time course with the patch's prediction
and R squared; below it the visual field position under the patch centre. The patch tries positions and
sizes (grid search), settles on the best, and is refined. At the end the connective field the target was
simulated with is shown for comparison.

Everything is simulated with prfmodel (sim/simulate_cf.py) and is labelled so. Silent, 15 seconds, 2:1,
light and dark renders.
