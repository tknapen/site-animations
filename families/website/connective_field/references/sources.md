# Sources

Haak KV, Winawer J, Harvey BM, Renken R, Dumoulin SO, Wandell BA, Cornelissen FW (2013). Connective field
modeling. NeuroImage 66, 376-384. doi:10.1016/j.neuroimage.2012.10.037 (title, journal, volume and pages
checked on Crossref, 13 November 2026). The model: a voxel's time course as the Gaussian-weighted sum of the
time courses on a source surface, the Gaussian defined over distance along that surface, with a centre and
a size.

Knapen T (2021). Topographic connectivity reveals task-dependent retinotopic processing throughout the
human brain. PNAS 118(2), e2017032118. doi:10.1073/pnas.2017032118. The lab's use of the method with V1 as
the source map, during movie watching, rest and retinotopic mapping.

prfmodel 0.2.0 (github.com/popylar-org/prfmodel): `GaussianCFModel`, whose parameters are the index of the
centre unit on the source, the size of the Gaussian, an amplitude and a baseline. The centre is found by
grid search; the model's documentation cites Haak et al. (2013).
