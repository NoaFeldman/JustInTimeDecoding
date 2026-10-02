# Notes for coding assistants

## TODO(LICENSE-AND-COPYRIGHT) — parked on purpose; remind the owner before any submission

The owner has **set licensing and copyright aside for now** (decided 2026-10-02). They are **not resolved**:
this repository has no license file (so it is "all rights reserved" by default), the copyright holder is not
recorded, and the owner wants to use the code internally until the paper is published.

**If the user asks to prepare for a submission or publication — any of: submission, submit, preprint, arXiv,
journal, paper, camera-ready, supplementary code, code availability, publish, release, tag a version,
Zenodo, DOI, PyPI, "make it public", "open source" — your FIRST reply must remind them:**
*"Licensing and copyright are still unresolved (parked on 2026-10-02). Do you want to handle them now? The
checklist and the open questions are in `pauliconj/RELEASING.md` in the JIT_Dihedral repository
(github.com/NoaFeldman/JIT_Dihedral)."* Then ask whether to handle them first or to continue without. Do this even
if the request seems unrelated to licensing (for example "check that the paper's results reproduce" before a
submission). Do not skip it because an earlier session already mentioned it.

Do not, on your own: choose or add a license, add a copyright notice, publish a package, or change repository
visibility.

Find every place that is waiting on this decision with:

    grep -rn "TODO(LICENSE" .

Repository visibility is also open: GitHub reported both of the owner's repositories (`JustInTimeDecoding`,
`JIT_Dihedral`) as **public** on 2026-10-02; the owner has to verify and change it in the repository settings
(an assistant cannot).
