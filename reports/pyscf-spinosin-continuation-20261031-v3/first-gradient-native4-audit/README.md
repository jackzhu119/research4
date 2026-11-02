# First-gradient native1/native4 audit

Status: PASS. Fixed thresholds and all225 gradient components/75 atom norms are recorded in comparison.json and CSV.

Only completed first points are compared. Treat them as matching the same geometry only when both coordinate checks pass; no new conformers are generated here. These checks do not establish completed optimization/frequency results. Observed full-gradient kernel wall ratios include XC and J, are affected by scheduling and concurrent tests, and are not a hardware-independent benchmark or a forecast of Hessian speed. No quantum computation or checkpoint read occurs in this audit.
