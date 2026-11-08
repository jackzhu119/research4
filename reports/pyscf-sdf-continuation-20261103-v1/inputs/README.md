# Software validation input manifests

These inputs preserve previously optimized water and ethanol coordinates. They are software controls, not herbal samples. No quantum calculation was run while creating the manifests.

- Use `validation_water_prepare_input.json`, SHA-256 `865877bc001ebca8a5a9fde48dbb16cd9a21504a14b435c23a276bfb6f5e9ded`.
- Use `validation_ethanol_prepare_input_v2.json`, SHA-256 `7c3ddf300ab6cb806a475a8e08c3a11944c3cc1e744332dab65ffbd52b638e07`.
- `validation_ethanol_prepare_input.json` is an **unexecuted, superseded metadata attempt**: its old total-result does not expose the exact optimized-coordinate SHA required by the frozen strict state worker. It must not be used.

The new ethanol metadata bridge copies actual original DONE optimizer/SCF/gradient/method evidence and adds the original B8 child's already-recorded full-precision geometry SHA. It explicitly reports `new_quantum_calculations=0` and states that DONE mirrors the old completed calculation; it is not a new optimization result. Both old files remain unchanged and are bound by SHA. The ethanol warmstart checkpoint is omitted because the original total result did not bind a checkpoint SHA. The new state phase therefore uses a fresh default SCF initial guess at the old optimized geometry.
