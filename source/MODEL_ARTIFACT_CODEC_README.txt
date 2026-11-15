Lossless SVM model evidence storage

Files per encoded job
- fitted_artifacts.lossless.npz: every original array except support_vectors, plus exact support_training_rows.
- fitted_artifacts.recipe.json: input/worker checksums, original array shape/dtype/layout/byte-SHA256, original container provenance, reconstruction recipe and validation record.
- fitted_artifacts.npz: retained by default; --compact removes it only after completed/non-testing/no-active-attempt checks and successful on-disk all-array verification.

Required shared reconstruction dependencies (preserve and package with the jobs)
- inputs/current1672_paired612.npz, with the exact SHA recorded in each recipe.
- the frozen study_worker.py, exact SHA recorded in each recipe.
- Each recipe embeds the job/selection configuration. Saved scaler arrays, training record IDs, final indices, model coefficients, decisions and other original arrays remain in the lossless NPZ.
No model.fit, ranking, quantization or prediction is performed by the codec.

Python API
  from model_artifact_codec import encode_job, load_artifact, restore_job
  outcome = encode_job(job_dir, root=study_root, compact=True)
  arrays = load_artifact(job_dir, root=study_root)
  outcome = restore_job(job_dir, root=study_root)
  outcome = restore_job(job_dir, root=study_root, output_path='/absolute/restored.npz')
Optional progress_callback receives stage dictionaries; callers should throttle heartbeat I/O. CPU accounting belongs to the caller; the codec writes no global CPU ledger.

CLI (absolute paths recommended)
  python model_artifact_codec.py encode JOB_DIR --root STUDY_ROOT
  python model_artifact_codec.py encode JOB_DIR --root STUDY_ROOT --compact
  python model_artifact_codec.py verify JOB_DIR --root STUDY_ROOT
  python model_artifact_codec.py restore JOB_DIR --root STUDY_ROOT
  python model_artifact_codec.py restore JOB_DIR --root STUDY_ROOT --output RESTORED.npz

encode_job status: encoded / compacted / already_compact / skipped / error.
Default encoding preserves the original. A non-saving compact request returns skipped and keeps it. Failed checks leave the original in place. saved_bytes counts only bytes newly released by this call; equivalent_storage_reduction_bytes describes the compact layout. already_compact does not count a second release.

Exactness
Support vectors are matched by SHA256 followed by exact row-byte comparison against the frozen standardized training matrix. The codec replays only row-wise preprocessing, specified coverage transforms, stored scaling and final coordinate membership. Duplicate bitwise-equal training rows can share a row reference: this preserves the vector values but does not establish which identical physical specimen was the original SVM support observation.
All original arrays are restored with identical shape, dtype, memory-layout flag and C-order byte SHA256. An NPZ ZIP container restored by np.savez_compressed may have a different file SHA256 because member order/compression metadata can differ. The recipe retains the original ZIP SHA as provenance and does not claim to reproduce those packaging bytes.

Validation
model_artifact_codec_test_results.json records 21 real completed variants across full spectra, MC/drop, spacing, all coverage constructions, amplitude, context, label permutation and fixed-role materials. All original arrays passed independent byte comparisons. Two large examples additionally passed actual compact -> load -> restore -> byte verification -> compact cycles. No new model fits were performed.
