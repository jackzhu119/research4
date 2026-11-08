# Research4: sealed-result backfill while training continues

This independent backup branch contains previously untransmitted **closed**
research outputs from `research/species-relaunch-90000-20261108`. It neither
changes that branch's running jobs nor updates main, raw data, labels, model
source, CPU ledgers or its active Git index. The independent local repository
has a separate index, objects and refs. No scientific process is signalled.

The selected source is completed backup point **00575**. The immutable source
backup receipt, supervisor seal and its previously verified research commit
are in `evidence/`. `manifest.json` lists every original path, byte count and
SHA-256. It also lists the compressed transport parts and their SHA-256.
These are real saved results, not new experiments or newly generated metrics.

The source backlog mainly contains redundant, very large historical summary
JSONs and compressed prediction records. They are archived **without changing
their bytes**, using a streaming tar.gz divided into parts of at most32MiB.
Existing files and later results in the live research branch are not deleted
or replaced. This snapshot does not claim to back up jobs finished after its
source cutoff. Newer automatic backups continue independently.

`evidence/LATEST_BACKGROUND_RESULTS.md` is a readable copy of the sealed source
summary. Its claims remain TRAIN-only internal development validation. The old
mixed-kind96.0234% must not be presented as pure-acid origin performance.

## Verify or restore

Clone or download this branch, then run with standard-library Python:

```bash
python restore_backfill.py --manifest manifest.json --verify-only
python restore_backfill.py --manifest manifest.json --output /path/to/empty-restore-directory
```

Keep all parts in `archives/`. The tool verifies all part hashes first, then
the original per-file hashes, sizes, names and gzip integrity. It rejects
unsafe paths, links, unexpected/duplicate members, nonempty restore targets
and overwrites. No full-size JSON needs to be loaded into memory.

The archive is one continuous compressed stream. Individual parts are not
standalone gzip files; the restore tool concatenates them in manifest order.
Restored paths are repository-relative and can be inspected in an empty
directory before being copied into an appropriate checkout.

Local round-trip validation is saved in
`evidence/local_restore_verification.json`. Pushes use bounded batches and
verify the remote branch commit after every batch; only the final verified
receipt indicates the complete selected snapshot reached GitHub.

This finite backfill uses low CPU/IO priority and at most600CPU core-seconds
from the existing unused1,900-second preparation allocation. The original
88,000-second training ceiling and100-second monitor allocation stay fixed.
