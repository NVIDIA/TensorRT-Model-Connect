# SAM2 L4 qualification artifacts

The checked-in record and audit manifest are public, non-authorizing evidence
for one exact TensorRT bundle. They do not qualify a newly rebuilt bundle, even
when it uses the same checkpoint, source, GPU model, and TensorRT version.

The qualified bundle is exactly 400,409,218 bytes with SHA-256
`1c520fb705226156258c68909475a6a04ce8e4c85fc84a389451c1f8b956fa1c`.
It must be provisioned together with the matching record through an approved
artifact or model registry. The repository and wheel deliberately do not embed
the 400 MB bundle.

The production C API requires the caller to pass the external record path
explicitly; it never infers a sidecar. A production pin must remain inactive
until the exact bundle has a stable distribution location and a consumer smoke
test has loaded that distributed byte sequence with the checked-in record.

Rebuilding the bundle changes TensorRT plan hashes and requires a new Q3,
W3/N100/Q1 run, qualification record, audit manifest, and reviewed pin.
