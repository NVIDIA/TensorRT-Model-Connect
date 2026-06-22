"""Reference backends — reference inference for comparison.

Each module in this package should expose a module-level ``plugin`` attribute
that is an instance implementing the ReferenceBackendRunner protocol. The
registry auto-discovers these plugins on first access.
"""
