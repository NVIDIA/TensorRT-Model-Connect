"""Strategy runners — TRT inference execution for each task strategy.

Each module in this package should expose a module-level ``plugin`` attribute
that is an instance implementing the TaskStrategyRunner protocol. The registry
auto-discovers these plugins on first access.
"""
