from __future__ import annotations


class UAVBenchmarkError(RuntimeError):
    """Base class for benchmark-domain failures."""


class OptionalDependencyUnavailable(UAVBenchmarkError):
    """Raised when an optional dependency is required but unavailable."""


class ModelValidationError(ValueError):
    """Raised when a terrain or mission model violates the benchmark contract."""


class BenchmarkExecutionError(UAVBenchmarkError):
    """Raised when a benchmark task cannot be executed safely."""


class ArtifactReadError(UAVBenchmarkError):
    """Raised when a benchmark artifact cannot be read or interpreted."""


class ExternalToolError(UAVBenchmarkError):
    """Raised when an external tool reports a failure."""
