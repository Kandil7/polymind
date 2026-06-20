"""OpenTelemetry tracing setup — configures distributed tracing for PolyMind.

Provides:
- Tracer provider with OTLP export (Jaeger, Grafana Tempo)
- Auto-instrumentation for FastAPI
- Manual span helpers for graph nodes

Usage:
    from polymind.infrastructure.tracing import tracer, trace_span

    # Manual span
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # ... do work ...

    # Or use the convenience wrapper
    with trace_span("my_operation", {"key": "value"}):
        pass
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Tracer setup ─────────────────────────────────────────
_tracer = None


def _setup_tracer():
    """Initialize OpenTelemetry tracer."""
    global _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Determine exporter based on env
        exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "console")

        if exporter_type == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT", "localhost:4317")
            exporter = OTLPSpanExporter(endpoint=endpoint)
            logger.info("tracing.otlp_configured", endpoint=endpoint)
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            exporter = ConsoleSpanExporter()
            logger.info("tracing.console_configured")

        # Create provider
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Create tracer
        _tracer = trace.get_tracer("polymind")

        logger.info("tracing.initialized", exporter=exporter_type)
        return True

    except ImportError as e:
        logger.warning("tracing.import_failed", error=str(e))
        return False
    except Exception as e:
        logger.warning("tracing.setup_failed", error=str(e))
        return False


def get_tracer():
    """Get the configured tracer, initializing if needed."""
    global _tracer
    if _tracer is None:
        _setup_tracer()
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
):
    """Context manager for creating tracing spans.

    Args:
        name: Span name.
        attributes: Optional span attributes.

    Yields:
        The span object (or None if tracing unavailable).
    """
    tracer = get_tracer()

    if tracer is None:
        yield None
        return

    from opentelemetry.trace import StatusCode

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        else:
            span.set_status(StatusCode.OK)


def trace_graph_node(node_name: str, **attributes: Any):
    """Create a span for a LangGraph node execution.

    Args:
        node_name: Name of the graph node.
        **attributes: Additional span attributes.
    """
    return trace_span(f"graph.node.{node_name}", attributes)


def trace_retrieval(strategy: str, **attributes: Any):
    """Create a span for a retrieval operation.

    Args:
        strategy: Retrieval strategy used.
        **attributes: Additional span attributes.
    """
    return trace_span(f"retrieval.{strategy}", attributes)


def trace_llm_call(model: str, tier: str = "", **attributes: Any):
    """Create a span for an LLM API call.

    Args:
        model: Model name.
        tier: Model tier (fast, reasoning, etc.).
        **attributes: Additional span attributes.
    """
    attrs = {"llm.model": model, "llm.tier": tier}
    attrs.update(attributes)
    return trace_span("llm.call", attrs)
