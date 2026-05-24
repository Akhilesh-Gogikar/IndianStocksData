"""Unified multi-agent runtime for cross-product AI workflows."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from system.ai.vector_index import VectorIndexError, build_vector_index, search_vectors

DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_PROVIDER_REQUEST = "auto"
SUPPORTED_RETRIEVAL_MODES = {"rag", "sql", "hybrid"}
SUPPORTED_PROVIDER_REQUESTS = {"auto", "firebase-free-tier", "local-llama", "local-market-data"}

READ_ONLY_SQL_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
BLOCKED_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|ANALYZE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def build_unified_agentic_runtime(
    conn: sqlite3.Connection,
    index_dir: Path,
    product_id: str,
    objective: str,
    owner_id: str | None = None,
    retrieval_mode: str | None = None,
    provider_request: str | None = None,
    focus: str | None = None,
    sql_queries: list[str] | None = None,
    rag_query: str | None = None,
    run_id: int | None = None,
    source_name: str | None = None,
    evidence_limit: int = 8,
    max_rows_per_query: int = 25,
    include_deep_research: bool = True,
    include_documents: bool = True,
    auto_build_rag_index: bool = True,
    rag_build_limit: int = 1000,
    context: dict[str, Any] | None = None,
    deep_research_infer: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cleaned_product_id = (product_id or "").strip()
    cleaned_objective = (objective or "").strip()
    if not cleaned_product_id:
        raise ValueError("product_id_required")
    if not cleaned_objective:
        raise ValueError("objective_required")

    retrieval = _normalize_retrieval_mode(retrieval_mode)
    provider = _normalize_provider_request(provider_request)
    owner = (owner_id or "default").strip() or "default"
    normalized_focus = (focus or "").strip() or "cross-product intelligence synthesis"
    normalized_context = context or {}

    sql_payload = {
        "mode": "read-only",
        "queries": [],
        "count": 0,
    }
    rag_payload: dict[str, Any] = {
        "query": None,
        "count": 0,
        "results": [],
        "backend": None,
        "run_id": run_id,
        "notes": None,
    }
    data_gaps: list[str] = []

    if retrieval in {"sql", "hybrid"}:
        sql_payload = run_read_only_sql_queries(conn, sql_queries or [], max_rows_per_query)
        if not sql_payload["queries"]:
            data_gaps.append("No SQL queries were supplied for SQL retrieval.")

    if retrieval in {"rag", "hybrid"}:
        rag_payload = run_rag_retrieval(
            conn,
            index_dir,
            query=rag_query or f"{cleaned_product_id} {cleaned_objective} {normalized_focus}",
            run_id=run_id,
            source_name=source_name,
            evidence_limit=evidence_limit,
            auto_build=auto_build_rag_index,
            build_limit=rag_build_limit,
        )
        if int(rag_payload.get("count") or 0) == 0:
            data_gaps.append("No local semantic evidence matched the RAG query.")
        if rag_payload.get("notes"):
            data_gaps.append(str(rag_payload["notes"]))

    nodes = _agent_nodes(retrieval, include_deep_research, include_documents)
    evidence_summary = _evidence_summary(sql_payload, rag_payload)
    deep_research_prompt = _deep_research_prompt(
        cleaned_product_id,
        cleaned_objective,
        normalized_focus,
        retrieval,
        evidence_summary,
        normalized_context,
    )

    deep_research = {
        "status": "skipped",
        "provider": "none",
        "model": "deterministic-planner",
        "answer": _deterministic_summary(cleaned_product_id, cleaned_objective, evidence_summary),
        "usage": {},
    }
    if include_deep_research and deep_research_infer is not None:
        deep_research = deep_research_infer(deep_research_prompt)

    generated_documents = []
    if include_documents:
        generated_documents = _generated_documents(
            cleaned_product_id,
            owner,
            cleaned_objective,
            normalized_focus,
            retrieval,
            provider,
            evidence_summary,
            deep_research,
            nodes,
            data_gaps,
        )

    return {
        "kind": "unified_agentic_runtime",
        "status": "completed",
        "product_id": cleaned_product_id,
        "owner_id": owner,
        "objective": cleaned_objective,
        "focus": normalized_focus,
        "provider_request": provider,
        "retrieval_mode": retrieval,
        "agent_harness": {
            "nodes": nodes,
            "topology": "planner -> retrieval -> deep_research -> documenter",
            "policy": "Raw upstream files stay local; only processed payloads and retrieval evidence are used.",
        },
        "data_access": {
            "sql": sql_payload,
            "rag": rag_payload,
        },
        "deep_research_prompt": deep_research_prompt,
        "deep_research": deep_research,
        "generated_documents": generated_documents,
        "customer_next_actions": _next_actions(retrieval, deep_research, data_gaps),
        "data_gaps": data_gaps,
    }


def run_read_only_sql_queries(
    conn: sqlite3.Connection,
    sql_queries: list[str],
    max_rows_per_query: int,
) -> dict[str, Any]:
    limit = max(1, min(int(max_rows_per_query or 25), 200))
    results: list[dict[str, Any]] = []
    for raw_query in sql_queries:
        query = _validate_read_only_sql(raw_query)
        cursor = conn.execute(query)
        rows = cursor.fetchmany(limit + 1)
        columns = [item[0] for item in (cursor.description or [])]
        normalized_rows = [_normalize_sql_row(row, columns) for row in rows[:limit]]
        results.append(
            {
                "query": query,
                "columns": columns,
                "rows": normalized_rows,
                "row_count": len(normalized_rows),
                "truncated": len(rows) > limit,
            }
        )
    return {
        "mode": "read-only",
        "queries": results,
        "count": len(results),
        "max_rows_per_query": limit,
    }


def run_rag_retrieval(
    conn: sqlite3.Connection,
    index_dir: Path,
    query: str,
    run_id: int | None,
    source_name: str | None,
    evidence_limit: int,
    auto_build: bool,
    build_limit: int,
) -> dict[str, Any]:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("rag_query_required")

    try:
        payload = search_vectors(
            conn,
            cleaned_query,
            index_dir,
            run_id=run_id,
            source_name=source_name,
            k=max(1, evidence_limit),
        )
        return _compact_rag_payload(cleaned_query, payload)
    except VectorIndexError as exc:
        if not auto_build or "No vector index" not in str(exc):
            raise

    state = build_vector_index(
        conn,
        index_dir,
        run_id=run_id,
        source_name=source_name,
        limit=max(1, build_limit),
    )
    payload = search_vectors(
        conn,
        cleaned_query,
        index_dir,
        run_id=int(state["run_id"]),
        source_name=source_name,
        k=max(1, evidence_limit),
    )
    compact = _compact_rag_payload(cleaned_query, payload)
    compact["notes"] = "Vector index was built automatically before retrieval."
    return compact


def _normalize_retrieval_mode(value: str | None) -> str:
    normalized = (value or DEFAULT_RETRIEVAL_MODE).strip().lower()
    if normalized not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(f"unsupported_retrieval_mode:{normalized}")
    return normalized


def _normalize_provider_request(value: str | None) -> str:
    normalized = (value or DEFAULT_PROVIDER_REQUEST).strip().lower()
    if normalized not in SUPPORTED_PROVIDER_REQUESTS:
        raise ValueError(f"unsupported_provider_request:{normalized}")
    return normalized


def _validate_read_only_sql(value: str) -> str:
    query = (value or "").strip().rstrip(";")
    if not query:
        raise ValueError("empty_sql_query")
    if ";" in query:
        raise ValueError("multiple_sql_statements_not_allowed")
    if not READ_ONLY_SQL_START.match(query):
        raise ValueError("read_only_sql_required")
    if BLOCKED_SQL_KEYWORDS.search(query):
        raise ValueError("read_only_sql_required")
    return query


def _normalize_sql_row(row: sqlite3.Row | tuple[Any, ...], columns: list[str]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return {str(key): row[key] for key in row.keys()}
    if isinstance(row, tuple):
        return {columns[idx] if idx < len(columns) else f"column_{idx}": row[idx] for idx in range(len(row))}
    return {"value": row}


def _compact_rag_payload(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    results = []
    for item in payload.get("results", [])[:10]:
        preview = str(item.get("content_preview") or "").replace("\n", " ").strip()
        results.append(
            {
                "document_id": item.get("document_id"),
                "score": item.get("score"),
                "source_name": item.get("source_name"),
                "file_path": item.get("file_path"),
                "content_sha256": item.get("content_sha256"),
                "excerpt": preview[:400],
            }
        )
    return {
        "query": query,
        "run_id": payload.get("run_id"),
        "backend": payload.get("backend"),
        "count": int(payload.get("count") or 0),
        "results": results,
        "notes": payload.get("notes"),
    }


def _agent_nodes(retrieval_mode: str, include_deep_research: bool, include_documents: bool) -> list[dict[str, Any]]:
    nodes = [
        {
            "agent": "planner",
            "purpose": "Convert objective into executable multi-agent tasks across product surfaces.",
        },
        {
            "agent": "retrieval_router",
            "purpose": f"Run {retrieval_mode} retrieval with strict read-only SQL and local RAG evidence.",
        },
    ]
    if include_deep_research:
        nodes.append(
            {
                "agent": "deep_research",
                "purpose": "Synthesize evidence into grounded research using configured LLM providers.",
            }
        )
    if include_documents:
        nodes.append(
            {
                "agent": "documenter",
                "purpose": "Generate operator-ready markdown artifacts and evidence logs.",
            }
        )
    return nodes


def _evidence_summary(sql_payload: dict[str, Any], rag_payload: dict[str, Any]) -> dict[str, Any]:
    sql_queries = sql_payload.get("queries", [])
    sql_highlights = []
    for item in sql_queries[:3]:
        if item.get("rows"):
            sql_highlights.append(item["rows"][0])
    rag_results = rag_payload.get("results", [])
    rag_highlights = [
        {
            "document_id": item.get("document_id"),
            "source_name": item.get("source_name"),
            "score": item.get("score"),
            "excerpt": item.get("excerpt"),
        }
        for item in rag_results[:5]
    ]
    return {
        "sql_query_count": int(sql_payload.get("count") or 0),
        "sql_highlights": sql_highlights,
        "rag_count": int(rag_payload.get("count") or 0),
        "rag_highlights": rag_highlights,
    }


def _deep_research_prompt(
    product_id: str,
    objective: str,
    focus: str,
    retrieval_mode: str,
    evidence_summary: dict[str, Any],
    context: dict[str, Any],
) -> str:
    lines = [
        "You are a multi-agent research runtime for IndianStocksData.",
        "Stay non-advisory. Use only provided processed evidence. Explicitly flag data gaps.",
        f"Product ID: {product_id}",
        f"Objective: {objective}",
        f"Focus: {focus}",
        f"Retrieval mode: {retrieval_mode}",
        "",
        "SQL Highlights:",
        json.dumps(evidence_summary.get("sql_highlights", []), ensure_ascii=True, indent=2),
        "",
        "RAG Highlights:",
        json.dumps(evidence_summary.get("rag_highlights", []), ensure_ascii=True, indent=2),
    ]
    if context:
        lines.extend(["", "Additional Context:", json.dumps(context, ensure_ascii=True, indent=2)])
    lines.extend(
        [
            "",
            "Return:",
            "1) concise findings",
            "2) risks and unknowns",
            "3) suggested next checks",
        ]
    )
    return "\n".join(lines)


def _deterministic_summary(product_id: str, objective: str, evidence_summary: dict[str, Any]) -> str:
    sql_count = int(evidence_summary.get("sql_query_count") or 0)
    rag_count = int(evidence_summary.get("rag_count") or 0)
    return (
        f"Prepared deterministic multi-agent synthesis for {product_id}. "
        f"Objective: {objective}. SQL queries executed: {sql_count}. "
        f"RAG evidence items: {rag_count}. Use a configured LLM provider for narrative deep research."
    )


def _generated_documents(
    product_id: str,
    owner_id: str,
    objective: str,
    focus: str,
    retrieval_mode: str,
    provider_request: str,
    evidence_summary: dict[str, Any],
    deep_research: dict[str, Any],
    nodes: list[dict[str, Any]],
    data_gaps: list[str],
) -> list[dict[str, Any]]:
    execution = _execution_brief_markdown(
        product_id,
        owner_id,
        objective,
        focus,
        retrieval_mode,
        provider_request,
        deep_research,
        nodes,
        data_gaps,
    )
    evidence = _evidence_log_markdown(evidence_summary, data_gaps)
    return [
        {
            "name": "agentic-execution-brief.md",
            "title": "Agentic Execution Brief",
            "format": "markdown",
            "content": execution,
        },
        {
            "name": "agentic-evidence-log.md",
            "title": "Agentic Evidence Log",
            "format": "markdown",
            "content": evidence,
        },
    ]


def _execution_brief_markdown(
    product_id: str,
    owner_id: str,
    objective: str,
    focus: str,
    retrieval_mode: str,
    provider_request: str,
    deep_research: dict[str, Any],
    nodes: list[dict[str, Any]],
    data_gaps: list[str],
) -> str:
    lines = [
        f"# Unified Agentic Execution Brief: {product_id}",
        "",
        f"- Owner: {owner_id}",
        f"- Objective: {objective}",
        f"- Focus: {focus}",
        f"- Retrieval mode: {retrieval_mode}",
        f"- Provider request: {provider_request}",
        f"- Provider used: {deep_research.get('provider')}",
        f"- Model used: {deep_research.get('model')}",
        "",
        "## Multi-Agent Harness",
    ]
    lines.extend(f"- {node['agent']}: {node['purpose']}" for node in nodes)
    lines.extend(["", "## Deep Research Output", deep_research.get("answer", "No deep research output."), ""])
    lines.append("## Data Gaps")
    if data_gaps:
        lines.extend(f"- {item}" for item in data_gaps)
    else:
        lines.append("- None detected.")
    return "\n".join(lines)


def _evidence_log_markdown(evidence_summary: dict[str, Any], data_gaps: list[str]) -> str:
    lines = [
        "# Agentic Evidence Log",
        "",
        f"- SQL query count: {evidence_summary.get('sql_query_count')}",
        f"- RAG evidence count: {evidence_summary.get('rag_count')}",
        "",
        "## SQL Highlights",
    ]
    sql_highlights = evidence_summary.get("sql_highlights", [])
    if sql_highlights:
        for item in sql_highlights:
            lines.append(f"- {json.dumps(item, ensure_ascii=True)}")
    else:
        lines.append("- No SQL highlights available.")
    lines.extend(["", "## RAG Highlights"])
    rag_highlights = evidence_summary.get("rag_highlights", [])
    if rag_highlights:
        for item in rag_highlights:
            lines.append(
                f"- doc={item.get('document_id')} source={item.get('source_name')} score={item.get('score')}: {item.get('excerpt')}"
            )
    else:
        lines.append("- No RAG highlights available.")
    lines.extend(["", "## Gaps"])
    if data_gaps:
        lines.extend(f"- {item}" for item in data_gaps)
    else:
        lines.append("- None detected.")
    return "\n".join(lines)


def _next_actions(retrieval_mode: str, deep_research: dict[str, Any], data_gaps: list[str]) -> list[str]:
    actions = [
        f"Keep retrieval mode `{retrieval_mode}` tied to read-only evidence paths.",
        "Use generated markdown artifacts for product-level reviews and handoffs.",
        "Verify claims against cited SQL rows and RAG excerpts before external sharing.",
    ]
    if deep_research.get("provider") in {"local-market-data", "none"}:
        actions.insert(0, "Bring Firebase free-tier or local llama runtime online for richer deep research output.")
    if data_gaps:
        actions.insert(0, "Resolve listed data gaps before using this output for customer-facing workflows.")
    return actions
