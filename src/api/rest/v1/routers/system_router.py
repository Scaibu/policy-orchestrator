"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 SYSTEM ROUTER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Encapsulates HTTP REST endpoints for system-level capabilities:
   - System Health Probe (/health)
   - Knowledge Base & Grounded Hybrid Search (/rag/index, /rag/search)
   - Autonomous AI Policy Agents (/agent/run, /agents, /agents/{agent_id}/run)
   - Repository Invariant Auditing (/audit/scan)
   - Policy Knowledge Graph (/graph/build, /graph/query, /graph/impact/{rule_id})
   - Algorithm Catalog & Composition (/algos/contracts, /algos/adapters, /algos/compose, /algos/execute/{algo_id})

2. ARCHITECTURAL LAYOUT & DESIGN PILLARS:
   - Zero-Inline-Comment Doctrine: Docblock serves as full architectural specification.
   - Hexagonal Isolation: Handlers delegate directly to domain services via dependency injection.
   - Strict Protocol Envelope: All handlers return success/error envelopes with trace IDs.
===============================================================================
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from src.api.rest.envelope import build_success_envelope, build_error_envelope
from ..dependencies import get_services, get_code_engine_service
class RAGSearchRequestDTO(BaseModel):
    query: str = Field(..., description="Query string")
    top_k: int = Field(default=5, description="Top-k documents")
    category_filter: Optional[str] = Field(default=None)
    min_score: float = Field(default=0.0)

class IndexingStatusDTO(BaseModel):
    status: str = "UNAVAILABLE"

class AgentRunRequestDTO(BaseModel):
    prompt: str = Field(..., description="Task prompt")
    target_directory: str = Field(default=".")
    max_steps: int = Field(default=10)
    temperature: float = Field(default=0.0)
    session_id: Optional[str] = Field(default=None)
from src.domain.ports.agent_manifest_port import AgentRole
from src.domain.models.algorithm_contract import (
    AlgorithmCategory,
    Purity,
    SideEffectScope,
    AlgorithmContract,
    TypeAdapterContract,
)

router = APIRouter()

class GraphQueryDTO(BaseModel):
    query: str = Field(..., description="Cypher or pattern matching query")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")

class AlgoComposeRequestDTO(BaseModel):
    algo_ids: List[str] = Field(..., description="Ordered list of algorithm IDs to compose")
    strict_check: bool = Field(default=True, description="Enforce strict contract safety checks")

class AlgoExecuteRequestDTO(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Algorithm input parameters")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Optional algorithm execution parameters")

@router.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id") or request.headers.get("traceparent")
    return build_success_envelope(
        data={"status": "healthy", "service": "policy-orchestrator", "version": "0.1.0"},
        trace_id=trace_id,
    )


@router.post("/rag/index")
def index_policies(request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    count = svcs["rag"].index_all_rules()
    return build_success_envelope(
        data={"indexed_documents": count, "status": "COMPLETED"},
        trace_id=trace_id,
    )


@router.post("/rag/search")
def search_policies(payload: RAGSearchRequestDTO, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    rag_req = RAGQueryRequest(
        query=payload.query,
        top_k=payload.top_k,
        category_filter=payload.category_filter,
        min_relevance_score=payload.min_score,
    )
    res = svcs["rag"].retrieve_context(rag_req)
    docs_data = [
        {
            "id": d.id,
            "source_file": d.source_file,
            "section_title": d.section_title,
            "content": d.content,
            "category": d.category,
            "rrf_score": d.rrf_score,
            "metadata": d.metadata,
        }
        for d in res.documents
    ]
    return build_success_envelope(
        data={
            "query": res.query,
            "total_found": res.total_found,
            "documents": docs_data,
            "formatted_context_block": res.formatted_context_block,
            "latency_ms": res.latency_ms,
        },
        trace_id=trace_id,
    )


@router.post("/agent/run")
def run_agent(payload: AgentRunRequestDTO, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    agent_req = AgentExecutionRequest(
        prompt=payload.prompt,
        target_directory=payload.target_directory,
        max_steps=payload.max_steps,
        temperature=payload.temperature,
        session_id=payload.session_id,
    )
    result = svcs["agent"].execute_agent_loop(agent_req)
    steps_data = [
        {
            "step_number": s.step_number,
            "thought": s.thought,
            "action": s.action,
            "observation": s.observation,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "output": tc.output,
                    "duration_ms": tc.duration_ms,
                    "status": tc.status,
                }
                for tc in s.tool_calls
            ],
        }
        for s in result.steps
    ]
    return build_success_envelope(
        data={
            "session_id": result.session_id,
            "status": result.status,
            "final_response": result.final_response,
            "steps": steps_data,
            "total_steps": result.total_steps,
            "total_tokens": result.total_tokens,
            "grounded_sources": result.grounded_sources,
            "duration_ms": result.duration_ms,
        },
        trace_id=trace_id,
    )


@router.get("/agents")
def list_declarative_agents(request: Request, category: Optional[str] = None) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    manifests = svcs["agent_registry"].list_manifests(category=category)
    return build_success_envelope(
        data={
            "total_agents": len(manifests),
            "agents": [
                {
                    "agent_id": m.agent_id,
                    "name": m.name,
                    "role": m.role.value,
                    "category": m.category,
                    "description": m.description,
                    "algorithms": m.algorithms,
                    "allowed_tools": m.allowed_tools,
                    "tags": m.tags,
                }
                for m in manifests
            ],
        },
        trace_id=trace_id,
    )


@router.post("/agents/{agent_id}/run")
def run_specialized_agent(agent_id: str, payload: AgentRunRequestDTO, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    manifest = svcs["agent_registry"].get_manifest(agent_id)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Specialized Agent '{agent_id}' not found.")

    agent_req = AgentExecutionRequest(
        prompt=payload.prompt,
        target_directory=payload.target_directory,
        max_steps=payload.max_steps,
        temperature=payload.temperature,
        session_id=payload.session_id,
    )
    result = svcs["agent"].execute_agent_loop(agent_req, manifest=manifest)
    steps_data = [
        {
            "step_number": s.step_number,
            "thought": s.thought,
            "action": s.action,
            "observation": s.observation,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "output": tc.output,
                    "duration_ms": tc.duration_ms,
                    "status": tc.status,
                }
                for tc in s.tool_calls
            ],
        }
        for s in result.steps
    ]
    return build_success_envelope(
        data={
            "agent_id": agent_id,
            "agent_name": manifest.name,
            "session_id": result.session_id,
            "status": result.status,
            "final_response": result.final_response,
            "steps": steps_data,
            "total_steps": result.total_steps,
            "total_tokens": result.total_tokens,
            "grounded_sources": result.grounded_sources,
            "duration_ms": result.duration_ms,
        },
        trace_id=trace_id,
    )


@router.post("/audit/scan")
def scan_repository(request: Request, target_directory: str = ".") -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    findings = svcs["audit"].audit_repository(target_directory)
    findings_data = [
        {
            "rule_id": f.rule_id,
            "category": f.category,
            "severity": f.severity,
            "description": f.description,
            "file": f.file,
            "line": f.line,
            "snippet": f.snippet,
            "recommendation": f.recommendation,
        }
        for f in findings
    ]
    return build_success_envelope(
        data={"total_findings": len(findings), "findings": findings_data},
        trace_id=trace_id,
    )

@router.post("/algos/execute/{algo_id}")
def execute_algorithm_direct(algo_id: str, payload: AlgoExecuteRequestDTO, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svc = get_code_engine_service()
    try:
        res = svc.execute_algorithm(algo_id=algo_id, inputs=payload.inputs, parameters=payload.parameters)
        return build_success_envelope(data={"algo_id": algo_id, "result": res}, trace_id=trace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/build")
def build_graph(request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    summary = svcs["graph"].build_graph_from_rules()
    return build_success_envelope(data=summary, trace_id=trace_id)


@router.post("/graph/query")
def query_graph(payload: GraphQueryDTO, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    res = svcs["graph"].query_cypher(payload.query, payload.parameters)
    return build_success_envelope(
        data={
            "records": res.records,
            "nodes_count": len(res.nodes),
            "relationships_count": len(res.relationships),
        },
        trace_id=trace_id,
    )


@router.get("/graph/impact/{rule_id}")
def get_rule_impact(rule_id: str, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    impact = svcs["graph"].get_rule_impact(rule_id)
    return build_success_envelope(data={"rule_id": rule_id, "impact": impact}, trace_id=trace_id)


@router.get("/algos/contracts")
def list_algorithm_contracts(
    request: Request,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    purity: Optional[str] = None,
    side_effects: Optional[str] = None,
) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    
    cat_enum = AlgorithmCategory(category) if category else None
    purity_enum = Purity(purity) if purity else None
    se_enum = SideEffectScope(side_effects) if side_effects else None
    tags_filter = [tag] if tag else None

    contracts = svcs["algo_registry"].list_algorithms(
        category=cat_enum,
        tags=tags_filter,
        purity=purity_enum,
        side_effects=se_enum,
    )
    return build_success_envelope(
        data={"total_contracts": len(contracts), "contracts": [c.model_dump() for c in contracts]},
        trace_id=trace_id,
    )


@router.get("/algos/contracts/{algo_id}")
def get_algorithm_contract(algo_id: str, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    contract = svcs["algo_registry"].get_algorithm(algo_id)
    if not contract:
        raise HTTPException(status_code=404, detail=f"Algorithm contract '{algo_id}' not found.")
    return build_success_envelope(data=contract.model_dump(), trace_id=trace_id)


@router.get("/algos/adapters")
def list_type_adapters(request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    adapters = svcs["algo_registry"].list_adapters()
    return build_success_envelope(
        data={"total_adapters": len(adapters), "adapters": [a.model_dump() for a in adapters]},
        trace_id=trace_id,
    )


@router.post("/algos/compose")
def compose_algorithm_pipeline(payload: AlgoComposeRequestDTO, request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id")
    svcs = get_services()
    try:
        plan = svcs["composer"].compose_pipeline(
            algo_ids=payload.algo_ids,
            strict_contract_check=payload.strict_check,
        )
        return build_success_envelope(data=plan, trace_id=trace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
