#!/usr/bin/env python3
"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: ERGONOMIC POLICY ORCHESTRATOR CLI
================================================================================

1. OVERVIEW & OBJECTIVE:
   Provides a clean, intuitive, Single Responsibility Principle (SRP) compliant CLI
   with consistent argument conventions, beginner-friendly shortcuts, JSON outputs,
   and shell auto-completion generation (bash/zsh/fish).

2. SINGLE RESPONSIBILITY SUBCOMMANDS:
   - `search <patterns> [dir]`: Fast multi-pattern search (ALGO-SRCH-03 + 11 + 14).
   - `scan [dir]`: Recursive repository file tree discovery (ALGO-SRCH-01/02).
   - `outline <file>`: Hierarchical AST symbol tree generator (ALGO-OBS-17 + 21).
   - `lint <file>`: Zero-Inline-Comment doctrine compliance linter (ALGO-OBS-19).
   - `deps [dir]`: Module import dependency graph & cycle detector (ALGO-OBS-20).
   - `diff <file> <find> <replace>`: Unified context diff generator (ALGO-UPD-24).
   - `patch <file> <find> <replace> [--apply]`: AST/CST atomic patcher (ALGO-UPD-22/23).
   - `exec <algo_id> [input_json]`: Direct universal algorithm execution (All 42 algos).
   - `contracts [--category]`: Layer 1 algorithm contract discovery.
   - `compose <algo_ids...>`: Dynamic pipeline execution DAG composer.
   - `completion <bash|zsh|fish>`: Generates shell tab-completion scripts.
   - `algo <action>`: Legacy backwards-compatible unified dispatcher.
   - `migrate`: Database migrations and catalog seeding.
   - `serve`: FastAPI Uvicorn REST server.

3. ARCHITECTURAL INVARIANTS:
   - Zero-Inline-Comment Doctrine: 100% pure method/function bodies.
   - Standardized Posix Exit Codes: 0 on success, 1 on error/violation.
================================================================================
"""

import sys

ALGO_ALIASES: dict = {}
import os
import re
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import argparse
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from src.features.code_engine.service.code_engine_service import CodeEngineService
from src.features.code_engine.registry.algorithm_catalog import BUILTIN_ALGORITHM_CONTRACTS, BUILTIN_TYPE_ADAPTERS
from src.features.code_engine.service.algorithm_composer_service import AlgorithmComposerService
from src.infra.adapters.database.in_memory_algorithm_registry_adapter import InMemoryAlgorithmRegistryAdapter
from src.infra.adapters.database.migration_runner import DatabaseMigrationRunner




def handle_search_command(args: argparse.Namespace) -> int:
    svc = CodeEngineService()
    raw_patterns = getattr(args, "patterns", None) or "TODO,FIXME,error,critical"
    patterns = [p.strip() for p in raw_patterns.split(",") if p.strip()]
    target_dir = getattr(args, "directory", None) or getattr(args, "root", ".")
    results = svc.scan_directory_multipattern(target_dir, patterns)

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
        return 0

    print(f"\n{'='*75}")
    print(f"🔎 [SEARCH ENGINE] {len(results)} Files Matched in '{target_dir}'")
    print(f"{'='*75}\n")
    for res in results:
        print(f"📁 {res['file']} ({res['match_count']} matches)")
        for m in res["matches"][:3]:
            print(f"   Line {m['line']} [{m['pattern']}]: {m['context']}")
    return 0


def handle_scan_command(args: argparse.Namespace) -> int:
    target_dir = getattr(args, "root", None) or getattr(args, "directory", ".")
    svc = CodeEngineService()
    files = svc.execute_algorithm("ALGO-SRCH-01", {"root_dir": target_dir})

    if getattr(args, "json", False):
        print(json.dumps(files, indent=2))
        return 0

    file_list = files.get("files", [])
    print(f"\n{'='*75}")
    print(f"📂 [SCAN ENGINE] Discovered {len(file_list)} files under '{target_dir}'")
    print(f"{'='*75}\n")
    for f in file_list[:30]:
        print(f"  • {f}")
    if len(file_list) > 30:
        print(f"  ... and {len(file_list) - 30} more files.")
    return 0


def handle_outline_command(args: argparse.Namespace) -> int:
    file_path = getattr(args, "file", None)
    if not file_path:
        print("Error: file path is required")
        return 1

    svc = CodeEngineService()
    res = svc.inspect_file_outline(file_path)

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return 0

    print(res.get("markdown", "No outline available"))
    return 0


def handle_lint_command(args: argparse.Namespace) -> int:
    file_path = getattr(args, "file", None)
    if not file_path:
        print("Error: file path is required")
        return 1

    svc = CodeEngineService()
    res = svc.lint_zero_inline_comments(file_path)

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return 0 if res["is_compliant"] else 1

    status = "✅ COMPLIANT" if res["is_compliant"] else "❌ VIOLATION"
    print(f"\n{'='*75}")
    print(f"🛡️ [ZERO-INLINE-COMMENT DOCTRINE] {status}")
    print(f"   File: {res['file']}")
    print(f"   Banned Inline Comments: {res['banned_inline_count']}")
    print(f"   TODOs/FIXMEs: {res['todos_count']}")
    print(f"{'='*75}\n")
    for c in res["banned_comments"]:
        print(f"   Line {c['line']}: {c['text']}")
    return 0 if res["is_compliant"] else 1


def handle_deps_command(args: argparse.Namespace) -> int:
    target_dir = getattr(args, "directory", None) or getattr(args, "root", "src")
    svc = CodeEngineService()
    res = svc.analyze_module_dependencies(target_dir)

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return 0

    print(f"\n{'='*75}")
    print(f"🕸️ [IMPORT GRAPH] {res['total_modules']} Modules, {res['total_edges']} Edges")
    print(f"   Cycles: {'⚠️ DETECTED' if res['has_cycles'] else '✅ NONE'}")
    print(f"{'='*75}\n")
    if res["cycles"]:
        for cycle in res["cycles"]:
            print(f"  🔁 {' -> '.join(cycle)}")
    print(f"\nTopological Build Order ({len(res['topological_order'])} modules):")
    print(f"  {' -> '.join(res['topological_order'][:10])}{' ...' if len(res['topological_order']) > 10 else ''}")
    return 0


def handle_diff_command(args: argparse.Namespace) -> int:
    file_path = getattr(args, "file", None)
    find_str = getattr(args, "find", "")
    replace_str = getattr(args, "replace", "")
    is_regex = getattr(args, "regex", False)

    if not file_path or not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return 1

    with open(file_path, "r", encoding="utf-8") as f:
        orig = f.read()

    mod = orig.replace(find_str, replace_str) if not is_regex else re.sub(find_str, replace_str, orig)
    svc = CodeEngineService()
    diff_res = svc.generate_diff(orig, mod, file_path=file_path)

    if getattr(args, "json", False):
        print(json.dumps(diff_res, indent=2))
        return 0

    print(f"\n{'='*75}")
    print(f"📝 [UNIFIED DIFF] (+{diff_res['added_lines']} / -{diff_res['deleted_lines']})")
    print(f"{'='*75}\n")
    print(diff_res["patch"] if diff_res["has_changes"] else "No differences found.")
    return 0


def handle_patch_command(args: argparse.Namespace) -> int:
    file_path = getattr(args, "file", None)
    find_str = getattr(args, "find", "")
    replace_str = getattr(args, "replace", "")
    is_regex = getattr(args, "regex", False)
    apply_mutations = getattr(args, "apply", False)

    if not file_path:
        print("Error: file path is required")
        return 1

    svc = CodeEngineService()
    ops = [{
        "file_path": file_path,
        "find_pattern": find_str,
        "replace_text": replace_str,
        "is_regex": is_regex,
    }]
    res = svc.apply_batch_patch(ops, dry_run=not apply_mutations)

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return 0 if res[0]["success"] else 1

    status = "[APPLIED]" if apply_mutations else "[DRY-RUN]"
    print(f"\n{'='*75}")
    print(f"⚡ [ATOMIC PATCH] {status}")
    print(f"   File: {res[0]['file_path']}")
    print(f"   Matches: {res[0]['occurrences']}")
    print(f"   Pre SHA-256 : {res[0]['before_sha256'][:12]}...")
    print(f"   Post SHA-256: {res[0]['after_sha256'][:12]}...")
    print(f"{'='*75}\n")
    return 0 if res[0]["success"] else 1


def handle_exec_command(args: argparse.Namespace) -> int:
    algo_id = getattr(args, "id", None)
    if not algo_id:
        print("Error: Algorithm ID is required (e.g. ALGO-VEC-01, ALGO-SRCH-11, ALGO-GRAPH-01)")
        return 1

    input_str = getattr(args, "input", "{}") or "{}"
    try:
        if input_str.startswith("@") or (os.path.isfile(input_str) and not input_str.startswith("{")):
            path = input_str.removeprefix("@")
            with open(path, "r", encoding="utf-8") as f:
                inputs = json.load(f)
        else:
            inputs = json.loads(input_str)
    except Exception as e:
        print(f"Error parsing input JSON: {e}")
        return 1

    svc = CodeEngineService()
    try:
        result = svc.execute_algorithm(algo_id=algo_id, inputs=inputs)
        if getattr(args, "json", False):
            print(json.dumps({"algo_id": algo_id, "result": result}, indent=2))
        else:
            print(f"\n{'='*75}")
            print(f"⚡ [ALGORITHM ENGINE] EXECUTED: {algo_id}")
            print(f"{'='*75}\n")
            print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Execution failed: {e}")
        return 1


def handle_contracts_command(args: argparse.Namespace) -> int:
    category_filter = getattr(args, "category", None)
    tag_filter = getattr(args, "tag", None)
    contracts = [
        c for c in BUILTIN_ALGORITHM_CONTRACTS
        if (not category_filter or c.category.value == category_filter)
        and (not tag_filter or tag_filter in c.capability_tags)
    ]

    if getattr(args, "json", False):
        print(json.dumps([c.model_dump() for c in contracts], indent=2))
        return 0

    print(f"\n{'='*95}")
    print(f"📋 [ALGORITHM REGISTRY] {len(BUILTIN_ALGORITHM_CONTRACTS)} TOTAL CONTRACTS ({len(contracts)} matching)")
    print(f"{'='*95}")
    for c in contracts:
        print(f"  • [{c.id}] {c.name:<32} Category: {c.category.value:<14} Time: {c.complexity.time:<10} Target: {c.hardware_target.value}")
    return 0


def handle_compose_command(args: argparse.Namespace) -> int:
    raw_algos = getattr(args, "algos", "ALGO-SRCH-03,ALGO-OBS-17,ALGO-UPD-23") or "ALGO-SRCH-03,ALGO-OBS-17,ALGO-UPD-23"
    algo_ids = [a.strip() for a in raw_algos.split(",") if a.strip()]
    adapter = InMemoryAlgorithmRegistryAdapter(load_builtins=True)
    composer = AlgorithmComposerService(adapter)
    res = composer.compose_pipeline(algo_ids)

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return 0

    status = "✅ VALID" if res["is_valid"] else "❌ INVALID (Safety Warnings)"
    print(f"\n{'='*85}")
    print(f"🧩 [DYNAMIC PIPELINE COMPOSER] {status}")
    print(f"{'='*85}")
    print(f"Execution DAG Steps ({res['total_steps']}):")
    for step in res["pipeline_steps"]:
        print(f"  {step['step_index']}. ⚡ [{step['algo_id']}] {step['name']:<36} (Time: {step['complexity']['time']})")
    if res.get("inferred_adapters"):
        print("\nInjected G4 Type Adapters:")
        for ad in res["inferred_adapters"]:
            print(f"  🔧 Steps {ad['between_steps'][0]} -> {ad['between_steps'][1]}: {', '.join(ad['adapter_ids'])}")
    if res.get("safety_issues"):
        print("\nSafety Issues:")
        for w in res["safety_issues"]:
            print(f"  ⚠️  {w}")
    return 0 if res["is_valid"] else 1


def handle_run_command(args: argparse.Namespace) -> int:
    alias = getattr(args, "alias", None)
    if not alias:
        print("Error: alias name is required. Run `policy-orchestrator list` to see all available names.")
        return 1

    entry = ALGO_ALIASES.get(alias)
    if not entry:
        close = [k for k in ALGO_ALIASES if alias in k or k.startswith(alias[:3])]
        print(f"Unknown alias: '{alias}'")
        if close:
            print(f"Did you mean: {', '.join(close)}")
        print("Run `policy-orchestrator list` to see every available name.")
        return 1

    input_str = getattr(args, "input", "{}") or "{}"
    try:
        if input_str.startswith("@") or (os.path.isfile(input_str) and not input_str.startswith("{")):
            path = input_str.removeprefix("@")
            with open(path, "r", encoding="utf-8") as f:
                inputs = json.load(f)
        else:
            inputs = json.loads(input_str)
    except Exception as e:
        print(f"Bad input JSON: {e}")
        print(f"Example input for '{alias}': {entry['example']}")
        return 1

    svc = CodeEngineService()
    try:
        result = svc.execute_algorithm(algo_id=entry["id"], inputs=inputs)
    except Exception as e:
        print(f"Execution error: {e}")
        return 1

    print(f"\n{'='*75}")
    print(f"⚡  {entry['algo_name']}  ({entry['id']})")
    print(f"   {entry['plain_name']}")
    print(f"{'='*75}\n")
    print(json.dumps(result, indent=2))
    print(f"\n{'='*75}")
    print(f"📦 JSON output above. Add --json for machine-readable only output.")
    print(f"{'='*75}")
    return 0


def handle_list_command(args: argparse.Namespace) -> int:
    category_filter = getattr(args, "category", None)
    entries = {
        k: v for k, v in ALGO_ALIASES.items()
        if not category_filter or v["category"] == category_filter
    }

    if getattr(args, "json", False):
        print(json.dumps({
            k: {
                "id": v["id"],
                "algo_name": v["algo_name"],
                "plain_name": v["plain_name"],
                "what_it_does": v["what_it_does"],
                "example_input": v["example"],
                "category": v["category"],
            }
            for k, v in entries.items()
        }, indent=2))
        return 0

    categories_present = sorted({v["category"] for v in entries.values()})
    cat_labels = {"graph": "🕸️  Graph", "vector": "🔢 Vector", "search": "🔎 Search", "observability": "🔭 Observability", "update": "✏️  Update"}

    print(f"\n{'='*95}")
    print(f"  📋  ALGORITHM QUICK-REFERENCE  —  {len(entries)} commands  (filter: --category graph|vector|search|observability|update)")
    print(f"{'='*95}")
    print(f"  {'COMMAND NAME':<22} {'ALGO NAME':<32} {'PLAIN ENGLISH — WHAT IT DOES'}")
    print(f"  {'-'*22} {'-'*32} {'-'*35}")

    for cat in categories_present:
        print(f"\n  {cat_labels.get(cat, cat.upper())}")
        for alias, v in entries.items():
            if v["category"] != cat:
                continue
            print(f"  {alias:<22} {v['algo_name']:<32} {v['plain_name']}")
            print(f"  {'':22} {'':32} ↳ {v['what_it_does'][:72]}")

    print(f"\n{'='*95}")
    print(f"  HOW TO RUN:  policy-orchestrator run <COMMAND NAME> '<JSON-input>'")
    print(f"  EXAMPLE:     policy-orchestrator run bfs '{{\"graph\": {{\"A\":[\"B\"]}}, \"start_node\": \"A\"}}'")
    print(f"  EXAMPLE:     policy-orchestrator run normalize '{{\"vectors\": [[1,2,3]]}}'")
    print(f"  SEE EXAMPLE: policy-orchestrator list --json   (shows example input for each command)")
    print(f"{'='*95}\n")
    return 0


def handle_completion_command(args: argparse.Namespace) -> int:
    shell = getattr(args, "shell", "bash")
    algo_ids = [c.id for c in BUILTIN_ALGORITHM_CONTRACTS]
    alias_names = list(ALGO_ALIASES.keys())
    subcmds = [
        "search", "scan", "outline", "lint", "deps", "diff", "patch", "exec", "run", "list",
        "contracts", "compose", "migrate", "serve", "completion", "algo", "create-file", "link-file"
    ]
    categories = ["search", "observability", "update", "vector", "graph", "filter"]
    rel_types = [
        "CALLS", "DECLARES_STATE", "CONTROLS_STATE", "DEFINES_MODEL", "RENDERS_STATE",
        "DISPATCHES_ACTION", "IMPLEMENTS_PORT", "CONSUMES_PORT", "EXECUTES_QUERY",
        "EVALUATES_RULES", "TRANSITIONS_MACHINE", "RUNS_WORKFLOW", "DEFINES_SCHEMA", "CONTAINS_FEATURE"
    ]
    roles = [
        "Router", "Handler", "Schema", "Types", "Context", "Service", "Rules", "Machine",
        "Workflow", "RepositoryPort", "RepositoryAdapter", "Queries", "Index", "Test"
    ]

    if shell == "bash":
        script = f"""# Bash completion for policy-orchestrator
_policy_orchestrator_completion() {{
    local cur prev subcmd
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    subcmd="${{COMP_WORDS[1]}}"

    local subcommands="{' '.join(subcmds)}"
    local algo_ids="{' '.join(algo_ids)}"
    local alias_names="{' '.join(alias_names)}"
    local categories="{' '.join(categories)}"
    local rel_types="{' '.join(rel_types)}"
    local roles="{' '.join(roles)}"

    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "$subcommands" -- "$cur") )
        return 0
    fi

    case "$prev" in
        --rel|--rel-type)
            COMPREPLY=( $(compgen -W "$rel_types" -- "$cur") )
            return 0
            ;;
        --role)
            COMPREPLY=( $(compgen -W "$roles" -- "$cur") )
            return 0
            ;;
        --direction)
            COMPREPLY=( $(compgen -W "outgoing incoming" -- "$cur") )
            return 0
            ;;
        --store)
            COMPREPLY=( $(compgen -W "memory neo4j" -- "$cur") )
            return 0
            ;;
        --category)
            COMPREPLY=( $(compgen -W "$categories" -- "$cur") )
            return 0
            ;;
        --path|--target|--source|--file)
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
        --base-dir|--root|--dir)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
    esac

    case "$subcmd" in
        exec|execute)
            if [ $COMP_CWORD -eq 2 ] || [ "$prev" = "--id" ]; then
                COMPREPLY=( $(compgen -W "$algo_ids" -- "$cur") )
                return 0
            fi
            ;;
        run)
            if [ $COMP_CWORD -eq 2 ]; then
                COMPREPLY=( $(compgen -W "$alias_names" -- "$cur") )
                return 0
            fi
            ;;
        outline|lint|diff|patch|create-file|link-file|file-lineage)
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
        search|scan|deps|graph-scan|scaffold-feature|scaffold-package)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
    esac

    local opts="--json --help --root --directory --file --find --replace --regex --apply --patterns --path --role --rel --target --direction --base-dir --name --store --neo4j"
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}}
complete -F _policy_orchestrator_completion policy-orchestrator
complete -F _policy_orchestrator_completion python3 -m src.api.cli.main
"""
    elif shell == "zsh":
        script = f"""#compdef policy-orchestrator
_policy_orchestrator() {{
    local -a subcommands
    subcommands=({' '.join([f"'{s}:{s} command'" for s in subcmds])})
    _arguments '1: :->subcmd' '*: :->args'
    case $state in
        subcmd)
            _describe 'command' subcommands
            ;;
        args)
            case $words[2] in
                exec|execute)
                    _values 'algorithm_id' {' '.join(algo_ids)}
                    ;;
                contracts)
                    _values 'category' {' '.join(categories)}
                    ;;
                *)
                    _files
                    ;;
            esac
            ;;
    esac
}}
compdef _policy_orchestrator policy-orchestrator
"""
    else:
        script = f"""# Fish completion for policy-orchestrator
complete -c policy-orchestrator -f -n '__fish_use_subcommand' -a "{' '.join(subcmds)}"
"""
    print(script)
    return 0


def handle_algo_command(args: argparse.Namespace) -> int:
    action = getattr(args, "action", None)
    if action in {"search", "scan"} and hasattr(args, "patterns"):
        return handle_search_command(args)
    elif action == "scan":
        return handle_scan_command(args)
    elif action == "outline":
        return handle_outline_command(args)
    elif action == "lint-comments":
        return handle_lint_command(args)
    elif action == "dependencies":
        return handle_deps_command(args)
    elif action == "patch":
        return handle_patch_command(args)
    elif action == "diff":
        return handle_diff_command(args)
    elif action == "contracts":
        return handle_contracts_command(args)
    elif action == "compose":
        return handle_compose_command(args)
    elif action == "execute":
        return handle_exec_command(args)
    return 0





def handle_migrate_command(args: argparse.Namespace) -> int:
    from src.infra.config.config_loader import AppConfigLoader
    db_target = getattr(args, "db", "all")
    if db_target == "sqlite":
        db_url = f"sqlite:///{getattr(args, 'sqlite_path', 'policy_registry.db')}"
    else:
        db_url = getattr(args, "postgres_url", None) or AppConfigLoader.get_database_url()

    runner = DatabaseMigrationRunner(db_url)
    action = getattr(args, "action", "run")

    if action == "run":
        if db_target in ("all", "alloydb", "postgres"):
            runner.run_migrations()
            count = runner.seed_algorithm_catalog()
            print(f"✅ AlloyDB SQL migration and catalog seed successful: {count} contracts.")
        if db_target in ("all", "neo4j"):
            graph_res = runner.run_graph_migrations()
            print(f"✅ Neo4j Graph constraint migration successful: {graph_res}")
        return 0
    elif action == "rollback":
        if db_target in ("all", "neo4j"):
            graph_res = runner.rollback_graph_migrations()
            print(f"🔄 Neo4j Graph rollback successful: {graph_res}")
        if db_target in ("all", "alloydb", "postgres", "sqlite"):
            res = runner.rollback_migrations()
            print(f"🔄 SQL rollback successful: {res}")
        return 0
    elif action == "seed":
        count = runner.seed_algorithm_catalog()
        print(f"🌱 Seeded {count} algorithm contracts & type adapters into {db_target}.")
        return 0
    elif action == "parity":
        parity = runner.verify_database_parity()
        print(json.dumps(parity, indent=2))
        return 0 if parity.get("parity_matched") else 1
    elif action == "status":
        status = runner.get_migration_status()
        print(json.dumps(status, indent=2))
        return 0
    return 0



def resolve_graph_adapter(args: argparse.Namespace):
    from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
    from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
    store = getattr(args, "store", None)
    neo4j_flag = getattr(args, "neo4j", False)
    if neo4j_flag or store == "neo4j" or os.environ.get("GRAPH_STORE") == "neo4j":
        return Neo4jGraphAdapter()
    return InMemoryGraphAdapter()






def handle_serve_command(args: argparse.Namespace) -> int:
    import uvicorn
    os.environ["POLICY_RULES_DIR"] = args.rules_dir
    os.environ["LLM_BACKEND"] = args.backend
    uvicorn.run("src.api.rest.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-orchestrator",
        description="Repository Invariant Auditor, Layer 1 Algorithm Suite & Policy Orchestrator",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. Search Subcommand
    search_p = subparsers.add_parser("search", help="Fast multi-pattern code search (ALGO-SRCH-03 + 11 + 14)")
    search_p.add_argument("patterns", nargs="?", default="TODO,FIXME,error,critical", help="Comma-separated patterns to find")
    search_p.add_argument("directory", nargs="?", default=".", help="Target directory (default: current)")
    search_p.add_argument("--patterns", dest="patterns_flag", default=None, help="Explicit --patterns flag")
    search_p.add_argument("--directory", "--root", dest="dir_flag", default=None, help="Explicit directory flag")
    search_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 2. Scan Subcommand
    scan_p = subparsers.add_parser("scan", help="Recursive repository file tree discovery (ALGO-SRCH-01/02)")
    scan_p.add_argument("directory", nargs="?", default=".", help="Root directory (default: .)")
    scan_p.add_argument("--root", dest="root_flag", default=None, help="Explicit --root flag")
    scan_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 3. Outline Subcommand
    outline_p = subparsers.add_parser("outline", help="Hierarchical AST symbol outline (ALGO-OBS-17 + 21)")
    outline_p.add_argument("file", nargs="?", default=None, help="Source code file path")
    outline_p.add_argument("--file", dest="file_flag", default=None, help="Explicit --file flag")
    outline_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 4. Lint Subcommand
    lint_p = subparsers.add_parser("lint", help="Zero-Inline-Comment doctrine compliance linter (ALGO-OBS-19)")
    lint_p.add_argument("file", nargs="?", default=None, help="Source code file path")
    lint_p.add_argument("--file", dest="file_flag", default=None, help="Explicit --file flag")
    lint_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 5. Dependencies / Deps Subcommand
    deps_p = subparsers.add_parser("deps", aliases=["dependencies"], help="Module import dependency graph & cycle detector (ALGO-OBS-20)")
    deps_p.add_argument("directory", nargs="?", default="src", help="Target source directory")
    deps_p.add_argument("--directory", dest="dir_flag", default=None, help="Explicit directory flag")
    deps_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 6. Diff Subcommand
    diff_p = subparsers.add_parser("diff", help="Unified context diff generator (ALGO-UPD-24)")
    diff_p.add_argument("file", nargs="?", default=None, help="File to inspect")
    diff_p.add_argument("find", nargs="?", default="", help="Pattern to match")
    diff_p.add_argument("replace", nargs="?", default="", help="Replacement string")
    diff_p.add_argument("--file", dest="file_flag", default=None, help="Explicit --file flag")
    diff_p.add_argument("--find", dest="find_flag", default=None, help="Explicit --find flag")
    diff_p.add_argument("--replace", dest="replace_flag", default=None, help="Explicit --replace flag")
    diff_p.add_argument("--regex", action="store_true", help="Treat pattern as regex")
    diff_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 7. Patch Subcommand
    patch_p = subparsers.add_parser("patch", help="AST/CST atomic batch patcher (ALGO-UPD-22/23)")
    patch_p.add_argument("file", nargs="?", default=None, help="File to patch")
    patch_p.add_argument("find", nargs="?", default="", help="Pattern to match")
    patch_p.add_argument("replace", nargs="?", default="", help="Replacement string")
    patch_p.add_argument("--file", dest="file_flag", default=None, help="Explicit --file flag")
    patch_p.add_argument("--find", dest="find_flag", default=None, help="Explicit --find flag")
    patch_p.add_argument("--replace", dest="replace_flag", default=None, help="Explicit --replace flag")
    patch_p.add_argument("--apply", action="store_true", help="Apply mutations to disk (default is dry-run)")
    patch_p.add_argument("--regex", action="store_true", help="Treat pattern as regex")
    patch_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 8. Exec Subcommand
    exec_p = subparsers.add_parser("exec", aliases=["execute"], help="Direct universal algorithm execution across all 42 algos")
    exec_p.add_argument("id", nargs="?", default=None, help="Algorithm ID (e.g. ALGO-VEC-01, ALGO-GRAPH-01, ALGO-SRCH-11)")
    exec_p.add_argument("input", nargs="?", default="{}", help="Input payload JSON string or @filepath")
    exec_p.add_argument("--id", dest="id_flag", default=None, help="Explicit --id flag")
    exec_p.add_argument("--input", dest="input_flag", default=None, help="Explicit --input flag")
    exec_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 9. Contracts Subcommand
    contracts_p = subparsers.add_parser("contracts", help="List and filter Layer 1 Algorithm Contracts")
    contracts_p.add_argument("--category", choices=["search", "observability", "update", "vector", "graph"], default=None, help="Filter by category")
    contracts_p.add_argument("--tag", default=None, help="Filter by capability tag")
    contracts_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 10. Compose Subcommand
    compose_p = subparsers.add_parser("compose", help="Compose dynamic DAG pipeline from algorithm IDs")
    compose_p.add_argument("algos", nargs="?", default="ALGO-SRCH-03,ALGO-OBS-17,ALGO-UPD-23", help="Comma-separated algorithm IDs")
    compose_p.add_argument("--algos", dest="algos_flag", default=None, help="Explicit --algos flag")
    compose_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 11. Run Subcommand (human-friendly alias runner)
    run_p = subparsers.add_parser("run", help="Run any algorithm by its human-friendly name (e.g. bfs, dijkstra, normalize, chunk)")
    run_p.add_argument("alias", nargs="?", default=None, help="Human-friendly algorithm name (run `policy-orchestrator list` to see all)")
    run_p.add_argument("input", nargs="?", default="{}", help="Input as JSON string or @filepath")
    run_p.add_argument("--alias", dest="alias_flag", default=None, help="Explicit --alias flag")
    run_p.add_argument("--input", dest="input_flag", default=None, help="Explicit --input flag")
    run_p.add_argument("--json", action="store_true", help="Output as JSON")

    # 12. List Subcommand (human-friendly discovery)
    list_p = subparsers.add_parser("list", help="List every algorithm with plain-English names and descriptions")
    list_p.add_argument("--category", choices=["search", "observability", "update", "vector", "graph"], default=None, help="Filter by category")
    list_p.add_argument("--json", action="store_true", help="Output full detail as JSON")

    # 13. Completion Subcommand
    comp_p = subparsers.add_parser("completion", help="Generate shell auto-completion script (bash, zsh, fish)")
    comp_p.add_argument("shell", choices=["bash", "zsh", "fish"], default="bash", nargs="?", help="Target shell")

    # 12. Legacy Algo Subcommand (Backwards Compatibility)
    algo_parser = subparsers.add_parser("algo", help="Execute Categorized Search, Observability, Update & Vector Algorithms")
    algo_parser.add_argument("action", choices=["search", "scan", "outline", "lint-comments", "dependencies", "patch", "diff", "contracts", "compose", "execute"], help="Algorithm action")
    algo_parser.add_argument("--id", default=None, help="Algorithm ID to execute")
    algo_parser.add_argument("--input", default="{}", help="Input payload JSON string or @filepath")
    algo_parser.add_argument("--root", default=".", help="Root directory")
    algo_parser.add_argument("--patterns", default="TODO,FIXME,error,critical", help="Comma-separated patterns")
    algo_parser.add_argument("--file", default="src/api/cli/main.py", help="File to inspect or patch")
    algo_parser.add_argument("--directory", default="src", help="Directory to analyze")
    algo_parser.add_argument("--find", default="", help="Pattern to find for update/diff/patch")
    algo_parser.add_argument("--replace", default="", help="Replacement string for update/diff/patch")
    algo_parser.add_argument("--regex", action="store_true", help="Treat find pattern as regex")
    algo_parser.add_argument("--apply", action="store_true", help="Apply patch modifications to disk")
    algo_parser.add_argument("--category", default=None, help="Filter contracts by category")
    algo_parser.add_argument("--algos", default="ALGO-SRCH-03,ALGO-OBS-17,ALGO-UPD-23", help="Comma-separated algorithm IDs to compose")
    algo_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Standard existing commands
    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations and seed algorithm contracts")
    migrate_parser.add_argument("action", choices=["run", "status", "rollback", "seed", "parity"], default="run", nargs="?", help="Migration action")
    migrate_parser.add_argument("--db", choices=["all", "alloydb", "postgres", "neo4j", "sqlite"], default="all", help="Target database (all runs both AlloyDB and Neo4j)")
    migrate_parser.add_argument("--sqlite-path", default="policy_registry.db", help="SQLite database path")
    migrate_parser.add_argument("--postgres-url", default=None, help="PostgreSQL connection string")

    # Scaffolding & Knowledge Graph Subcommands (Modularized in scaffold_cli.py)
    from src.api.cli.scaffold_cli import (
        register_scaffold_subparsers,
        handle_create_file,
        handle_link_file,
    )
    register_scaffold_subparsers(subparsers)

    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument("--rules-dir", default="../rules", help="Path to rules folder")
    serve_parser.add_argument("--backend", default="mock", choices=["mock", "openai", "ollama"], help="LLM backend")

    return parser


def main() -> None:
    from src.api.cli.scaffold_cli import (
        handle_create_file,
        handle_link_file,
    )
    parser = build_parser()
    args = parser.parse_args()

    # Normalise flags vs positional shortcuts
    if hasattr(args, "patterns_flag") and args.patterns_flag:
        args.patterns = args.patterns_flag
    if hasattr(args, "dir_flag") and args.dir_flag:
        args.directory = args.dir_flag
        args.root_dir = args.dir_flag
    if hasattr(args, "root_flag") and args.root_flag:
        args.root = args.root_flag
    if hasattr(args, "file_flag") and args.file_flag:
        args.file = args.file_flag
    if hasattr(args, "find_flag") and args.find_flag:
        args.find = args.find_flag
    if hasattr(args, "replace_flag") and args.replace_flag:
        args.replace = args.replace_flag
    if hasattr(args, "id_flag") and args.id_flag:
        args.id = args.id_flag
    if hasattr(args, "input_flag") and args.input_flag:
        args.input = args.input_flag
    if hasattr(args, "algos_flag") and args.algos_flag:
        args.algos = args.algos_flag
    if hasattr(args, "name_flag") and args.name_flag:
        args.feature_name = args.name_flag
        args.package_name = args.name_flag
    if hasattr(args, "target_flag") and args.target_flag:
        args.target_id = args.target_flag

    if hasattr(args, "alias_flag") and args.alias_flag:
        args.alias = args.alias_flag

    dispatch = {
        "search": handle_search_command,
        "scan": handle_scan_command,
        "outline": handle_outline_command,
        "lint": handle_lint_command,
        "deps": handle_deps_command,
        "dependencies": handle_deps_command,
        "diff": handle_diff_command,
        "patch": handle_patch_command,
        "exec": handle_exec_command,
        "execute": handle_exec_command,
        "run": handle_run_command,
        "list": handle_list_command,
        "contracts": handle_contracts_command,
        "compose": handle_compose_command,
        "completion": handle_completion_command,
        "algo": handle_algo_command,
        "migrate": handle_migrate_command,
        "create-file": handle_create_file,
        "add-file": handle_create_file,
        "link-file": handle_link_file,
        "link-nodes": handle_link_file,
        "connect-files": handle_link_file,
        "serve": handle_serve_command,
    }

    handler = dispatch.get(args.subcommand)
    if handler:
        sys.exit(handler(args))


if __name__ == "__main__":
    main()

