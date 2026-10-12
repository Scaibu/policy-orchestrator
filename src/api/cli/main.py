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
   - `audit`: Multi-vector invariant and architecture scanner.
   - `rag`: Grounded semantic policy retrieval.
   - `agent`: Autonomous AI agent execution.
   - `migrate`: Database migrations and catalog seeding.
   - `serve`: FastAPI Uvicorn REST server.

3. ARCHITECTURAL INVARIANTS:
   - Zero-Inline-Comment Doctrine: 100% pure method/function bodies.
   - Standardized Posix Exit Codes: 0 on success, 1 on error/violation.
================================================================================
"""

import sys

ALGO_ALIASES: dict = {
    "bfs": {
        "id": "ALGO-GRAPH-01",
        "algo_name": "BFS Traversal",
        "plain_name": "Level-by-level graph walk",
        "what_it_does": "Visits every node starting from a point, layer by layer — like ripples spreading in water.",
        "example": '{"graph": {"A":["B","C"],"B":["D"],"C":[],"D":[]}, "start_node": "A"}',
        "category": "graph",
    },
    "dfs": {
        "id": "ALGO-GRAPH-02",
        "algo_name": "DFS Traversal",
        "plain_name": "Deep-dive graph walk",
        "what_it_does": "Goes as deep as possible down one path before backtracking — like exploring a maze by hugging one wall.",
        "example": '{"graph": {"A":["B","C"],"B":["D"],"C":[],"D":[]}, "start_node": "A"}',
        "category": "graph",
    },
    "dijkstra": {
        "id": "ALGO-GRAPH-03",
        "algo_name": "Dijkstra Shortest Path",
        "plain_name": "Find cheapest route between two points",
        "what_it_does": "Finds the least-cost path between two nodes in a network — like a GPS choosing the fastest road.",
        "example": '{"graph": {"A":{"B":1,"C":4},"B":{"C":2,"D":5},"C":{"D":1},"D":{}}, "source": "A", "target": "D"}',
        "category": "graph",
    },
    "shortest-path": {
        "id": "ALGO-GRAPH-03",
        "algo_name": "Dijkstra Shortest Path",
        "plain_name": "Find cheapest route between two points",
        "what_it_does": "Finds the least-cost path between two nodes in a network — like a GPS choosing the fastest road.",
        "example": '{"graph": {"A":{"B":1,"C":4},"B":{"C":2,"D":5},"C":{"D":1},"D":{}}, "source": "A", "target": "D"}',
        "category": "graph",
    },
    "astar": {
        "id": "ALGO-GRAPH-04",
        "algo_name": "A* Search",
        "plain_name": "Smart guided route finder",
        "what_it_does": "Finds the fastest path using a hint about the direction of the goal — smarter than plain shortest-path.",
        "example": '{"graph": {"A":{"B":1},"B":{"C":1},"C":{}}, "start": "A", "goal": "C", "heuristic": {"A":2,"B":1,"C":0}}',
        "category": "graph",
    },
    "pagerank": {
        "id": "ALGO-GRAPH-05",
        "algo_name": "PageRank Centrality",
        "plain_name": "Rank nodes by importance",
        "what_it_does": "Scores nodes by how many important nodes point to them — the same idea Google uses to rank web pages.",
        "example": '{"graph": {"A":["B","C"],"B":["C"],"C":["A"]}, "damping": 0.85, "iterations": 100}',
        "category": "graph",
    },
    "centrality": {
        "id": "ALGO-GRAPH-06",
        "algo_name": "Degree Centrality",
        "plain_name": "Who is the most connected node?",
        "what_it_does": "Ranks each node by how many direct connections it has — the more connections, the more central.",
        "example": '{"graph": {"A":["B","C"],"B":["A","C"],"C":["A","B"]}}',
        "category": "graph",
    },
    "components": {
        "id": "ALGO-GRAPH-07",
        "algo_name": "Connected Components",
        "plain_name": "Find isolated groups in a graph",
        "what_it_does": "Finds all groups of nodes that are connected to each other but completely separate from other groups.",
        "example": '{"graph": {"A":["B"],"B":["A"],"C":["D"],"D":["C"],"E":[]}}',
        "category": "graph",
    },
    "tarjan": {
        "id": "ALGO-GRAPH-08",
        "algo_name": "Tarjan SCC",
        "plain_name": "Find circular dependency groups",
        "what_it_does": "Finds groups where every node can reach every other — catches circular imports, deadlock candidates.",
        "example": '{"graph": {"A":["B"],"B":["C"],"C":["A"],"D":["B"]}}',
        "category": "graph",
    },
    "pattern-match": {
        "id": "ALGO-GRAPH-09",
        "algo_name": "Subgraph Isomorphism",
        "plain_name": "Does this pattern exist in the graph?",
        "what_it_does": "Checks whether a small graph pattern exists inside a larger graph — like searching for a shape inside a diagram.",
        "example": '{"graph": {"A":["B","C"],"B":["C"],"C":[]}, "pattern": {"X":["Y"],"Y":[]}}',
        "category": "graph",
    },
    "normalize": {
        "id": "ALGO-VEC-01",
        "algo_name": "L2 Normalization",
        "plain_name": "Scale all vectors to the same size",
        "what_it_does": "Adjusts number lists so each has the same total size — like converting different speeds into directions.",
        "example": '{"vectors": [[1,2,3],[4,5,6]]}',
        "category": "vector",
    },
    "mean-center": {
        "id": "ALGO-VEC-02",
        "algo_name": "Mean Centering",
        "plain_name": "Remove average bias from data",
        "what_it_does": "Shifts all values so the average becomes zero — removes any dataset-wide offset before comparison.",
        "example": '{"vectors": [[1,2,3],[4,5,6]]}',
        "category": "vector",
    },
    "scale": {
        "id": "ALGO-VEC-03",
        "algo_name": "MinMax / ZScore Normalization",
        "plain_name": "Rescale numbers into a standard range",
        "what_it_does": "Squishes all values into 0–1 (minmax) or standard-deviation units (zscore) so they're comparable.",
        "example": '{"vectors": [[1,2,3]], "method": "minmax"}',
        "category": "vector",
    },
    "token-pool": {
        "id": "ALGO-VEC-04",
        "algo_name": "Token Pooling",
        "plain_name": "Collapse many token vectors into one summary",
        "what_it_does": "Combines many small vectors (one per word/token) into a single summary vector using mean, max, or first.",
        "example": '{"token_embeddings": [[0.1,0.2],[0.3,0.4]], "strategy": "mean"}',
        "category": "vector",
    },
    "chunk": {
        "id": "ALGO-VEC-05",
        "algo_name": "Semantic Chunker",
        "plain_name": "Split long text by topic",
        "what_it_does": "Breaks a long document into chunks where each chunk stays on the same topic — not just by word count.",
        "example": '{"sentences": ["Hello world.", "This is a test."], "threshold": 0.8}',
        "category": "vector",
    },
    "quantize": {
        "id": "ALGO-VEC-06",
        "algo_name": "Scalar Quantization",
        "plain_name": "Compress vectors to save memory",
        "what_it_does": "Reduces floating point precision in vectors — dramatically cuts memory use with minimal accuracy loss.",
        "example": '{"vectors": [[1.5,2.5,3.5]], "bits": 8}',
        "category": "vector",
    },
    "binary-quantize": {
        "id": "ALGO-VEC-07",
        "algo_name": "Binary Quantization",
        "plain_name": "Shrink vectors to just 0s and 1s",
        "what_it_does": "Converts each vector value to a single bit — extreme compression for very fast similarity search.",
        "example": '{"vectors": [[0.5,-0.3,1.2,-0.8]]}',
        "category": "vector",
    },
    "slice": {
        "id": "ALGO-VEC-08",
        "algo_name": "Matryoshka Slicing",
        "plain_name": "Trim big vectors to smaller useful sizes",
        "what_it_does": "Cuts a large embedding to smaller sizes (like 128 or 256 dims) that still work well for search.",
        "example": '{"vector": [0.1,0.2,0.3,0.4,0.5,0.6], "dimensions": [2,4]}',
        "category": "vector",
    },
    "layer-norm": {
        "id": "ALGO-VEC-09",
        "algo_name": "Layer Normalization",
        "plain_name": "Stabilize vector values for AI models",
        "what_it_does": "Normalizes values within each individual vector — keeps AI model inputs in a stable, consistent range.",
        "example": '{"vectors": [[1.0,2.0,3.0]]}',
        "category": "vector",
    },
    "file-walk": {
        "id": "ALGO-SRCH-01",
        "algo_name": "Recursive File Walker",
        "plain_name": "List all files in a folder",
        "what_it_does": "Walks through every folder and subfolder and returns a list of every file it finds.",
        "example": '{"root_dir": "."}',
        "category": "search",
    },
    "git-walk": {
        "id": "ALGO-SRCH-02",
        "algo_name": "Work-Stealing Parallel Walker",
        "plain_name": "List files using multiple CPU cores",
        "what_it_does": "Walks directory trees across multiple CPU threads simultaneously — much faster in large repos.",
        "example": '{"root_dir": ".", "num_workers": 4}',
        "category": "search",
    },
    "git-ignore-walk": {
        "id": "ALGO-SRCH-03",
        "algo_name": "Git-Aware Walker",
        "plain_name": "List files respecting .gitignore",
        "what_it_does": "Lists files but automatically skips everything your .gitignore file says to ignore.",
        "example": '{"root_dir": ".", "respect_gitignore": true}',
        "category": "search",
    },
    "glob": {
        "id": "ALGO-SRCH-04",
        "algo_name": "Glob Matcher",
        "plain_name": "Match files by wildcard pattern",
        "what_it_does": "Filters file paths using wildcards like *.py, src/**/*.ts — like a smart filename filter.",
        "example": '{"pattern": "**/*.py", "paths": ["src/main.py","src/test.go"]}',
        "category": "search",
    },
    "trigram": {
        "id": "ALGO-SRCH-05",
        "algo_name": "Trigram Index Search",
        "plain_name": "Fast fuzzy text search using 3-letter chunks",
        "what_it_does": "Builds a 3-letter (trigram) index for extremely fast substring and fuzzy text searches.",
        "example": '{"documents": ["hello world","foo bar"], "query": "hello"}',
        "category": "search",
    },
    "mmap-scan": {
        "id": "ALGO-SRCH-06",
        "algo_name": "Memory-Mapped File Scanner",
        "plain_name": "Scan huge files without loading them fully",
        "what_it_does": "Reads files directly from disk memory — searches gigabyte-sized files instantly without RAM issues.",
        "example": '{"file_path": "src/api/cli/main.py", "pattern": "def handle"}',
        "category": "search",
    },
    "stream-scan": {
        "id": "ALGO-SRCH-07",
        "algo_name": "Streaming Chunk Scanner",
        "plain_name": "Scan large files in small pieces",
        "what_it_does": "Reads files piece-by-piece so even enormous files can be searched without using too much memory.",
        "example": '{"file_path": "src/api/cli/main.py", "pattern": "policy"}',
        "category": "search",
    },
    "size-filter": {
        "id": "ALGO-SRCH-08",
        "algo_name": "Size/Line Bouncer",
        "plain_name": "Skip files that are too big or too long",
        "what_it_does": "Checks if a file exceeds a size or line-count limit before wasting time scanning it.",
        "example": '{"file_path": "src/api/cli/main.py", "max_size_bytes": 1000000, "max_lines": 50000}',
        "category": "search",
    },
    "regex-scan": {
        "id": "ALGO-SRCH-09",
        "algo_name": "Lazy DFA Regex Engine",
        "plain_name": "Search using a regex pattern (no backtracking)",
        "what_it_does": "Runs regular expressions efficiently using a deterministic finite automaton — never hangs on bad patterns.",
        "example": '{"pattern": "def \\\\w+", "text": "def foo(): pass"}',
        "category": "search",
    },
    "snippet": {
        "id": "ALGO-SRCH-10",
        "algo_name": "Context Snippet Collector",
        "plain_name": "Show lines around a match",
        "what_it_does": "Returns the lines before and after a match so you can read results in context, not just the hit line.",
        "example": '{"file_path": "src/api/cli/main.py", "pattern": "def main", "context_lines": 3}',
        "category": "search",
    },
    "byte-search": {
        "id": "ALGO-SRCH-11",
        "algo_name": "SIMD Memchr Byte Search",
        "plain_name": "Ultra-fast single-byte or character search",
        "what_it_does": "Finds a specific byte or character using CPU hardware acceleration — fastest possible byte scan.",
        "example": '{"text": "hello world", "byte_value": 32}',
        "category": "search",
    },
    "is-binary": {
        "id": "ALGO-SRCH-12",
        "algo_name": "Binary File Classifier",
        "plain_name": "Check if a file is binary or text",
        "what_it_does": "Quickly determines whether a file is binary (image, executable) or plain text — so you know if it's readable.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "search",
    },
    "content-type": {
        "id": "ALGO-SRCH-13",
        "algo_name": "Content-Type Prober",
        "plain_name": "Detect what type of file this is",
        "what_it_does": "Sniffs a file's content to identify its type (Python, JSON, Markdown, etc.) — like a file-type detector.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "search",
    },
    "is-generated": {
        "id": "ALGO-SRCH-14",
        "algo_name": "Generated Code Classifier",
        "plain_name": "Was this file auto-generated?",
        "what_it_does": "Detects files created by code generators (protobuf, swagger, etc.) that you should never edit by hand.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "search",
    },
    "parallel-walk": {
        "id": "ALGO-SRCH-15",
        "algo_name": "Memory-Mapped File Scanner",
        "plain_name": "Zero-copy memory mapped scanner",
        "what_it_does": "Scans huge files directly through OS page cache without heap allocation.",
        "example": '{"file_path": "src/api/cli/main.py", "pattern": "TODO"}',
        "category": "search",
    },
    "wu-manber": {
        "id": "ALGO-SRCH-25",
        "algo_name": "Wu-Manber Multi-Pattern Search",
        "plain_name": "Sub-linear multi-pattern scanner",
        "what_it_does": "Searches for thousands of keywords simultaneously using B-gram block hashing.",
        "example": '{"text": "the quick brown fox", "patterns": ["quick", "fox"], "block_size": 2}',
        "category": "search",
    },
    "z-algorithm": {
        "id": "ALGO-SRCH-26",
        "algo_name": "Z-Algorithm Prefix Preprocessor",
        "plain_name": "Linear prefix periodicity finder",
        "what_it_does": "Calculates longest prefix matches in strict O(N) linear time.",
        "example": '{"text": "abracadabra", "pattern": "abra"}',
        "category": "search",
    },
    "levenshtein-distance": {
        "id": "ALGO-SRCH-27",
        "algo_name": "Levenshtein Dynamic Programming Distance",
        "plain_name": "Exact edit distance with traceback",
        "what_it_does": "Computes insert/delete/substitute costs and edit paths between strings.",
        "example": '{"source": "kitten", "target": "sitting"}',
        "category": "search",
    },
    "myers-bit-parallel": {
        "id": "ALGO-SRCH-28",
        "algo_name": "Myers Bit-Parallel Edit Distance",
        "plain_name": "Bit-parallel approximate search",
        "what_it_does": "Evaluates whole DP columns via 64-bit integer vectors in O(N).",
        "example": '{"text": "hello world", "pattern": "word", "max_distance": 1}',
        "category": "search",
    },
    "levenshtein-automaton": {
        "id": "ALGO-SRCH-29",
        "algo_name": "Levenshtein Automaton",
        "plain_name": "Parametric NFA/DFA for fuzzy matching",
        "what_it_does": "Accepts words within edit distance K in O(|Word|) per test candidate.",
        "example": '{"pattern": "user", "candidates": ["user", "uses", "admin"], "max_distance": 1}',
        "category": "search",
    },
    "bk-tree": {
        "id": "ALGO-SRCH-30",
        "algo_name": "Burkhard-Keller Metric Tree",
        "plain_name": "Metric tree for fuzzy dictionary queries",
        "what_it_does": "Prunes search trees using triangle inequality for fast spell/symbol lookup.",
        "example": '{"dictionary": ["book", "books", "cake"], "query": "book", "max_distance": 1}',
        "category": "search",
    },
    "minhash-jaccard": {
        "id": "ALGO-SRCH-31",
        "algo_name": "MinHash & Jaccard Similarity",
        "plain_name": "LSH near-duplicate and clone detector",
        "what_it_does": "Hashes shingles to find duplicate code blocks and near-duplicate files.",
        "example": '{"documents": [{"id": "d1", "text": "foo bar"}, {"id": "d2", "text": "foo bar baz"}]}',
        "category": "search",
    },
    "fzf-fuzzy": {
        "id": "ALGO-SRCH-32",
        "algo_name": "fzf Fuzzy Subsequence Scorer",
        "plain_name": "Fuzzy path and symbol matcher",
        "what_it_does": "Scores alignments rewarding word boundaries, camelCase humps, and consecutive runs.",
        "example": '{"candidates": ["src/service.py", "tests/test.py"], "query": "srv"}',
        "category": "search",
    },
    "regex-parser": {
        "id": "ALGO-SRCH-33",
        "algo_name": "Recursive Descent Regex Parser",
        "plain_name": "Regex pattern to AST compiler",
        "what_it_does": "Parses regex syntax into explicit AST nodes.",
        "example": '{"pattern": "abc(def|ghi)+[0-9]*"}',
        "category": "search",
    },
    "thompson-nfa": {
        "id": "ALGO-SRCH-34",
        "algo_name": "Thompson NFA Construction",
        "plain_name": "O(M) state NFA assembler",
        "what_it_does": "Builds ReDoS-safe linear NFA state graphs from regex patterns.",
        "example": '{"pattern": "a(b|c)*d"}',
        "category": "search",
    },
    "pike-vm": {
        "id": "ALGO-SRCH-35",
        "algo_name": "Pike VM Linear NFA Evaluator",
        "plain_name": "Linear-time thread runner with captures",
        "what_it_does": "Executes NFAs in strict O(N*M) time without catastrophic backtracking.",
        "example": '{"text": "abc123xyz", "pattern": "c123x"}',
        "category": "search",
    },
    "backtracking-regex": {
        "id": "ALGO-SRCH-36",
        "algo_name": "Safe Backtracking Regex Engine",
        "plain_name": "Lookaround & backreference engine",
        "what_it_does": "Evaluates complex assertions under strict step execution caps.",
        "example": '{"text": "foo123 bar456", "pattern": "(?<=foo)\\\\d+"}',
        "category": "search",
    },
    "subset-dfa": {
        "id": "ALGO-SRCH-37",
        "algo_name": "Subset Construction DFA Engine",
        "plain_name": "Powerset state transition table generator",
        "what_it_does": "Converts NFA to minimal DFA table for O(1) byte matching.",
        "example": '{"text": "abcab", "pattern": "ab"}',
        "category": "search",
    },
    "lazy-hybrid-dfa": {
        "id": "ALGO-SRCH-38",
        "algo_name": "Lazy Hybrid Dynamic DFA",
        "plain_name": "On-the-fly state caching DFA",
        "what_it_does": "Caches transitions dynamically on demand with memory caps.",
        "example": '{"text": "hello world", "pattern": "world"}',
        "category": "search",
    },
    "literal-extraction": {
        "id": "ALGO-SRCH-39",
        "algo_name": "Required Literal Extraction Engine",
        "plain_name": "SIMD prefilter literal extractor",
        "what_it_does": "Extracts required prefixes, suffixes, and inner literals from regexes.",
        "example": '{"pattern": "fetch\\\\w+Async"}',
        "category": "search",
    },
    "reverse-inner-optimizer": {
        "id": "ALGO-SRCH-40",
        "algo_name": "Reverse Suffix & Inner Optimizer",
        "plain_name": "Bidirectional anchored regex scanner",
        "what_it_does": "Scans for rare literals first, then evaluates regex backwards and forwards.",
        "example": '{"text": "class UserException: pass", "pattern": "\\\\w+Exception"}',
        "category": "search",
    },
    "hyperscan-regex-set": {
        "id": "ALGO-SRCH-41",
        "algo_name": "Hyperscan Multi-Regex Set Matcher",
        "plain_name": "Multi-rule parallel regex scanner",
        "what_it_does": "Evaluates hundreds of security/lint rules in a single scan pass.",
        "example": '{"text": "eval(x)", "rules": [{"id": "R1", "pattern": "eval\\\\("}]}',
        "category": "search",
    },
    "redos-protection": {
        "id": "ALGO-SRCH-42",
        "algo_name": "ReDoS Static Analyzer & Tripwire",
        "plain_name": "Catastrophic backtracking detector",
        "what_it_does": "Detects nested quantifiers and exponential explosion hazards in regexes.",
        "example": '{"pattern": "(a+)+$"}',
        "category": "search",
    },
    "inverted-index": {
        "id": "ALGO-SRCH-43",
        "algo_name": "Compressed Inverted Posting Index",
        "plain_name": "Full-text Boolean query resolver",
        "what_it_does": "Maintains term posting lists with AND/OR multi-word intersection.",
        "example": '{"documents": [{"id": "1", "text": "import sys"}], "query_terms": ["import"]}',
        "category": "search",
    },
    "trigram-inverted-index": {
        "id": "ALGO-SRCH-44",
        "algo_name": "Inverted Trigram 3-Gram Index",
        "plain_name": "Corpus-wide substring prefilter",
        "what_it_does": "Indexes 3-grams to prune candidate files for fast grep sweeps.",
        "example": '{"documents": [{"id": "1", "text": "hello world"}], "query": "world"}',
        "category": "search",
    },
    "positional-trigram-index": {
        "id": "ALGO-SRCH-45",
        "algo_name": "Positional Trigram Offset Index",
        "plain_name": "Contiguous offset alignment index",
        "what_it_does": "Verifies adjacent trigram offsets (Zoekt style) for precision filtering.",
        "example": '{"documents": [{"id": "1", "text": "quick brown fox"}], "query": "quick brown"}',
        "category": "search",
    },
    "sparse-ngrams": {
        "id": "ALGO-SRCH-46",
        "algo_name": "Sparse Weighted N-Gram Index",
        "plain_name": "Variable-length entropy n-gram index",
        "what_it_does": "Indexes high-entropy boundary grams for compact code search.",
        "example": '{"text": "AuthenticationManagerMiddlewareHandler"}',
        "category": "search",
    },
    "suffix-array-sais": {
        "id": "ALGO-SRCH-47",
        "algo_name": "Suffix Array & Binary Search",
        "plain_name": "Sorted suffix array for instant search",
        "what_it_does": "Binary searches suffixes in O(M log N) time.",
        "example": '{"text": "banana", "pattern": "an"}',
        "category": "search",
    },
    "lcp-array-kasai": {
        "id": "ALGO-SRCH-48",
        "algo_name": "Kasai Linear LCP Array Builder",
        "plain_name": "Longest common prefix duplicate analyzer",
        "what_it_does": "Finds longest repeated code blocks in linear O(N) time.",
        "example": '{"text": "banana"}',
        "category": "search",
    },
    "suffix-automaton": {
        "id": "ALGO-SRCH-49",
        "algo_name": "Suffix Automaton (DAWG)",
        "plain_name": "Minimal substring automaton",
        "what_it_does": "Accepts all substrings and counts occurrences in strict O(|Query|).",
        "example": '{"text": "abracadabra", "query": "cadabra"}',
        "category": "search",
    },
    "burrows-wheeler-transform": {
        "id": "ALGO-SRCH-50",
        "algo_name": "Burrows-Wheeler Transform & LF-Mapping",
        "plain_name": "Block-sorting compression & search transform",
        "what_it_does": "Performs reversible BWT and LF-mapping for compressed indices.",
        "example": '{"text": "banana", "sentinel": "$"}',
        "category": "search",
    },
    "fm-index": {
        "id": "ALGO-SRCH-51",
        "algo_name": "FM-Index (Compressed Full-Text Index)",
        "plain_name": "Compressed BWT text search with rank count",
        "what_it_does": "Searches and counts occurrences in compressed full-text without decompressing.",
        "example": '{"text": "abracadabra", "pattern": "abra"}',
        "category": "search",
    },
    "trie": {
        "id": "ALGO-SRCH-52",
        "algo_name": "Trie (Prefix Tree)",
        "plain_name": "Fast prefix matching & symbol enumeration",
        "what_it_does": "Finds all keys starting with a prefix in O(|prefix|) time.",
        "example": '{"keys": ["apple", "app", "application"], "prefix": "app"}',
        "category": "search",
    },
    "radix-tree": {
        "id": "ALGO-SRCH-53",
        "algo_name": "Radix (Patricia) Tree",
        "plain_name": "Space-optimized prefix & route matcher",
        "what_it_does": "Compressed prefix tree for longest-prefix match (LPM) and route dispatch.",
        "example": '{"routes": [{"path": "/api/v1/users", "owner": "team_auth"}], "target_path": "/api/v1/users/123"}',
        "category": "search",
    },
    "fst": {
        "id": "ALGO-SRCH-54",
        "algo_name": "Finite State Transducer (FST)",
        "plain_name": "Minimal suffix-sharing dictionary transducer",
        "what_it_does": "Maps ordered string keys to output weights with extreme suffix compression.",
        "example": '{"entries": [["apple", 10], ["apply", 20]], "search_key": "apple"}',
        "category": "search",
    },
    "b-plus-tree": {
        "id": "ALGO-SRCH-55",
        "algo_name": "B+ Tree Index",
        "plain_name": "Balanced search tree with linked leaves",
        "what_it_does": "Self-balancing block index for fast range scans and logarithmic lookups.",
        "example": '{"items": [[10, "A"], [20, "B"], [30, "C"]], "range_start": 10, "range_end": 25}',
        "category": "search",
    },
    "lsm-tree": {
        "id": "ALGO-SRCH-56",
        "algo_name": "Log-Structured Merge Tree (LSM Tree)",
        "plain_name": "Write-optimized index with compaction & tombstones",
        "what_it_does": "Buffers writes in MemTable and flushes immutable SSTables with compaction.",
        "example": '{"operations": [{"op": "put", "key": "k1", "value": "v1"}], "query_key": "k1"}',
        "category": "search",
    },
    "bloom-filter": {
        "id": "ALGO-SRCH-57",
        "algo_name": "Bloom Filter",
        "plain_name": "Probabilistic set membership filter",
        "what_it_does": "Tests set membership with zero false negatives for fast shard pruning.",
        "example": '{"add_items": ["func_a", "func_b"], "query_items": ["func_a", "func_c"]}',
        "category": "search",
    },
    "xor-filter": {
        "id": "ALGO-SRCH-58",
        "algo_name": "XOR / Binary Fuse Filter",
        "plain_name": "Fast 3-way peeling membership filter",
        "what_it_does": "Compact static membership filter using ~9 bits per key with 3 lookups.",
        "example": '{"keys": ["key1", "key2"], "queries": ["key1", "key3"]}',
        "category": "search",
    },
    "posting-list": {
        "id": "ALGO-SRCH-59",
        "algo_name": "Posting List (Block & Positional Index)",
        "plain_name": "Inverted index document & position list",
        "what_it_does": "Stores document IDs and occurrence positions with block skip headers.",
        "example": '{"term": "auth", "postings": [{"doc_id": 1, "positions": [10, 25]}]}',
        "category": "search",
    },
    "delta-gap-encoding": {
        "id": "ALGO-SRCH-60",
        "algo_name": "Delta (Gap) Encoding",
        "plain_name": "Differential integer sequence compressor",
        "what_it_does": "Encodes sorted integers as small differences for high-density compression.",
        "example": '{"integers": [1000, 1003, 1010, 1011]}',
        "category": "search",
    },
    "varint-encoding": {
        "id": "ALGO-SRCH-61",
        "algo_name": "Variable-Byte (Varint / LEB128)",
        "plain_name": "Byte-aligned variable length integer codec",
        "what_it_does": "Encodes integers in 1-10 bytes with ZigZag signed representation.",
        "example": '{"integers": [1, 300, 100000], "signed": false}',
        "category": "search",
    },
    "pfor-delta": {
        "id": "ALGO-SRCH-62",
        "algo_name": "PForDelta Patched Bit-Packing",
        "plain_name": "SIMD-friendly block integer bit packing",
        "what_it_does": "Packs majority values in B bits with auxiliary exception list patching.",
        "example": '{"values": [1, 3, 2, 4, 100, 2, 1], "block_size": 8}',
        "category": "search",
    },
    "elias-fano": {
        "id": "ALGO-SRCH-63",
        "algo_name": "Elias-Fano Quasi-Succinct Encoding",
        "plain_name": "Near-optimal sorted list representation",
        "what_it_does": "Encodes sorted integers with low-bits array and unary high-bits bitvector.",
        "example": '{"integers": [2, 3, 5, 7, 11, 13, 17], "access_index": 2}',
        "category": "search",
    },
    "roaring-bitmap": {
        "id": "ALGO-SRCH-64",
        "algo_name": "Roaring Bitmap",
        "plain_name": "Adaptive 32-bit chunked integer set",
        "what_it_does": "Maintains sparse array and dense bitmap containers for set operations.",
        "example": '{"integers": [1, 2, 3, 65536, 65537], "query_values": [2, 65536, 99]}',
        "category": "search",
    },
    "merge-intersection": {
        "id": "ALGO-SRCH-65",
        "algo_name": "Merge Intersection",
        "plain_name": "Two-pointer posting list intersection",
        "what_it_does": "Intersects sorted integer lists in O(|A| + |B|) linear time.",
        "example": '{"lists": [[1, 3, 5, 7], [3, 5, 8, 10]]}',
        "category": "search",
    },
    "galloping-intersection": {
        "id": "ALGO-SRCH-66",
        "algo_name": "Galloping (Exponential Search) Intersection",
        "plain_name": "Asymmetric jumping posting intersection",
        "what_it_does": "Performs exponential jumps for fast intersection when |A| << |B|.",
        "example": '{"list_a": [5, 100], "list_b": [1, 2, 3, 5, 10, 20, 50, 100, 200]}',
        "category": "search",
    },
    "k-way-merge-heap": {
        "id": "ALGO-SRCH-67",
        "algo_name": "K-Way Merge with Min-Heap",
        "plain_name": "Multi-stream sorted union & deduplication",
        "what_it_does": "Merges K sorted posting streams into globally sorted deduplicated output.",
        "example": '{"streams": [[1, 5, 10], [2, 5, 8], [3, 7, 10]], "deduplicate": true}',
        "category": "search",
    },
    "block-max-wand": {
        "id": "ALGO-SRCH-68",
        "algo_name": "WAND & Block-Max WAND Pruning",
        "plain_name": "Dynamic top-K score threshold pruning",
        "what_it_does": "Skips non-competitive postings using term and block upper bounds.",
        "example": '{"term_postings": {"t1": [[1, 2.5], [2, 1.0]], "t2": [[1, 1.5], [3, 3.0]]}, "k": 2}',
        "category": "search",
    },
    "regex-to-trigram": {
        "id": "ALGO-SRCH-69",
        "algo_name": "Regex to Trigram Query Converter",
        "plain_name": "Translates regex patterns to trigram trees",
        "what_it_does": "Extracts trigrams and boolean AND/OR query trees from regexes.",
        "example": '{"pattern": "(funcA|funcB)Handler"}',
        "category": "search",
    },
    "boolean-query-simplifier": {
        "id": "ALGO-SRCH-70",
        "algo_name": "Boolean Query Simplifier",
        "plain_name": "Rewrites query trees into minimal canonical form",
        "what_it_does": "Applies boolean identities (flattening, absorption, idempotency).",
        "example": '{"query_tree": {"op": "AND", "children": [{"term": "a"}, {"op": "AND", "children": [{"term": "b"}]}]}}',
        "category": "search",
    },
    "rarest-first-ordering": {
        "id": "ALGO-SRCH-71",
        "algo_name": "Rarest-First Query Planner",
        "plain_name": "Orders conjunction terms by selectivity",
        "what_it_does": "Plans multi-term queries evaluating lowest document frequency terms first.",
        "example": '{"term_postings": {"common_term": [1, 2, 3, 4, 5], "rare_term": [2, 5]}}',
        "category": "search",
    },
    "candidate-verification": {
        "id": "ALGO-SRCH-72",
        "algo_name": "Candidate Verification & Pruning",
        "plain_name": "Validates index candidate files against raw content",
        "what_it_does": "Verifies candidate files and computes exact line/byte offsets.",
        "example": '{"candidates": [{"path": "main.py", "content": "def hello(): pass"}], "pattern": "hello"}',
        "category": "search",
    },
    "early-termination": {
        "id": "ALGO-SRCH-73",
        "algo_name": "Early Termination Search Bouncer",
        "plain_name": "Enforces result count and byte scan budgets",
        "what_it_does": "Terminates search when budget thresholds are reached and sets truncation flags.",
        "example": '{"items": [{"file": "a.py", "text": "hit1"}, {"file": "a.py", "text": "hit2"}], "max_total_results": 1}',
        "category": "search",
    },
    "scatter-gather": {
        "id": "ALGO-SRCH-74",
        "algo_name": "Scatter-Gather Multi-Shard Search",
        "plain_name": "Distributed parallel shard query aggregator",
        "what_it_does": "Dispatches queries to shards, gathers results, and isolates timeouts.",
        "example": '{"shard_responses": [{"shard_id": "s1", "status": "ok", "results": [{"id": 1, "score": 0.9}]}]}',
        "category": "search",
    },
    "hedged-requests": {
        "id": "ALGO-SRCH-75",
        "algo_name": "Hedged Requests Dispatcher",
        "plain_name": "Tail-latency mitigating speculative queries",
        "what_it_does": "Dispatches backup duplicate query if primary exceeds p95 delay.",
        "example": '{"primary_latency_ms": 120.0, "backup_latency_ms": 30.0, "hedge_delay_ms": 50.0}',
        "category": "search",
    },
    "index-versioned-cache": {
        "id": "ALGO-SRCH-76",
        "algo_name": "Index-Versioned Query Cache",
        "plain_name": "LRU cache keyed by commit version & query",
        "what_it_does": "Auto-invalidates cached results when commit/index version shifts.",
        "example": '{"action": "put", "query": "find_users", "index_version": "v1", "data": {"hits": [1, 2]}}',
        "category": "search",
    },
    "tokenizer-search": {
        "id": "ALGO-SRCH-77",
        "algo_name": "Tokenizer-Based Code Search",
        "plain_name": "Lexical token-stream search filtering by kind",
        "what_it_does": "Matches identifiers while filtering comments and string literals.",
        "example": '{"code": "def user(): # comment user\\n  return 1", "query": "user", "allowed_kinds": ["IDENTIFIER"]}',
        "category": "search",
    },
    "cst-parser": {
        "id": "ALGO-SRCH-78",
        "algo_name": "Concrete Syntax Tree (CST) Parser",
        "plain_name": "Trivia-preserving lossless syntax tree",
        "what_it_does": "Preserves comments and whitespace for byte-exact code edits.",
        "example": '{"code": "x = 1 # keep comment\\n", "rename_from": "x", "rename_to": "y"}',
        "category": "search",
    },
    "ast-analyzer": {
        "id": "ALGO-SRCH-79",
        "algo_name": "Abstract Syntax Tree (AST) Analyzer",
        "plain_name": "Semantic AST node & structure extractor",
        "what_it_does": "Extracts functions, classes, and call expressions from code.",
        "example": '{"code": "def add(a, b):\\n  return a + b"}',
        "category": "search",
    },
    "incremental-parser": {
        "id": "ALGO-SRCH-80",
        "algo_name": "Incremental Error-Tolerant Parser",
        "plain_name": "Tree-sitter style incremental AST parser",
        "what_it_does": "Re-parses modified ranges while reusing untouched syntax subtrees.",
        "example": '{"code": "def valid(): pass\\ndef broken(: pass"}',
        "category": "search",
    },
    "tree-sitter-query": {
        "id": "ALGO-SRCH-81",
        "algo_name": "Tree-Sitter S-Expression Query Matcher",
        "plain_name": "S-expression pattern matcher with node captures",
        "what_it_does": "Matches AST shapes and captures callee names and arguments.",
        "example": '{"code": "fetch(url, timeout=10)", "target_name": "fetch"}',
        "category": "search",
    },
    "ast-grep-pattern": {
        "id": "ALGO-SRCH-82",
        "algo_name": "AST-Grep Pattern & Metavariable Matcher",
        "plain_name": "Code pattern matching with $VAR placeholders",
        "what_it_does": "Matches structural patterns and applies rewrite templates.",
        "example": '{"code": "api.get(url)", "pattern": "$OBJ.get($URL)", "rewrite": "$OBJ.post($URL)"}',
        "category": "search",
    },
    "semgrep-equivalence": {
        "id": "ALGO-SRCH-83",
        "algo_name": "Semgrep-Style Equivalence Matcher",
        "plain_name": "Semantic matching with import alias tracking",
        "what_it_does": "Resolves import aliases and negative exclusion patterns.",
        "example": '{"code": "import os.system as run\\nrun(\'ls\')", "target_api": "os.system"}',
        "category": "search",
    },
    "comby-delimiter": {
        "id": "ALGO-SRCH-84",
        "algo_name": "Comby Balanced-Delimiter Matcher",
        "plain_name": "Language-agnostic balanced bracket replacer",
        "what_it_does": "Matches nested parentheses and replaces calls safely.",
        "example": '{"code": "old_fn(a, (b + c))", "target_func": "old_fn", "replacement_func": "new_fn"}',
        "category": "search",
    },
    "subtree-hash-clone": {
        "id": "ALGO-SRCH-85",
        "algo_name": "AST Subtree Hashing (Clone Detector)",
        "plain_name": "Merkle-tree AST clone & pattern detector",
        "what_it_does": "Hashes AST nodes bottom-up to find duplicate logic and clones.",
        "example": '{"code": "def f1(x): return x + 1\\ndef f2(y): return y + 1"}',
        "category": "search",
    },
    "gumtree-diff": {
        "id": "ALGO-SRCH-86",
        "algo_name": "GumTree Structural AST Diff",
        "plain_name": "Structural node update, insert, delete, move diff",
        "what_it_does": "Computes AST edit scripts across two code revisions.",
        "example": '{"old_code": "def foo(): pass", "new_code": "def bar(): pass"}',
        "category": "search",
    },
    "symbol-table": {
        "id": "ALGO-SRCH-87",
        "algo_name": "Hierarchical Symbol Table",
        "plain_name": "Lexical scope stack & identifier resolver",
        "what_it_does": "Resolves variable scopes and detects shadowed identifiers.",
        "example": '{"declarations": [{"action": "define", "name": "x", "kind": "var"}], "lookups": ["x"]}',
        "category": "search",
    },
    "scope-graph": {
        "id": "ALGO-SRCH-88",
        "algo_name": "Scope Graph Name Resolver",
        "plain_name": "Path-based scope & declaration graph",
        "what_it_does": "Resolves reference nodes across nested scopes and import edges.",
        "example": '{"nodes": [{"id": "s1", "type": "Scope"}, {"id": "d1", "type": "Declaration", "label": "foo"}, {"id": "r1", "type": "Reference", "label": "foo"}], "edges": [{"source": "s1", "target": "d1", "type": "D"}, {"source": "r1", "target": "s1", "type": "R"}], "references_to_resolve": ["r1"]}',
        "category": "search",
    },
    "stack-graph": {
        "id": "ALGO-SRCH-89",
        "algo_name": "Stack Graph Incremental Resolver",
        "plain_name": "Push/pop symbol stack path resolver",
        "what_it_does": "Resolves cross-file qualified symbols via push/pop stack paths.",
        "example": '{"edges": [{"source": "ref", "target": "mid", "action": "push", "symbol": "foo"}, {"source": "mid", "target": "def", "action": "pop", "symbol": "foo"}], "start_node": "ref", "target_node": "def"}',
        "category": "search",
    },
    "lsp-protocol": {
        "id": "ALGO-SRCH-90",
        "algo_name": "LSP Protocol JSON-RPC Handler",
        "plain_name": "Language Server Protocol message builder",
        "what_it_does": "Generates standard LSP definition/reference requests and parses diagnostics.",
        "example": '{"action": "build_definition", "uri": "file:///app.py", "line": 10, "character": 5}',
        "category": "search",
    },
    "scip-lsif-index": {
        "id": "ALGO-SRCH-91",
        "algo_name": "SCIP & LSIF Symbol Index",
        "plain_name": "Precomputed code intelligence index",
        "what_it_does": "Indexes symbol definition and reference occurrences with global URIs.",
        "example": '{"occurrences": [{"document_uri": "main.py", "symbol_uri": "pkg mod func.", "role": "definition"}], "query_symbol": "pkg mod func."}',
        "category": "search",
    },
    "call-graph": {
        "id": "ALGO-SRCH-92",
        "algo_name": "Call Graph Impact Analyzer",
        "plain_name": "Inter-procedural function call graph",
        "what_it_does": "Extracts call edges and finds transitive upstream callers.",
        "example": '{"edges": [{"caller": "main", "callee": "helper"}, {"caller": "helper", "callee": "core"}], "target_function": "core"}',
        "category": "search",
    },
    "import-dependency-graph": {
        "id": "ALGO-SRCH-93",
        "algo_name": "Import Dependency Graph & Cycle Detector",
        "plain_name": "Module import graph with topological sort",
        "what_it_does": "Analyzes module dependencies, detects circular cycles, and sorts topologically.",
        "example": '{"dependencies": [["app", "utils"], ["utils", "config"]], "target_module": "config"}',
        "category": "search",
    },
    "control-flow-graph": {
        "id": "ALGO-SRCH-94",
        "algo_name": "Control Flow Graph (CFG)",
        "plain_name": "Basic block partitioner & jump analyzer",
        "what_it_does": "Decomposes code into basic blocks and detects unreachable code paths.",
        "example": '{"statements": ["x = 1", "if x > 0:", "y = 2", "return y"]}',
        "category": "search",
    },
    "ssa-form": {
        "id": "ALGO-SRCH-95",
        "algo_name": "Static Single Assignment (SSA) Converter",
        "plain_name": "Versioned variables & phi-node inserter",
        "what_it_does": "Converts assignments into versioned single assignments with phi functions.",
        "example": '{"statements": [{"type": "assign", "var": "x", "uses": []}, {"type": "assign", "var": "x", "uses": ["x"]}]}',
        "category": "search",
    },
    "dataflow-worklist": {
        "id": "ALGO-SRCH-96",
        "algo_name": "Dataflow Worklist Iteration Engine",
        "plain_name": "Reaching definitions & lattice fixpoint solver",
        "what_it_does": "Solves dataflow lattices using GEN/KILL sets and worklist queue.",
        "example": '{"blocks": [{"id": 0, "gen": ["d1"], "kill": []}], "successors": {"0": []}, "predecessors": {"0": []}}',
        "category": "search",
    },
    "taint-analysis": {
        "id": "ALGO-SRCH-97",
        "algo_name": "Taint Analysis Vulnerability Engine",
        "plain_name": "Source-to-sink taint propagation tracker",
        "what_it_does": "Traces untrusted data flow into sinks and verifies sanitizers.",
        "example": '{"statements": [{"type": "assign", "target": "param", "inputs": ["user_input"]}, {"type": "call", "func": "sql_execute", "inputs": ["param"]}], "sources": ["user_input"], "sinks": ["sql_execute"]}',
        "category": "search",
    },
    "datalog-codeql": {
        "id": "ALGO-SRCH-98",
        "algo_name": "Datalog CodeQL Relational Engine",
        "plain_name": "Deductive database & recursive rule evaluator",
        "what_it_does": "Evaluates relational facts and computes transitive reachability closures.",
        "example": '{"facts": [{"relation": "call", "args": ["a", "b"]}, {"relation": "call", "args": ["b", "c"]}], "query_relation": "call"}',
        "category": "search",
    },
    "ast-chunking": {
        "id": "ALGO-SRCH-99",
        "algo_name": "AST Syntax-Aware Code Chunker",
        "plain_name": "Semantic function & class boundary chunker",
        "what_it_does": "Partitions code along function/class AST nodes with rich context metadata.",
        "example": '{"file_path": "server.py", "code": "def start():\\n  pass\\n\\ndef stop():\\n  pass"}',
        "category": "search",
    },
    "code-embeddings": {
        "id": "ALGO-SRCH-100",
        "algo_name": "Lexical-Semantic Code Embeddings",
        "plain_name": "Subtoken-hashed dense vector embedding generator",
        "what_it_does": "Generates L2-normalized dense embedding vectors and computes cosine similarity.",
        "example": '{"code": "def process_payment(amount): return amount", "query": "payment processing logic"}',
        "category": "search",
    },
    "ast-parse": {
        "id": "ALGO-OBS-16",
        "algo_name": "Tree-Sitter AST Parser",
        "plain_name": "Parse source code into a structured tree",
        "what_it_does": "Converts source code into a tree of functions, classes, and expressions — the foundation for all code analysis.",
        "example": '{"file_path": "src/api/cli/main.py", "language": "python"}',
        "category": "observability",
    },
    "outline": {
        "id": "ALGO-OBS-17",
        "algo_name": "Code Outline Generator",
        "plain_name": "Show a file's structure at a glance",
        "what_it_does": "Extracts all functions, classes, and methods from a file — like a table of contents for source code.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "observability",
    },
    "extract-comments": {
        "id": "ALGO-OBS-18",
        "algo_name": "Comment Extractor",
        "plain_name": "Pull out every comment and docstring",
        "what_it_does": "Collects every comment, inline note, and docstring from a source file into one list.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "observability",
    },
    "no-inline": {
        "id": "ALGO-OBS-19",
        "algo_name": "Zero-Inline-Comment Linter",
        "plain_name": "Check for banned inline comments in code",
        "what_it_does": "Flags code that has inline comments — enforces the rule that all docs must be in the file header, not inline.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "observability",
    },
    "dep-graph": {
        "id": "ALGO-OBS-20",
        "algo_name": "Import Dependency Grapher",
        "plain_name": "Map what each module imports from what",
        "what_it_does": "Builds a graph of all module imports in a codebase and detects circular dependencies that can break builds.",
        "example": '{"directory": "src"}',
        "category": "observability",
    },
    "symbols": {
        "id": "ALGO-OBS-21",
        "algo_name": "Symbol Scope Resolver",
        "plain_name": "Find where variables and functions are defined",
        "what_it_does": "Traces every variable and function name back to where it was defined — for understanding code ownership.",
        "example": '{"file_path": "src/api/cli/main.py"}',
        "category": "observability",
    },
    "smart-patch": {
        "id": "ALGO-UPD-22",
        "algo_name": "CST Matcher / Patcher",
        "plain_name": "Edit code without breaking its structure",
        "what_it_does": "Edits code using the syntax tree so renames and replacements are structure-aware — no accidental breakage.",
        "example": '{"file_path": "src/api/cli/main.py", "find_pattern": "old_name", "replace_text": "new_name", "dry_run": true}',
        "category": "update",
    },
    "batch-patch": {
        "id": "ALGO-UPD-23",
        "algo_name": "Batch Patcher",
        "plain_name": "Edit many files at once safely",
        "what_it_does": "Applies find-and-replace across many files at once with a SHA-256 checksum and rollback on any failure.",
        "example": '{"operations": [{"file_path": "src/api/cli/main.py", "find_pattern": "TODO", "replace_text": "DONE", "is_regex": false}], "dry_run": true}',
        "category": "update",
    },
    "show-diff": {
        "id": "ALGO-UPD-24",
        "algo_name": "Unified Diff Engine",
        "plain_name": "Show exactly what changed between two versions",
        "what_it_does": "Generates a standard unified diff (+/-) showing every line added or removed between two texts.",
        "example": '{"original": "hello world", "modified": "hello earth", "file_path": "test.py"}',
        "category": "update",
    },
    "vec-gemm": {
        "id": "ALGO-VEC-SRCH-51",
        "algo_name": "Brute-Force kNN with GEMM",
        "plain_name": "Exact nearest neighbor search using matrix math",
        "what_it_does": "Scores query against all database vectors in one fast BLAS matrix multiply — the gold standard ground truth.",
        "example": '{"database_vectors": [[1,0],[0,1]], "query_vectors": [[1,0]], "k": 2}',
        "category": "vector",
    },
    "vec-simd": {
        "id": "ALGO-VEC-SRCH-52",
        "algo_name": "SIMD Distance Kernels",
        "plain_name": "Hardware-accelerated vector distance calculator",
        "what_it_does": "Computes L2, dot, cosine, or binary Hamming distance using CPU SIMD vector lanes.",
        "example": '{"vector_a": [1,2,3,4], "vector_b": [1,2,3,5], "metric": "l2"}',
        "category": "vector",
    },
    "vec-topk": {
        "id": "ALGO-VEC-SRCH-53",
        "algo_name": "Heap Top-K Selection",
        "plain_name": "Find the k best items without sorting everything",
        "what_it_does": "Maintains a bounded heap of size k to pick the highest or lowest scoring candidates with low memory.",
        "example": '{"candidates": [{"id": "doc1", "score": 10.5}, {"id": "doc2", "score": 20.0}], "k": 1}',
        "category": "vector",
    },
    "vec-radix": {
        "id": "ALGO-VEC-SRCH-54",
        "algo_name": "Radix / Quickselect Top-K",
        "plain_name": "Ultra-fast bucket partitioning for top scores",
        "what_it_does": "Selects top-k scores in linear time without sorting, inspired by GPU radix selection algorithms.",
        "example": '{"scores": [10.0, 50.0, 20.0, 40.0], "k": 2}',
        "category": "vector",
    },
    "vec-early-abandon": {
        "id": "ALGO-VEC-SRCH-55",
        "algo_name": "Early Abandoning Distance",
        "plain_name": "Stop distance calculation early if candidate is too far",
        "what_it_does": "Halts math as soon as partial distance exceeds current best threshold — saves up to 70% of calculations.",
        "example": '{"database_vectors": [[0,0],[5,5]], "query_vector": [0,0], "k": 1}',
        "category": "vector",
    },
    "vec-pivot": {
        "id": "ALGO-VEC-SRCH-56",
        "algo_name": "Pivot Triangle-Inequality Pruning",
        "plain_name": "Skip far vectors using reference landmark points",
        "what_it_does": "Uses precalculated distances to landmark pivots to rule out distant vectors without full calculations.",
        "example": '{"database_vectors": [[0,0],[1,1],[10,10]], "pivots": [[0,0]], "query_vector": [0.1,0.1], "k": 1}',
        "category": "vector",
    },
    "vec-kdtree": {
        "id": "ALGO-VEC-SRCH-57",
        "algo_name": "KD-Tree Spatial Index",
        "plain_name": "Fast spatial search tree across dimensions",
        "what_it_does": "Recursively splits coordinate axes to search points in logarithmic time instead of scanning everything.",
        "example": '{"vectors": [[2,3],[5,4],[9,6],[4,7]], "query": [9,5], "k": 2}',
        "category": "vector",
    },
    "vec-balltree": {
        "id": "ALGO-VEC-SRCH-58",
        "algo_name": "Ball Tree Metric Index",
        "plain_name": "Group vectors into nested bounding spheres",
        "what_it_does": "Encloses points into spherical clusters to beat the curse of dimensionality in higher dimensions.",
        "example": '{"vectors": [[0,0],[1,1],[5,5],[10,10]], "query": [0.9,0.9], "k": 1}',
        "category": "vector",
    },
    "vec-vptree": {
        "id": "ALGO-VEC-SRCH-59",
        "algo_name": "Vantage-Point Tree",
        "plain_name": "Split metric space using concentric shells",
        "what_it_does": "Partitions data into spherical shells around vantage points — works in any metric distance space.",
        "example": '{"vectors": [[0,0],[1,1],[5,5],[10,10]], "query": [5.1,5.1], "k": 1}',
        "category": "vector",
    },
    "vec-rpforest": {
        "id": "ALGO-VEC-SRCH-60",
        "algo_name": "Random Projection Tree Forest (Annoy-style)",
        "plain_name": "Fast approximate search using random planes",
        "what_it_does": "Splits space with random cutting planes across multiple trees for fast approximate nearest neighbor search.",
        "example": '{"vectors": [[0,0],[1,2],[2,4],[3,6]], "query": [1,2], "k": 2, "num_trees": 4}',
        "category": "vector",
    },
    "vec-ivf": {
        "id": "ALGO-VEC-SRCH-61",
        "algo_name": "Inverted File Index (IVF)",
        "plain_name": "Group vectors into Voronoi buckets and search top cells",
        "what_it_does": "Clusters vectors with k-means and only searches the closest cluster cells (nprobe) at query time.",
        "example": '{"vectors": [[0,0],[1,1],[5,5],[6,6]], "query": [0.5,0.5], "k": 2, "num_clusters": 2, "nprobe": 1}',
        "category": "vector",
    },
    "vec-ivfpq": {
        "id": "ALGO-VEC-SRCH-62",
        "algo_name": "IVF with Product Quantization (IVF-PQ)",
        "plain_name": "Search billions of compressed vectors in RAM",
        "what_it_does": "Combines cluster routing with sub-vector byte compression and fast distance lookup tables.",
        "example": '{"vectors": [[1,2,3,4],[5,6,7,8]], "query": [1,2,3,4], "k": 1, "num_clusters": 2, "nprobe": 1}',
        "category": "vector",
    },
    "vec-nprobe": {
        "id": "ALGO-VEC-SRCH-63",
        "algo_name": "nprobe Multi-Probe Tuner",
        "plain_name": "Automatically find the best balance of speed vs accuracy",
        "what_it_does": "Sweeps nprobe values against exact ground truth to find the fastest setting meeting your recall goal.",
        "example": '{"database_vectors": [[1,1],[2,2],[3,3],[4,4]], "sample_queries": [[1.1,1.1]], "target_recall": 0.9}',
        "category": "vector",
    },
    "vec-imi": {
        "id": "ALGO-VEC-SRCH-64",
        "algo_name": "Inverted Multi-Index (IMI)",
        "plain_name": "Ultra-fine dual-codebook Voronoi search",
        "what_it_does": "Splits vectors into halves and pairs centroids to create millions of fine cells with tiny memory.",
        "example": '{"vectors": [[1,2,3,4],[5,6,7,8]], "query": [1,2,3,4], "k": 1, "codebook_k1": 2, "codebook_k2": 2}',
        "category": "vector",
    },
    "vec-nsw": {
        "id": "ALGO-VEC-SRCH-65",
        "algo_name": "Navigable Small World (NSW) Graph",
        "plain_name": "Fast proximity graph routing with highway shortcuts",
        "what_it_does": "Navigates a small-world proximity network using greedy routing from long highway edges to local neighbors.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "query": [1,2], "k": 2}',
        "category": "vector",
    },
    "vec-hnsw-search": {
        "id": "ALGO-VEC-SRCH-66",
        "algo_name": "Hierarchical NSW Search",
        "plain_name": "Logarithmic multi-layer vector graph search",
        "what_it_does": "Explores coarse sparse layers greedily down to layer 0 for bounded beam search with ef candidates.",
        "example": '{"vectors": [[1,2],[3,4]], "layers": [{"0": [1], "1": [0]}], "entry_point": 0, "top_layer": 0, "query": [1,2], "k": 1}',
        "category": "vector",
    },
    "vec-hnsw-insert": {
        "id": "ALGO-VEC-SRCH-67",
        "algo_name": "HNSW Layer Insertion & Diversity Pruning",
        "plain_name": "Scale-free graph insertion with directional link pruning",
        "what_it_does": "Builds multi-layer scale-free graphs, pruning links that point in redundant directions.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "m": 2, "ef_construction": 8}',
        "category": "vector",
    },
    "vec-beam": {
        "id": "ALGO-VEC-SRCH-68",
        "algo_name": "Bounded Graph Beam Search",
        "plain_name": "Priority queue exploration with ef budget limit",
        "what_it_does": "Traverses graph edges using two bounded heaps, stopping when unexpanded nodes are farther than worst result.",
        "example": '{"vectors": [[1,2],[3,4]], "adjacency": {"0": [1], "1": [0]}, "start_nodes": [0], "query": [1,2], "k": 1}',
        "category": "vector",
    },
    "vec-vamana": {
        "id": "ALGO-VEC-SRCH-69",
        "algo_name": "Vamana / DiskANN Graph Index",
        "plain_name": "Single-layer graph index for billion-scale SSD search",
        "what_it_does": "Constructs two-pass graph index from medoid with RobustPrune for fast disk-resident retrieval.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "query": [1,2], "k": 1, "r_max_degree": 4}',
        "category": "vector",
    },
    "vec-robust-prune": {
        "id": "ALGO-VEC-SRCH-70",
        "algo_name": "RobustPrune Alpha Diversity Filter",
        "plain_name": "Eliminate redundant neighbor paths while preserving long-range shortcuts",
        "what_it_does": "Prunes candidate neighbors using alpha triangle inequality to maintain high connectivity with bounded degree.",
        "example": '{"point": [0,0], "candidate_vectors": [[1,0],[0.5,0.1],[5,5]], "alpha": 1.2, "r_max_degree": 2}',
        "category": "vector",
    },
    "vec-nsg": {
        "id": "ALGO-VEC-SRCH-71",
        "algo_name": "Navigating Spreading-out Graph (NSG)",
        "plain_name": "Monotonic relative neighborhood graph with high recall",
        "what_it_does": "Prunes kNN graphs with MRNG rule and repairs connectivity from a central navigating node.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "query": [1,2], "k": 1}',
        "category": "vector",
    },
    "vec-cagra": {
        "id": "ALGO-VEC-SRCH-72",
        "algo_name": "CAGRA Fixed-Degree Graph",
        "plain_name": "GPU-optimized regularized proximity graph",
        "what_it_does": "Transforms proximity graph into strict fixed-degree layout for parallel GPU batch search.",
        "example": '{"vectors": [[1,2],[3,4],[5,6],[7,8]], "query": [1,2], "k": 1, "fixed_degree": 2}',
        "category": "vector",
    },
    "vec-entry": {
        "id": "ALGO-VEC-SRCH-73",
        "algo_name": "Entry-Point Selection",
        "plain_name": "Pick geometric medoid or multi-seed starting points",
        "what_it_does": "Computes medoid or furthest-point sampling seeds to prevent graph trapping on multi-modal datasets.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "strategy": "medoid"}',
        "category": "vector",
    },
    "vec-repair": {
        "id": "ALGO-VEC-SRCH-74",
        "algo_name": "Graph Connectivity Repair",
        "plain_name": "Detect unreachable islands and reconnect isolated nodes",
        "what_it_does": "Performs BFS reachability sweep and links unreachable nodes to their nearest reachable neighbors.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "adjacency": {"0": [], "1": [], "2": []}, "entry_points": [0]}',
        "category": "vector",
    },
    "vec-filtered-diskann": {
        "id": "ALGO-VEC-SRCH-75",
        "algo_name": "Filtered-DiskANN",
        "plain_name": "Strict label-constrained in-index vector search",
        "what_it_does": "Traverses graph paths starting from label-specific medoid, ensuring only nodes matching target label are inspected.",
        "example": '{"vectors": [[1,2],[3,4]], "labels": ["tenantA", "tenantB"], "query": [1,2], "target_label": "tenantA", "k": 1}',
        "category": "vector",
    },
    "vec-spann": {
        "id": "ALGO-VEC-SRCH-76",
        "algo_name": "SPANN Memory-Disk Hybrid Index",
        "plain_name": "Centroids in RAM and boundary-duplicated postings on disk",
        "what_it_does": "Clusters vectors with boundary duplication into adjacent posting lists for fast sequential disk reads.",
        "example": '{"vectors": [[1,2],[3,4],[5,6],[7,8]], "query": [1,2], "k": 1, "num_centroids": 2, "nprobe": 1}',
        "category": "vector",
    },
    "vec-lsh-hyperplane": {
        "id": "ALGO-VEC-SRCH-77",
        "algo_name": "Random-Hyperplane LSH",
        "plain_name": "Locality-sensitive cosine hashing with random hyperplanes",
        "what_it_does": "Projects vectors onto random hyperplanes to generate bit signatures for fast sub-linear bucket lookup.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "query": [1,2], "k": 1, "num_bits": 2, "num_tables": 2}',
        "category": "vector",
    },
    "vec-lsh-multiprobe": {
        "id": "ALGO-VEC-SRCH-78",
        "algo_name": "Multi-Probe LSH",
        "plain_name": "Check nearby perturbed hash buckets for high recall",
        "what_it_does": "Probes primary and perturbed bit buckets ranked by proximity to hyperplane boundary.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "query": [1,2], "k": 1, "num_bits": 4, "probe_budget": 2}',
        "category": "vector",
    },
    "vec-e2lsh": {
        "id": "ALGO-VEC-SRCH-79",
        "algo_name": "Exact Euclidean L2 LSH (E2LSH)",
        "plain_name": "P-stable Gaussian random projection hashing for Euclidean space",
        "what_it_does": "Uses 2-stable Gaussian projections and uniform quantization slots for theoretical Euclidean collision guarantees.",
        "example": '{"vectors": [[1,2],[3,4],[5,6]], "query": [1,2], "k": 1, "slot_width_w": 4.0}',
        "category": "vector",
    },
    "vec-filter-pre": {
        "id": "ALGO-VEC-FLTR-80",
        "algo_name": "Vector Metadata Pre-Filtering",
        "plain_name": "Filter candidates before scoring vectors",
        "what_it_does": "Evaluates metadata predicates first to ensure zero security leaks before distance computation.",
        "example": '{"vectors": [[0,0],[1,1]], "metadata": [{"tenant":"A"},{"tenant":"B"}], "query": [0.1,0.1], "filters": {"tenant":"A"}, "k": 1}',
        "category": "filter",
    },
    "vec-filter-post": {
        "id": "ALGO-VEC-FLTR-81",
        "algo_name": "Vector Post-Filtering with Oversampling",
        "plain_name": "Search first with extra items then remove non-matching",
        "what_it_does": "Retrieves k * f candidates from index and rejects records that fail soft non-security filters.",
        "example": '{"vectors": [[0,0],[1,1]], "metadata": [{"lang":"en"},{"lang":"fr"}], "query": [0.1,0.1], "filters": {"lang":"en"}, "k": 1, "oversample_factor": 2.0}',
        "category": "filter",
    },
    "vec-filter-in-graph": {
        "id": "ALGO-VEC-FLTR-82",
        "algo_name": "ACORN In-Graph Filtering",
        "plain_name": "Hop across filtered graph nodes without breaking connectivity",
        "what_it_does": "Traverses graph routing paths while strictly admitting only allowed nodes into top-k candidates.",
        "example": '{"vectors": [[0,0],[1,0],[2,0]], "metadata": [{"status":"active"},{"status":"disabled"},{"status":"active"}], "adjacency": {"0":[1],"1":[2],"2":[]}, "entry_point": 0, "query": [1.9,0.0], "filters": {"status":"active"}, "k": 1}',
        "category": "filter",
    },
    "vec-filter-plan": {
        "id": "ALGO-VEC-FLTR-83",
        "algo_name": "Selectivity-Based Query Planner",
        "plain_name": "Choose best search strategy from filter selectivity",
        "what_it_does": "Estimates match ratio and picks pre-filtering, in-graph ACORN traversal, or post-filtering.",
        "example": '{"total_vectors": 10000, "metadata_sample": [{"tenant":"T1"},{"tenant":"T2"}], "filters": {"tenant":"T1"}, "is_security_filter": true}',
        "category": "filter",
    },
    "vec-filter-partition": {
        "id": "ALGO-VEC-FLTR-84",
        "algo_name": "Partitioned Multi-Tenant Index Search",
        "plain_name": "Route query strictly into tenant or collection slice",
        "what_it_does": "Directs search exclusively into the isolated physical partition with zero cross-tenant contamination.",
        "example": '{"partitions": {"tenant_A": [{"id":"d1","vector":[1,0],"metadata":{}}]}, "target_partition": "tenant_A", "query": [0.9,0.1], "k": 1}',
        "category": "filter",
    },
    "vec-bm25": {
        "id": "ALGO-VEC-SRCH-85",
        "algo_name": "Okapi BM25 Keyword Search",
        "plain_name": "Fast keyword search using term frequencies and document lengths",
        "what_it_does": "Scores documents with probabilistic TF-IDF and field length normalization.",
        "example": '{"corpus": ["hello world document", "another text"], "query": "hello", "k": 1}',
        "category": "vector",
    },
    "vec-hybrid": {
        "id": "ALGO-VEC-SRCH-86",
        "algo_name": "Sparse-Dense Hybrid Search",
        "plain_name": "Combine keyword search and semantic vector search",
        "what_it_does": "Blends lexical sparse keyword scores with dense vector similarities via linear interpolation.",
        "example": '{"dense_results": [{"id": "1", "score": 0.8}], "sparse_results": [{"id": "1", "score": 12.0}], "alpha": 0.5, "k": 1}',
        "category": "vector",
    },
    "vec-rrf": {
        "id": "ALGO-VEC-SRCH-87",
        "algo_name": "Reciprocal Rank Fusion",
        "plain_name": "Merge multiple ranking lists using reciprocal rank scores",
        "what_it_does": "Combines ranked lists without score calibration: score = sum(1 / (k + rank)).",
        "example": '{"rankings": [[{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}]], "k_rrf": 60, "top_k": 2}',
        "category": "vector",
    },
    "vec-convex-fusion": {
        "id": "ALGO-VEC-SRCH-88",
        "algo_name": "Convex Score Fusion",
        "plain_name": "Weighted combination of normalized retrieval scores",
        "what_it_does": "Normalizes score scales (min-max or z-score) and fuses candidate channels with tuned weights.",
        "example": '{"score_lists": [[{"id": "1", "score": 0.9}], [{"id": "1", "score": 15.0}]], "norm_method": "minmax", "top_k": 1}',
        "category": "vector",
    },
    "vec-mmr": {
        "id": "ALGO-VEC-SRCH-89",
        "algo_name": "Maximal Marginal Relevance",
        "plain_name": "Balance relevance and diversity to prevent duplicate passages",
        "what_it_does": "Reranks candidates iteratively penalizing similarity to already chosen candidates.",
        "example": '{"candidate_vectors": [[1,0],[0.99,0.01],[0,1]], "candidate_ids": ["a","b","c"], "query_vector": [1,0], "lambda_mult": 0.7, "k": 2}',
        "category": "vector",
    },
    "vec-range": {
        "id": "ALGO-VEC-SRCH-90",
        "algo_name": "Range Radius Search",
        "plain_name": "Find all vectors within distance threshold rather than fixed top-k",
        "what_it_does": "Returns every neighbor within distance radius r, bounded by maximum allowed results.",
        "example": '{"vectors": [[0,0],[1,1],[5,5]], "query": [0,0], "radius": 1.5, "metric": "l2"}',
        "category": "vector",
    },
    "vec-maxsim": {
        "id": "ALGO-VEC-SRCH-91",
        "algo_name": "ColBERT Late-Interaction MaxSim",
        "plain_name": "Fine-grained token-level vector matching across query and document",
        "what_it_does": "Scores documents by summing maximum similarity of each query token across all document tokens.",
        "example": '{"document_token_vectors": [[[1,0],[0,1]]], "query_token_vectors": [[1,0]], "k": 1}',
        "category": "vector",
    },
    "vec-multi-query": {
        "id": "ALGO-VEC-SRCH-92",
        "algo_name": "Multi-Query Expansion",
        "plain_name": "Rewrite query into variants, search all, and fuse results",
        "what_it_does": "Runs multiple query variations in parallel and combines hits using rank fusion.",
        "example": '{"vectors": [[1,0],[0,1]], "expanded_queries": [[1,0],[0.9,0.1]], "k": 1}',
        "category": "vector",
    },
    "vec-rescore": {
        "id": "ALGO-VEC-SRCH-93",
        "algo_name": "Full Precision Rescoring",
        "plain_name": "Retrieve cheap candidates then re-score exact distances",
        "what_it_does": "Fetches uncompressed vectors for top compressed candidates to restore full recall.",
        "example": '{"candidate_ids": ["1"], "full_precision_vectors": {"1": [1.0, 0.0]}, "query_vector": [1.0, 0.0], "top_k": 1}',
        "category": "vector",
    },
    "vec-cross-encoder": {
        "id": "ALGO-VEC-SRCH-94",
        "algo_name": "Cross-Encoder Transformer Reranker",
        "plain_name": "Re-score top candidates by feeding query and passage jointly",
        "what_it_does": "Applies full cross-attention scoring across top candidate texts for maximum precision.",
        "example": '{"query": "machine learning", "candidates": [{"id": "1", "text": "ml intro"}], "top_k": 1}',
        "category": "vector",
    },
    "vec-funnel": {
        "id": "ALGO-VEC-SRCH-95",
        "algo_name": "Multi-Stage Retrieval Funnel",
        "plain_name": "Cascaded progressive ranking across candidate subsets",
        "what_it_does": "Applies cheap coarse search, intermediate rescoring, and precise final ranking stages.",
        "example": '{"stage1_candidates": [{"id": "1", "score": 0.9}], "stage2_top_m": 10, "stage3_top_k": 1}',
        "category": "vector",
    },
    "vec-llm-rerank": {
        "id": "ALGO-VEC-SRCH-96",
        "algo_name": "LLM Listwise Reranker",
        "plain_name": "Order small candidate list using language model reasoning",
        "what_it_does": "Prompts a language model with candidate summaries and strictly parses ranked order.",
        "example": '{"query": "best database", "candidates": [{"id": "1", "text": "Postgres is great"}], "top_k": 1}',
        "category": "vector",
    },
    "vec-hyde": {
        "id": "ALGO-VEC-SRCH-97",
        "algo_name": "Hypothetical Document Embeddings",
        "plain_name": "Generate hypothetical answer, embed it, and retrieve documents",
        "what_it_does": "Embeds generated hypothetical answers and fuses them with original query vector.",
        "example": '{"corpus_vectors": [[1,0],[0,1]], "query_vector": [1,0], "hypothetical_vectors": [[0.9,0.1]], "k": 1}',
        "category": "vector",
    },
    "vec-query-routing": {
        "id": "ALGO-VEC-SRCH-98",
        "algo_name": "Query Intent Routing",
        "plain_name": "Classify query and direct it to the best collection index",
        "what_it_does": "Routes incoming questions to appropriate indices (code, documentation, tickets).",
        "example": '{"query": "def parse_tokens", "available_routes": {"code": ["def", "class"], "docs": ["guide", "readme"]}}',
        "category": "vector",
    },
    "vec-scatter-gather": {
        "id": "ALGO-VEC-SRCH-99",
        "algo_name": "Sharded Scatter-Gather Search",
        "plain_name": "Query shards in parallel and merge results into global top-k",
        "what_it_does": "Broadcasts search query to shards and merges top-k candidates with min-heap.",
        "example": '{"shard_results": {"s1": [{"id": "1", "score": 0.9}], "s2": [{"id": "2", "score": 0.95}]}, "top_k": 2}',
        "category": "vector",
    },
    "vec-partition-routing": {
        "id": "ALGO-VEC-SRCH-100",
        "algo_name": "Partition-Aware Centroid Routing",
        "plain_name": "Route query to closest centroid shards saving search work",
        "what_it_does": "Compares query against cluster centroids to prune shards that cannot contain nearest neighbors.",
        "example": '{"centroids": [[1,0],[0,1]], "centroid_to_shard_map": {"0": "shard_1", "1": "shard_2"}, "query_vector": [0.9,0.1], "num_target_shards": 1}',
        "category": "vector",
    },
    "vec-load-balancer": {
        "id": "ALGO-VEC-SRCH-101",
        "algo_name": "Replication Load Balancer",
        "plain_name": "Select replica node by least loaded or lowest latency",
        "what_it_does": "Directs query requests among healthy replicas using round-robin, least-loaded, or latency heuristics.",
        "example": '{"replicas": [{"id": "r1", "healthy": true, "active_queries": 2}, {"id": "r2", "healthy": true, "active_queries": 0}], "strategy": "least_loaded"}',
        "category": "vector",
    },
    "vec-hedged-requests": {
        "id": "ALGO-VEC-SRCH-102",
        "algo_name": "Hedged Requests",
        "plain_name": "Send backup query if primary is slow cutting tail latency",
        "what_it_does": "Dispatches secondary request to backup replica when primary exceeds latency SLA threshold.",
        "example": '{"primary_latency_ms": 120.0, "backup_latency_ms": 25.0, "hedge_delay_threshold_ms": 50.0, "is_read_only": true}',
        "category": "vector",
    },
    "vec-kway-merge": {
        "id": "ALGO-VEC-SRCH-103",
        "algo_name": "K-Way Shard Merge",
        "plain_name": "Merge sorted candidate lists from multiple shards using heap",
        "what_it_does": "Produces single sorted top-k output from M pre-sorted shard streams in O(k log M).",
        "example": '{"shard_sorted_lists": [[{"id": "a", "score": 0.9}], [{"id": "b", "score": 0.8}]], "k": 2}',
        "category": "vector",
    },
    "vec-query-cache": {
        "id": "ALGO-VEC-SRCH-104",
        "algo_name": "Query & Result Cache",
        "plain_name": "Cache repeated queries and invalidate automatically on index updates",
        "what_it_does": "Stores query results keyed by query, tenant, filters, and index version with LRU eviction.",
        "example": '{"cache_store": {}, "query": "auth", "tenant_id": "t1", "filters": {}, "index_version": "v1", "results": [{"id": "1"}]}',
        "category": "vector",
    },
    "vec-semantic-cache": {
        "id": "ALGO-VEC-SRCH-105",
        "algo_name": "Semantic Similarity Cache",
        "plain_name": "Reuse cached responses when incoming query is semantically identical",
        "what_it_does": "Matches incoming query embedding against cached queries above similarity threshold.",
        "example": '{"cached_entries": [{"query_vector": [1,0], "result": "cached_answer"}], "query_vector": [0.99,0.01], "tenant_id": "t1", "similarity_threshold": 0.95}',
        "category": "vector",
    },
    "vec-batching": {
        "id": "ALGO-VEC-SRCH-106",
        "algo_name": "Query Batching",
        "plain_name": "Group concurrent queries into micro-batches for throughput",
        "what_it_does": "Packs concurrent queries into matrix batches to maximize GPU and SIMD utilization.",
        "example": '{"pending_queries": [{"id": "q1", "arrival_time": 0.1}], "max_batch_size": 16}',
        "category": "vector",
    },
    "vec-memory-tiering": {
        "id": "ALGO-VEC-SRCH-107",
        "algo_name": "RAM/SSD Memory Tiering",
        "plain_name": "Place hot index parts in RAM and large parts on SSD",
        "what_it_does": "Partitions vector index components into RAM residency or memory-mapped SSD storage.",
        "example": '{"components": [{"name": "centroids", "size_mb": 64, "priority": 10}, {"name": "vectors", "size_mb": 4096, "priority": 1}], "ram_budget_mb": 512}',
        "category": "vector",
    },
    "vec-disk-scheduler": {
        "id": "ALGO-VEC-SRCH-108",
        "algo_name": "Disk I/O Beam Scheduler",
        "plain_name": "Organize SSD reads in page-aligned batches for graph ANN",
        "what_it_does": "Batches proximity graph node lookups into page-aligned I/O requests to reduce SSD read latency.",
        "example": '{"requested_node_ids": [10, 20, 30], "cached_nodes": [10], "max_batch_size": 8}',
        "category": "vector",
    },
    "vec-admission-control": {
        "id": "ALGO-VEC-SRCH-109",
        "algo_name": "Token Bucket Admission Control",
        "plain_name": "Throttle and shed load to protect against runaway agent loops",
        "what_it_does": "Maintains token bucket and concurrency limit to reject or shed queries during overload.",
        "example": '{"current_tokens": 50.0, "max_tokens": 100.0, "refill_rate_per_sec": 10.0, "last_refill_timestamp": 0.0, "current_concurrency": 5, "max_concurrency": 20, "now": 1.0}',
        "category": "vector",
    },
    "vec-search-autotune": {
        "id": "ALGO-VEC-SRCH-110",
        "algo_name": "Pareto Search Parameter Autotuner",
        "plain_name": "Find optimal search parameters meeting recall target at minimum latency",
        "what_it_does": "Sweeps parameters against ground truth to construct Pareto curve and pick fastest setting.",
        "example": '{"ground_truth_topk": [1, 2, 3], "parameter_evaluations": [{"ef_search": 16, "topk": [1, 2, 4], "latency_ms": 1.2}, {"ef_search": 64, "topk": [1, 2, 3], "latency_ms": 2.5}], "target_recall": 0.95}',
        "category": "vector",
    },
    "vec-trfm-subword-tokenization": {
        "id": "ALGO-VEC-TRFM-01",
        "algo_name": "VectorTransformAlgoSubwordTokenization",
        "plain_name": "Subword BPE/WordPiece Tokenizer",
        "what_it_does": "Splits text into subword vocabulary tokens and maps them to numerical token IDs.",
        "example": '{"text": "unbelievable token", "vocab": {"[UNK]": 0, "un": 1, "believ": 2, "able": 3, "token": 4}}',
        "category": "transform",
    },
    "vec-trfm-bi-encoder-forward": {
        "id": "ALGO-VEC-TRFM-02",
        "algo_name": "VectorTransformAlgoBiEncoderForwardPass",
        "plain_name": "Bi-Encoder Forward Pass Simulator",
        "what_it_does": "Processes token ID sequences through dense transformer embedding projection layers.",
        "example": '{"token_ids": [101, 2054, 102], "token_embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}',
        "category": "transform",
    },
    "vec-trfm-mean-pooling": {
        "id": "ALGO-VEC-TRFM-03",
        "algo_name": "VectorTransformAlgoMeanPooling",
        "plain_name": "Attention-Masked Mean Pooling",
        "what_it_does": "Averages token embeddings across the sequence dimension respecting attention masks.",
        "example": '{"token_embeddings": [[1.0, 2.0], [3.0, 4.0]], "attention_mask": [1, 1]}',
        "category": "transform",
    },
    "vec-trfm-cls-pooling": {
        "id": "ALGO-VEC-TRFM-04",
        "algo_name": "VectorTransformAlgoCLSPooling",
        "plain_name": "Classification ([CLS]) Token Pooling",
        "what_it_does": "Extracts the designated CLS token vector as the single sentence representation.",
        "example": '{"token_embeddings": [[0.5, 0.9], [1.0, 2.0]], "cls_index": 0}',
        "category": "transform",
    },
    "vec-trfm-last-token-pooling": {
        "id": "ALGO-VEC-TRFM-05",
        "algo_name": "VectorTransformAlgoLastTokenPooling",
        "plain_name": "Autoregressive Last-Token Pooling",
        "what_it_does": "Extracts the final unpadded token embedding as sentence representation for causal LLMs.",
        "example": '{"token_embeddings": [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]], "sequence_lengths": [2]}',
        "category": "transform",
    },
    "vec-trfm-instruction-prefixes": {
        "id": "ALGO-VEC-TRFM-06",
        "algo_name": "VectorTransformAlgoInstructionPrefixes",
        "plain_name": "Task-Specific Instruction Prefix Prepender",
        "what_it_does": "Prepends asymmetric query/passage instruction prefixes to optimize embedding alignment.",
        "example": '{"text": "what is rust memory safety?", "task_type": "query"}',
        "category": "transform",
    },
    "vec-trfm-contrastive-infonce": {
        "id": "ALGO-VEC-TRFM-07",
        "algo_name": "VectorTransformAlgoContrastiveInfoNCE",
        "plain_name": "Contrastive InfoNCE Loss Calculator",
        "what_it_does": "Computes normalized temperature-scaled cross-entropy loss over positive and negative pairs.",
        "example": '{"query_vector": [1.0, 0.0], "positive_vector": [0.99, 0.01], "negative_vectors": [[0.0, 1.0]], "temperature": 0.05}',
        "category": "transform",
    },
    "vec-trfm-hard-negative-mining": {
        "id": "ALGO-VEC-TRFM-08",
        "algo_name": "VectorTransformAlgoHardNegativeMining",
        "plain_name": "Cross-Batch Hard Negative Miner",
        "what_it_does": "Finds false-positive or highly confusing negatives that are close to query in vector space.",
        "example": '{"query_vector": [1.0, 0.0], "candidate_vectors": [[0.85, 0.1], [0.1, 0.9]], "top_k": 1, "threshold": 0.8}',
        "category": "transform",
    },
    "vec-trfm-matryoshka-learning": {
        "id": "ALGO-VEC-TRFM-09",
        "algo_name": "VectorTransformAlgoMatryoshkaLearning",
        "plain_name": "Matryoshka Representation Learning Slicer",
        "what_it_does": "Slices high-dimensional vectors to nested prefixes (e.g. 64, 128, 256) and renormalizes.",
        "example": '{"full_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "target_dimensions": [2, 4], "normalize": true}',
        "category": "transform",
    },
    "vec-trfm-late-chunking": {
        "id": "ALGO-VEC-TRFM-10",
        "algo_name": "VectorTransformAlgoLateChunking",
        "plain_name": "Late Chunking over Contextualized Embeddings",
        "what_it_does": "Applies chunk boundary pooling after full-document contextualization to retain cross-chunk context.",
        "example": '{"token_embeddings": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "chunk_spans": [[0, 2], [2, 3]]}',
        "category": "transform",
    },
    "vec-trfm-sliding-window": {
        "id": "ALGO-VEC-TRFM-11",
        "algo_name": "VectorTransformAlgoSlidingWindow",
        "plain_name": "Sliding Window Passage Chunker with Overlap",
        "what_it_does": "Slices token sequences into fixed-size windows with configurable step size and boundary overlap.",
        "example": '{"tokens": ["a", "b", "c", "d", "e"], "window_size": 3, "step_size": 2}',
        "category": "transform",
    },
    "vec-trfm-semantic-chunking": {
        "id": "ALGO-VEC-TRFM-12",
        "algo_name": "VectorTransformAlgoSemanticChunking",
        "plain_name": "Embedding Dissimilarity Semantic Chunker",
        "what_it_does": "Finds semantic boundaries where consecutive sentence embedding cosine similarity drops below threshold.",
        "example": '{"sentences": ["hello world", "hi there", "completely different topic"], "sentence_embeddings": [[1,0],[0.99,0.01],[0,1]], "similarity_threshold": 0.5}',
        "category": "transform",
    },
    "vec-trfm-recursive-chunking": {
        "id": "ALGO-VEC-TRFM-13",
        "algo_name": "VectorTransformAlgoRecursiveChunking",
        "plain_name": "Hierarchical Recursive Text Chunker",
        "what_it_does": "Recursively splits document by paragraph, sentence, and word boundaries to fit max token budget.",
        "example": '{"text": "Paragraph 1.\\n\\nParagraph 2 is very long and has details.", "max_chunk_size": 30}',
        "category": "transform",
    },
    "vec-trfm-dynamic-padding-batching": {
        "id": "ALGO-VEC-TRFM-14",
        "algo_name": "VectorTransformAlgoDynamicPaddingBatching",
        "plain_name": "Dynamic Length Padding & Batching Engine",
        "what_it_does": "Pads variable-length sequences to batch maximum length and generates binary attention masks.",
        "example": '{"sequences": [[1, 2, 3], [4, 5]], "pad_token_id": 0}',
        "category": "transform",
    },
    "vec-trfm-l2-norm": {
        "id": "ALGO-VEC-TRFM-15",
        "algo_name": "VectorTransformAlgoL2Norm",
        "plain_name": "Euclidean (L2) Vector Normalization",
        "what_it_does": "Projects float vectors onto unit sphere for fast inner product equivalence with cosine similarity.",
        "example": '{"vector": [3.0, 4.0], "epsilon": 1e-12}',
        "category": "transform",
    },
    "vec-trfm-mean-centering": {
        "id": "ALGO-VEC-TRFM-16",
        "algo_name": "VectorTransformAlgoMeanCentering",
        "plain_name": "Corpus Mean Centering",
        "what_it_does": "Subtracts the mean vector across the corpus to center embeddings at the coordinate origin.",
        "example": '{"vectors": [[2.0, 4.0], [4.0, 6.0]]}',
        "category": "transform",
    },
    "vec-trfm-whitening": {
        "id": "ALGO-VEC-TRFM-17",
        "algo_name": "VectorTransformAlgoWhitening",
        "plain_name": "ZCA / PCA Vector Whitening",
        "what_it_does": "Centers vectors and decorrelates features so covariance becomes an identity matrix.",
        "example": '{"vectors": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "epsilon": 1e-5}',
        "category": "transform",
    },
    "vec-trfm-remove-dominant-directions": {
        "id": "ALGO-VEC-TRFM-18",
        "algo_name": "VectorTransformAlgoRemoveDominantDirections",
        "plain_name": "Dominant Eigenvector Direction Removal",
        "what_it_does": "Removes principal dominant directions that cause word frequency anisotropy in embeddings.",
        "example": '{"vectors": [[10.0, 1.0], [10.1, 2.0], [9.9, 0.5]], "top_components": 1}',
        "category": "transform",
    },
    "vec-trfm-mips-to-nns": {
        "id": "ALGO-VEC-TRFM-19",
        "algo_name": "VectorTransformAlgoMIPSToNNS",
        "plain_name": "MIPS to Euclidean NNS Transformation",
        "what_it_does": "Appends an extra coordinate to vectors converting Maximum Inner Product Search to Euclidean distance.",
        "example": '{"query_vector": [1.0, 2.0], "base_vectors": [[3.0, 4.0], [1.0, 1.0]]}',
        "category": "transform",
    },
    "vec-trfm-score-calibration": {
        "id": "ALGO-VEC-TRFM-20",
        "algo_name": "VectorTransformAlgoScoreCalibration",
        "plain_name": "Similarity Score Calibrator (Platt / Temp / Min-Max)",
        "what_it_does": "Calibrates raw retrieval scores into well-behaved posterior probabilities.",
        "example": '{"raw_scores": [0.8, 0.4, 0.1], "method": "temperature", "temperature": 0.5}',
        "category": "transform",
    },
    "vec-trfm-csls-hubness-reduction": {
        "id": "ALGO-VEC-TRFM-21",
        "algo_name": "VectorTransformAlgoCSLSHubnessReduction",
        "plain_name": "Cross-Domain Similarity Local Scaling (CSLS)",
        "what_it_does": "Mitigates the hubness problem in high dimensions by penalizing nearest neighbors of many vectors.",
        "example": '{"query_vector": [1.0, 0.0], "target_vectors": [[0.9, 0.1], [0.8, 0.2]], "k_neighbors": 1}',
        "category": "transform",
    },
    "vec-trfm-procrustes-alignment": {
        "id": "ALGO-VEC-TRFM-22",
        "algo_name": "VectorTransformAlgoProcrustesAlignment",
        "plain_name": "Orthogonal Procrustes Vector Space Alignment",
        "what_it_does": "Finds the optimal orthogonal rotation matrix aligning source embedding space to target space.",
        "example": '{"source_vectors": [[1.0, 0.0], [0.0, 1.0]], "target_vectors": [[0.0, 1.0], [-1.0, 0.0]]}',
        "category": "transform",
    },
    "vec-trfm-pca": {
        "id": "ALGO-VEC-TRFM-23",
        "algo_name": "VectorTransformAlgoPCA",
        "plain_name": "Principal Component Analysis (PCA)",
        "what_it_does": "Projects vectors onto orthogonal directions of maximum variance for dimension reduction.",
        "example": '{"vectors": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], "target_dimension": 2}',
        "category": "transform",
    },
    "vec-trfm-truncated-svd": {
        "id": "ALGO-VEC-TRFM-24",
        "algo_name": "VectorTransformAlgoTruncatedSVD",
        "plain_name": "Truncated Singular Value Decomposition (SVD / LSA)",
        "what_it_does": "Computes top-k singular vectors without centering data, preserving sparsity in large matrices.",
        "example": '{"matrix": [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], "n_components": 2}',
        "category": "transform",
    },
    "vec-trfm-random-projection": {
        "id": "ALGO-VEC-TRFM-25",
        "algo_name": "VectorTransformAlgoRandomProjection",
        "plain_name": "Johnson-Lindenstrauss Random Projection",
        "what_it_does": "Projects vectors to lower dimensions with random Gaussian or Achlioptas matrices preserving distances.",
        "example": '{"vectors": [[1.0, 2.0, 3.0, 4.0]], "target_dimension": 2, "method": "gaussian"}',
        "category": "transform",
    },
    "vec-trfm-autoencoder-compression": {
        "id": "ALGO-VEC-TRFM-26",
        "algo_name": "VectorTransformAlgoAutoencoderCompression",
        "plain_name": "Non-Linear Autoencoder Vector Compressor",
        "what_it_does": "Trained neural bottleneck compression for complex non-linear embedding manifolds.",
        "example": '{"vectors": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "bottleneck_dim": 2, "epochs": 5}',
        "category": "transform",
    },
    "vec-trfm-umap": {
        "id": "ALGO-VEC-TRFM-27",
        "algo_name": "VectorTransformAlgoUMAP",
        "plain_name": "Uniform Manifold Approximation & Projection (UMAP)",
        "what_it_does": "Preserves both local and global topological structure during non-linear manifold projection.",
        "example": '{"vectors": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], "n_components": 2, "n_neighbors": 2}',
        "category": "transform",
    },
    "vec-trfm-tsne": {
        "id": "ALGO-VEC-TRFM-28",
        "algo_name": "VectorTransformAlgoTSNE",
        "plain_name": "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
        "what_it_does": "Maps high-dimensional points into 2D/3D space using Student-t distribution to visualize clusters.",
        "example": '{"vectors": [[1.0, 2.0], [1.1, 2.1], [5.0, 5.0]], "n_components": 2, "perplexity": 2.0, "iterations": 50}',
        "category": "transform",
    },
    "vec-trfm-projection-head": {
        "id": "ALGO-VEC-TRFM-29",
        "algo_name": "VectorTransformAlgoProjectionHead",
        "plain_name": "Dense MLP Non-Linear Projection Head",
        "what_it_does": "Transforms representations via linear projection and activation for fine-tuning tasks.",
        "example": '{"vectors": [[1.0, 2.0, 3.0]], "output_dim": 2, "activation": "relu"}',
        "category": "transform",
    },
    "vec-trfm-incremental-pca": {
        "id": "ALGO-VEC-TRFM-30",
        "algo_name": "VectorTransformAlgoIncrementalPCA",
        "plain_name": "Streaming Incremental PCA",
        "what_it_does": "Updates principal components batch by batch without holding entire dataset in memory.",
        "example": '{"vector_batch": [[1.0, 2.0], [3.0, 4.0]], "target_dimension": 1}',
        "category": "transform",
    },
    "vec-trfm-simhash": {
        "id": "ALGO-VEC-TRFM-31",
        "algo_name": "VectorTransformAlgoSimHash",
        "plain_name": "SimHash Locality-Sensitive Fingerprinting",
        "what_it_does": "Hashes vectors into 64-bit compact bitstrings where Hamming distance tracks angular distance.",
        "example": '{"vector": [0.5, -0.2, 0.8], "num_bits": 16}',
        "category": "transform",
    },
    "vec-trfm-learned-sparse-expansion": {
        "id": "ALGO-VEC-TRFM-32",
        "algo_name": "VectorTransformAlgoLearnedSparseExpansion",
        "plain_name": "Learned Sparse Expansion (SPLADE-style)",
        "what_it_does": "Projects dense embeddings into high-dimensional sparse representations with top-k term activations.",
        "example": '{"dense_vector": [0.1, 0.8, -0.3], "dictionary_dim": 16, "sparsity_k": 3}',
        "category": "transform",
    },
    "vec-trfm-scalar-quantization": {
        "id": "ALGO-VEC-TRFM-33",
        "algo_name": "VectorTransformAlgoScalarQuantization",
        "plain_name": "Uniform Scalar Quantization (SQ8 / SQ4)",
        "what_it_does": "Maps continuous 32-bit floats into discrete 8-bit or 4-bit integers with affine scale and offset.",
        "example": '{"vector": [-1.0, 0.0, 0.5, 1.0], "num_bits": 8}',
        "category": "transform",
    },
    "vec-trfm-binary-quantization": {
        "id": "ALGO-VEC-TRFM-34",
        "algo_name": "VectorTransformAlgoBinaryQuantization",
        "plain_name": "1-Bit Binary Quantization (BQ)",
        "what_it_does": "Binarizes vector dimensions to 0 or 1 based on threshold for 32x memory compression.",
        "example": '{"vector": [0.5, -0.2, 0.8, -0.9], "threshold": 0.0}',
        "category": "transform",
    },
    "vec-trfm-product-quantization": {
        "id": "ALGO-VEC-TRFM-35",
        "algo_name": "VectorTransformAlgoProductQuantization",
        "plain_name": "Product Quantization (PQ)",
        "what_it_does": "Splits vectors into sub-vectors and quantizes each with an independent codebook of centroids.",
        "example": '{"vectors": [[1,2,3,4], [5,6,7,8], [1,2,4,4]], "num_subvectors": 2, "num_centroids": 2}',
        "category": "transform",
    },
    "vec-trfm-optimized-product-quantization": {
        "id": "ALGO-VEC-TRFM-36",
        "algo_name": "VectorTransformAlgoOptimizedProductQuantization",
        "plain_name": "Optimized Product Quantization (OPQ)",
        "what_it_does": "Learns an orthogonal space rotation minimizing quantization distortion across PQ sub-spaces.",
        "example": '{"vectors": [[1,2,3,4], [4,3,2,1]], "num_subvectors": 2, "num_centroids": 2, "iterations": 3}',
        "category": "transform",
    },
    "vec-trfm-residual-quantization": {
        "id": "ALGO-VEC-TRFM-37",
        "algo_name": "VectorTransformAlgoResidualQuantization",
        "plain_name": "Residual Quantization (RQ)",
        "what_it_does": "Iteratively quantizes residual errors in cascade across multiple hierarchical codebook stages.",
        "example": '{"vectors": [[1.0, 2.0], [3.0, 4.0]], "num_stages": 2, "num_centroids": 2}',
        "category": "transform",
    },
    "vec-trfm-anisotropic-quantization": {
        "id": "ALGO-VEC-TRFM-38",
        "algo_name": "VectorTransformAlgoAnisotropicQuantization",
        "plain_name": "Anisotropic Vector Quantization",
        "what_it_does": "Penalizes quantization error parallel to vector directions more heavily than orthogonal error.",
        "example": '{"vectors": [[1,2,3,4], [5,6,7,8]], "num_subvectors": 2, "num_centroids": 2, "lambda_penalty": 0.2}',
        "category": "transform",
    },
    "vec-trfm-kmeans-clustering": {
        "id": "ALGO-VEC-TRFM-39",
        "algo_name": "VectorTransformAlgoKMeansClustering",
        "plain_name": "Lloyd's K-Means Vector Clustering",
        "what_it_does": "Partitions vectors into k Voronoi clusters iteratively updating centroids to minimize WCSS.",
        "example": '{"vectors": [[0,0],[1,1],[5,5],[6,6]], "k": 2}',
        "category": "transform",
    },
    "vec-trfm-kmeans-plus-plus": {
        "id": "ALGO-VEC-TRFM-40",
        "algo_name": "VectorTransformAlgoKMeansPlusPlus",
        "plain_name": "K-Means++ Probabilistic Seeding",
        "what_it_does": "Initializes centroids with probability proportional to squared distance to nearest centroid.",
        "example": '{"vectors": [[0,0],[1,1],[5,5],[6,6]], "k": 2, "random_seed": 42}',
        "category": "transform",
    },
    "vec-trfm-minibatch-kmeans": {
        "id": "ALGO-VEC-TRFM-41",
        "algo_name": "VectorTransformAlgoMinibatchKMeans",
        "plain_name": "Mini-Batch K-Means Fast Clustering",
        "what_it_does": "Updates centroids using small random mini-batches for 10x faster clustering on large datasets.",
        "example": '{"vectors": [[0,0],[1,1],[10,10],[11,11]], "k": 2, "batch_size": 2}',
        "category": "transform",
    },
    "vec-trfm-hierarchical-kmeans": {
        "id": "ALGO-VEC-TRFM-42",
        "algo_name": "VectorTransformAlgoHierarchicalKMeans",
        "plain_name": "Hierarchical Tree K-Means (HKM)",
        "what_it_does": "Builds a balanced clustering tree recursively for logarithmic multi-stage vector routing.",
        "example": '{"vectors": [[0,0],[1,1],[5,5],[6,6],[10,10]], "branching_factor": 2, "depth": 2}',
        "category": "transform",
    },
    "vec-trfm-adc-lookup": {
        "id": "ALGO-VEC-TRFM-43",
        "algo_name": "VectorTransformAlgoADCLookup",
        "plain_name": "Asymmetric Distance Computation (ADC) Lookup",
        "what_it_does": "Precomputes query-to-centroid distance tables for high-speed table lookup vector distance calculation.",
        "example": '{"query_vector": [1.0, 2.0], "codebook": [[[0.0], [1.0]], [[0.0], [2.0]]], "encoded_vectors": [[1, 1], [0, 0]]}',
        "category": "transform",
    },
    "vec-trfm-fast-scan-pq": {
        "id": "ALGO-VEC-TRFM-44",
        "algo_name": "VectorTransformAlgoFastScanPQ",
        "plain_name": "SIMD Register Fast-Scan PQ",
        "what_it_does": "Interleaves 4-bit PQ codes across vectors to evaluate distances via SIMD shuffle instructions.",
        "example": '{"query_vector": [1.0, 2.0], "codebook": [[[1.0], [0.0]], [[2.0], [0.0]]], "encoded_vectors": [[0, 0], [1, 1]]}',
        "category": "transform",
    },
    "vec-trfm-rabitq": {
        "id": "ALGO-VEC-TRFM-45",
        "algo_name": "VectorTransformAlgoRaBiTQ",
        "plain_name": "Randomized 1-Bit Binary Quantizer (RaBiTQ)",
        "what_it_does": "Quantizes vectors to random hypercube vertices with theoretical distortion error bounds.",
        "example": '{"vector": [1.2, -0.8, 3.4], "target_bits": 1}',
        "category": "transform",
    },
    "vec-trfm-half-precision": {
        "id": "ALGO-VEC-TRFM-46",
        "algo_name": "VectorTransformAlgoHalfPrecision",
        "plain_name": "FP16 / BF16 Half Precision Encoder",
        "what_it_does": "Encodes IEEE 754 float32 numbers into compact float16 or bfloat16 16-bit representations.",
        "example": '{"vector": [1.0, -2.5, 0.003], "target_format": "float16"}',
        "category": "transform",
    },
    "vec-trfm-multi-vector-representation": {
        "id": "ALGO-VEC-TRFM-47",
        "algo_name": "VectorTransformAlgoMultiVectorRepresentation",
        "plain_name": "ColBERT Multi-Vector Representation",
        "what_it_does": "Generates a bag of contextualized token vectors per passage for fine-grained token-level matching.",
        "example": '{"token_embeddings": [[1.0, 0.0], [0.0, 1.0]], "max_tokens": 16, "normalize": true}',
        "category": "transform",
    },
    "vec-trfm-multi-vector-compression": {
        "id": "ALGO-VEC-TRFM-48",
        "algo_name": "VectorTransformAlgoMultiVectorCompression",
        "plain_name": "Multi-Vector Residual Compression (ColBERT-PR/PLAID)",
        "what_it_does": "Compresses multi-vector bags by pruning punctuation and clustering redundant token vectors.",
        "example": '{"multi_vectors": [[1,0],[0.99,0.01],[0,1],[0.01,0.99]], "compression_ratio": 0.5}',
        "category": "transform",
    },
    "vec-trfm-sparse-vector-representation": {
        "id": "ALGO-VEC-TRFM-49",
        "algo_name": "VectorTransformAlgoSparseVectorRepresentation",
        "plain_name": "BM25 / SPLADE Sparse Lexical Vector Encoder",
        "what_it_does": "Encodes raw text or tokens into sparse term index and weight arrays for hybrid retrieval.",
        "example": '{"text_or_tokens": "fast vector search and transformation", "max_terms": 10}',
        "category": "transform",
    },
    "vec-trfm-embedding-cache": {
        "id": "ALGO-VEC-TRFM-50",
        "algo_name": "VectorTransformAlgoEmbeddingCache",
        "plain_name": "Exact Embedding Key-Value Cache",
        "what_it_does": "Caches computed embeddings keyed by SHA-256 text hashes to avoid redundant inference calls.",
        "example": '{"key": "query_sha256_hash", "vector": [0.1, 0.2], "action": "set", "ttl_seconds": 3600}',
        "category": "transform",
    },
    "vec-upd-upsert-stable-id": {
        "id": "ALGO-VEC-UPD-111",
        "algo_name": "VectorUpdateAlgoUpsertStableId",
        "plain_name": "Upsert by Stable ID",
        "what_it_does": "Deterministic upsert key derivation and deduplication engine avoiding duplicate entries.",
        "example": '{"source_id": "doc1", "chunk_id": "c0", "model_version": "1.0.0", "content": "Sample content text"}',
        "category": "update",
    },
    "vec-upd-wal": {
        "id": "ALGO-VEC-UPD-112",
        "algo_name": "VectorUpdateAlgoWal",
        "plain_name": "Write-Ahead Log",
        "what_it_does": "Append-only durable transaction log with group commits, truncation, and crash recovery.",
        "example": '{"operations": [{"op_type": "UPSERT", "record_id": "r1", "vector": [0.1, 0.2]}], "last_sequence_num": 0}',
        "category": "update",
    },
    "vec-upd-fresh-buffer": {
        "id": "ALGO-VEC-UPD-113",
        "algo_name": "VectorUpdateAlgoFreshBuffer",
        "plain_name": "Fresh Buffer",
        "what_it_does": "Low-latency in-memory write segment searchable via brute-force prior to flush.",
        "example": '{"buffer_records": [], "new_records": [{"id": "r1", "vector": [0.1, 0.2]}], "max_buffer_size": 1000}',
        "category": "update",
    },
    "vec-upd-lsm-storage": {
        "id": "ALGO-VEC-UPD-114",
        "algo_name": "VectorUpdateAlgoLsmStorage",
        "plain_name": "LSM Segment Storage",
        "what_it_does": "Multi-segment LSM storage coordinator with deletion bitmaps and fragmentation metrics.",
        "example": '{"segments": [{"segment_id": "s1", "records": [{"id": "r1", "vector": [0.1, 0.2]}], "tombstone_ids": []}], "top_k": 5}',
        "category": "update",
    },
    "vec-upd-segment-compaction": {
        "id": "ALGO-VEC-UPD-115",
        "algo_name": "VectorUpdateAlgoSegmentCompaction",
        "plain_name": "Segment Compaction",
        "what_it_does": "Merges vector segments and permanently purges tombstoned records into compacted tiers.",
        "example": '{"segments_to_merge": [{"segment_id": "s1", "records": [{"id": "r1"}], "tombstone_ids": ["r1"]}], "target_tier": "L1"}',
        "category": "update",
    },
    "vec-upd-tombstone-deletion": {
        "id": "ALGO-VEC-UPD-116",
        "algo_name": "VectorUpdateAlgoTombstoneDeletion",
        "plain_name": "Tombstone Deletion",
        "what_it_does": "Bitmask deletion marking mechanism filtering out deleted vectors instantly.",
        "example": '{"active_tombstones": ["r1"], "delete_ids": ["r2"], "total_index_size": 100}',
        "category": "update",
    },
    "vec-upd-hnsw-deletion-repair": {
        "id": "ALGO-VEC-UPD-117",
        "algo_name": "VectorUpdateAlgoHnswDeletionRepair",
        "plain_name": "HNSW Deletion Repair",
        "what_it_does": "Re-wires and repairs HNSW graph adjacency lists after soft node deletions.",
        "example": '{"adjacency_list": {"n1": ["n2", "n3"], "n2": ["n1"], "n3": ["n1"]}, "deleted_nodes": ["n1"]}',
        "category": "update",
    },
    "vec-upd-fresh-diskann-update": {
        "id": "ALGO-VEC-UPD-118",
        "algo_name": "VectorUpdateAlgoFreshDiskannUpdate",
        "plain_name": "FreshDiskANN Update",
        "what_it_does": "Maintains dual disk-memory graph structure with periodic streaming merges.",
        "example": '{"disk_graph_nodes": ["d1", "d2"], "mem_graph_nodes": ["m1"], "deleted_nodes": [], "new_records": ["m2"], "mem_threshold": 500}',
        "category": "update",
    },
    "vec-upd-incremental-ivf": {
        "id": "ALGO-VEC-UPD-119",
        "algo_name": "VectorUpdateAlgoIncrementalIvf",
        "plain_name": "Incremental IVF",
        "what_it_does": "Assigns new vectors to nearest existing IVF centroids without retraining codebooks.",
        "example": '{"centroids": [[0.0, 0.0], [1.0, 1.0]], "vectors_to_insert": [{"id": "v1", "vector": [0.1, 0.2]}], "inverted_lists": {}}',
        "category": "update",
    },
    "vec-upd-centroid-drift": {
        "id": "ALGO-VEC-UPD-120",
        "algo_name": "VectorUpdateAlgoCentroidDrift",
        "plain_name": "Centroid Drift Detection",
        "what_it_does": "Monitors quantization error drift and cluster imbalance over time.",
        "example": '{"baseline_quantization_error": 0.05, "current_quantization_error": 0.08, "inverted_list_lengths": [10, 15, 20]}',
        "category": "update",
    },
    "vec-upd-reembedding-pipeline": {
        "id": "ALGO-VEC-UPD-121",
        "algo_name": "VectorUpdateAlgoReembeddingPipeline",
        "plain_name": "Re-Embedding Pipeline",
        "what_it_does": "Coordinates partition-based backfill and golden set recall validation for model upgrades.",
        "example": '{"total_chunks": 1000, "completed_chunks": 250, "elapsed_seconds": 50.0, "target_model_version": "v2.0"}',
        "category": "update",
    },
    "vec-upd-dual-write": {
        "id": "ALGO-VEC-UPD-122",
        "algo_name": "VectorUpdateAlgoDualWrite",
        "plain_name": "Dual-Write Shadow Index",
        "what_it_does": "Replicates live mutation events simultaneously to primary and shadow indexes.",
        "example": '{"mutation_events": [{"record_id": "r1", "op_type": "UPSERT"}], "primary_ids": [], "shadow_ids": []}',
        "category": "update",
    },
    "vec-upd-blue-green-swap": {
        "id": "ALGO-VEC-UPD-123",
        "algo_name": "VectorUpdateAlgoBlueGreenSwap",
        "plain_name": "Blue-Green Index Swap",
        "what_it_does": "Executes zero-downtime atomic alias pointer swaps between Blue and Green indexes.",
        "example": '{"current_alias_target": "blue", "candidate_target": "green", "is_candidate_warmed": true}',
        "category": "update",
    },
    "vec-upd-idempotent-ingestion": {
        "id": "ALGO-VEC-UPD-124",
        "algo_name": "VectorUpdateAlgoIdempotentIngestion",
        "plain_name": "Idempotent Ingestion",
        "what_it_does": "Skips redundant embeddings and index writes when source content hashes match.",
        "example": '{"incoming_chunks": [{"chunk_id": "c1", "content": "text"}], "stored_hash_map": {}}',
        "category": "update",
    },
    "vec-upd-cdc": {
        "id": "ALGO-VEC-UPD-125",
        "algo_name": "VectorUpdateAlgoCdc",
        "plain_name": "Change Data Capture",
        "what_it_does": "Streams database replication log records into partitioned vector mutation queues.",
        "example": '{"cdc_raw_events": [{"key": "k1", "op": "INSERT", "log_offset": 100}], "partition_count": 4}',
        "category": "update",
    },
    "vec-upd-transactional-outbox": {
        "id": "ALGO-VEC-UPD-126",
        "algo_name": "VectorUpdateAlgoTransactionalOutbox",
        "plain_name": "Transactional Outbox",
        "what_it_does": "Stages outbox events atomically with business writes for reliable streaming.",
        "example": '{"pending_outbox_rows": [{"event_id": "e1", "retry_count": 0}], "published_event_ids": []}',
        "category": "update",
    },
    "vec-upd-merkle-tree-sync": {
        "id": "ALGO-VEC-UPD-127",
        "algo_name": "VectorUpdateAlgoMerkleTreeSync",
        "plain_name": "Merkle-Tree Sync",
        "what_it_does": "Anti-entropy hash tree comparison identifying out-of-sync key intervals.",
        "example": '{"source_records": [{"id": "r1", "hash": "h1"}], "target_records": [{"id": "r1", "hash": "h1"}]}',
        "category": "update",
    },
    "vec-upd-watermarks-freshness": {
        "id": "ALGO-VEC-UPD-128",
        "algo_name": "VectorUpdateAlgoWatermarksFreshness",
        "plain_name": "Watermarks & Freshness",
        "what_it_does": "Monitors event-time watermarks and end-to-end data freshness latency.",
        "example": '{"stage_watermarks": {"cdc": 100.0, "embed": 98.0, "index": 95.0}, "current_time": 102.0}',
        "category": "update",
    },
    "vec-upd-idempotency-keys": {
        "id": "ALGO-VEC-UPD-129",
        "algo_name": "VectorUpdateAlgoIdempotencyKeys",
        "plain_name": "Idempotency Keys",
        "what_it_does": "Enforces exactly-once vector mutations using monotonic versions and dedup tokens.",
        "example": '{"record_id": "r1", "incoming_version": 2, "idempotency_token": "tok1", "stored_version": 1}',
        "category": "update",
    },
    "vec-upd-backfill-checkpoints": {
        "id": "ALGO-VEC-UPD-130",
        "algo_name": "VectorUpdateAlgoBackfillCheckpoints",
        "plain_name": "Backfill Checkpoints",
        "what_it_does": "Manages partition cursor progression and resumable checkpoints during backfills.",
        "example": '{"job_id": "j1", "last_cursor": "cur1", "processed_count": 50, "total_count": 100}',
        "category": "update",
    },
    "vec-upd-micro-batching": {
        "id": "ALGO-VEC-UPD-131",
        "algo_name": "VectorUpdateAlgoMicroBatching",
        "plain_name": "Streaming Micro-Batching",
        "what_it_does": "Buffers incoming vectors into dynamic micro-batches by count and latency boundaries.",
        "example": '{"incoming_items": [{"id": "1"}, {"id": "2"}], "max_batch_size": 2}',
        "category": "update",
    },
    "vec-upd-backpressure-priority": {
        "id": "ALGO-VEC-UPD-132",
        "algo_name": "VectorUpdateAlgoBackpressurePriority",
        "plain_name": "Backpressure & Prioritization",
        "what_it_does": "Prioritizes mutation queues (DELETES > UPDATES > BACKFILL) with load shedding.",
        "example": '{"queue_items": [{"type": "DELETE", "id": "d1"}, {"type": "BACKFILL", "id": "b1"}], "drain_limit": 10}',
        "category": "update",
    },
    "vec-upd-mvcc-snapshots": {
        "id": "ALGO-VEC-UPD-133",
        "algo_name": "VectorUpdateAlgoMvccSnapshots",
        "plain_name": "MVCC Snapshots",
        "what_it_does": "Pins multi-version snapshots for long-running queries while isolating writes.",
        "example": '{"active_versions": [{"version_id": "v1", "commit_seq": 10}], "pinned_version_ids": ["v1"]}',
        "category": "update",
    },
    "vec-upd-consistency-levels": {
        "id": "ALGO-VEC-UPD-134",
        "algo_name": "VectorUpdateAlgoConsistencyLevels",
        "plain_name": "Consistency Levels",
        "what_it_does": "Evaluates read consistency modes (EVENTUAL, READ_YOUR_WRITES, STRONG).",
        "example": '{"consistency_level": "READ_YOUR_WRITES", "replica_sequence_num": 100, "client_write_token_seq": 100}',
        "category": "update",
    },
    "vec-upd-leader-follower": {
        "id": "ALGO-VEC-UPD-135",
        "algo_name": "VectorUpdateAlgoLeaderFollower",
        "plain_name": "Leader-Follower Replication",
        "what_it_does": "Streams WAL to followers and routes queries to caught-up read replicas.",
        "example": '{"leader_sequence_num": 100, "followers": [{"replica_id": "f1", "sequence_num": 100}]}',
        "category": "update",
    },
    "vec-upd-raft-consensus": {
        "id": "ALGO-VEC-UPD-136",
        "algo_name": "VectorUpdateAlgoRaftConsensus",
        "plain_name": "Raft Consensus",
        "what_it_does": "Implements Raft consensus protocol logic for distributed vector log commits.",
        "example": '{"current_term": 1, "cluster_size": 3, "vote_responses": [{"vote_granted": true, "term": 1}, {"vote_granted": true, "term": 1}]}',
        "category": "update",
    },
    "vec-upd-quorum-reads-writes": {
        "id": "ALGO-VEC-UPD-137",
        "algo_name": "VectorUpdateAlgoQuorumReadsWrites",
        "plain_name": "Quorum Reads & Writes",
        "what_it_does": "Evaluates strict quorum invariants (R + W > N) and performs read-repair.",
        "example": '{"total_replicas_n": 3, "write_ack_count_w": 2, "read_responses": [{"version": 2, "replica_id": "r1"}, {"version": 1, "replica_id": "r2"}]}',
        "category": "update",
    },
    "vec-upd-snapshot-replay-recovery": {
        "id": "ALGO-VEC-UPD-138",
        "algo_name": "VectorUpdateAlgoSnapshotReplayRecovery",
        "plain_name": "Snapshot + Log Replay",
        "what_it_does": "Rebuilds state from base checkpoints and sequential WAL replays.",
        "example": '{"snapshot_records": {"r1": {"vector": [0.1]}}, "snapshot_seq": 1, "wal_log": [{"sequence_number": 2, "op_type": "UPSERT", "record_id": "r2", "vector": [0.2]}]}',
        "category": "update",
    },
    "vec-upd-consistent-hashing": {
        "id": "ALGO-VEC-UPD-139",
        "algo_name": "VectorUpdateAlgoConsistentHashing",
        "plain_name": "Consistent Hashing",
        "what_it_does": "Distributes partition keys across hash rings with virtual node balancing.",
        "example": '{"active_nodes": ["node1", "node2"], "keys_to_assign": ["doc1", "doc2", "doc3"]}',
        "category": "update",
    },
    "vec-upd-version-vectors": {
        "id": "ALGO-VEC-UPD-140",
        "algo_name": "VectorUpdateAlgoVersionVectors",
        "plain_name": "Version Vectors",
        "what_it_does": "Tracks per-replica causality vectors to detect concurrent conflicting mutations.",
        "example": '{"vector_a": {"node1": 2, "node2": 1}, "vector_b": {"node1": 2, "node2": 2}}',
        "category": "update",
    },
    "vec-upd-schema-versioning": {
        "id": "ALGO-VEC-UPD-141",
        "algo_name": "VectorUpdateAlgoSchemaVersioning",
        "plain_name": "Schema Versioning",
        "what_it_does": "Enforces metadata field migrations and expand-and-contract schema evolution.",
        "example": '{"metadata": {"source_id": "s1", "model_version": "v1", "old_title": "hello"}, "current_schema_version": 1, "target_schema_version": 2, "field_migration_map": {"old_title": "title"}}',
        "category": "update",
    },
    "vec-upd-multi-tenant-isolation": {
        "id": "ALGO-VEC-UPD-142",
        "algo_name": "VectorUpdateAlgoMultiTenantIsolation",
        "plain_name": "Multi-Tenant Isolation",
        "what_it_does": "Enforces strict tenant boundaries and validates per-tenant capacity quotas.",
        "example": '{"authenticated_tenant_id": "t1", "records": [{"tenant_id": "t1", "id": "1"}, {"tenant_id": "t2", "id": "2"}]}',
        "category": "update",
    },
    "vec-upd-ttl-expiry": {
        "id": "ALGO-VEC-UPD-143",
        "algo_name": "VectorUpdateAlgoTtlExpiry",
        "plain_name": "TTL Expiry",
        "what_it_does": "Filters out expired records from queries and collects IDs for tombstone purging.",
        "example": '{"records": [{"id": "r1", "ttl_expiry_timestamp": 100.0}], "current_time": 150.0}',
        "category": "update",
    },
    "vec-upd-orphan-gc": {
        "id": "ALGO-VEC-UPD-144",
        "algo_name": "VectorUpdateAlgoOrphanGc",
        "plain_name": "Orphan Garbage Collection",
        "what_it_does": "Identifies and purges index vectors whose source documents have been deleted.",
        "example": '{"index_vectors": [{"id": "v1", "source_id": "s1"}, {"id": "v2", "source_id": "s2"}], "authoritative_source_ids": ["s1"]}',
        "category": "update",
    },
    "vec-upd-near-duplicate-dedupe": {
        "id": "ALGO-VEC-UPD-145",
        "algo_name": "VectorUpdateAlgoNearDuplicateDedupe",
        "plain_name": "Near-Duplicate Deduplication",
        "what_it_does": "Clusters vectors with high semantic cosine similarity into canonical records.",
        "example": '{"candidates": [{"id": "1", "vector": [1.0, 0.0]}, {"id": "2", "vector": [0.99, 0.01]}], "similarity_threshold": 0.95}',
        "category": "update",
    },
    "vec-upd-rebuild-scheduling": {
        "id": "ALGO-VEC-UPD-146",
        "algo_name": "VectorUpdateAlgoRebuildScheduling",
        "plain_name": "Rebuild Scheduling",
        "what_it_does": "Evaluates index decay telemetry to schedule off-peak full rebuilds.",
        "example": '{"tombstone_ratio": 0.25, "measured_recall": 0.82, "target_recall": 0.90}',
        "category": "update",
    },
    "vec-upd-online-index-build": {
        "id": "ALGO-VEC-UPD-147",
        "algo_name": "VectorUpdateAlgoOnlineIndexBuild",
        "plain_name": "Online Index Build",
        "what_it_does": "Constructs new index concurrently and replays mutation logs until lag is minimal.",
        "example": '{"snapshot_records": [{"id": "r1", "vector": [0.1]}], "mutation_wal_entries": [{"sequence_number": 2, "op_type": "UPSERT", "record_id": "r2", "vector": [0.2]}]}',
        "category": "update",
    },
    "vec-upd-bulk-loading": {
        "id": "ALGO-VEC-UPD-148",
        "algo_name": "VectorUpdateAlgoBulkLoading",
        "plain_name": "Bulk Loading",
        "what_it_does": "Partitions and builds sorted bottom-up index segments in parallel.",
        "example": '{"vectors": [{"id": "1", "vector": [0.1, 0.2]}, {"id": "2", "vector": [0.3, 0.4]}], "target_segment_size": 1}',
        "category": "update",
    },
    "vec-upd-requantization-migration": {
        "id": "ALGO-VEC-UPD-149",
        "algo_name": "VectorUpdateAlgoRequantizationMigration",
        "plain_name": "Requantization Migration",
        "what_it_does": "Converts full-precision vectors to updated quantization formats.",
        "example": '{"full_precision_vectors": [[0.1, -0.5, 0.8]], "target_format": "INT8"}',
        "category": "update",
    },
    "vec-upd-deletion-verification": {
        "id": "ALGO-VEC-UPD-150",
        "algo_name": "VectorUpdateAlgoDeletionVerification",
        "plain_name": "Deletion Verification",
        "what_it_does": "Cryptographically audits erasure of vectors to satisfy GDPR Article 17.",
        "example": '{"source_id": "doc123", "index_contains_id": false, "cache_contains_id": false, "shadow_index_contains_id": false}',
        "category": "update",
    },
    "vec-upd-backward-compatible-training": {
        "id": "ALGO-VEC-UPD-151",
        "algo_name": "VectorUpdateAlgoBackwardCompatibleTraining",
        "plain_name": "Backward-Compatible Training",
        "what_it_does": "Evaluates cross-model retrieval compatibility between new and legacy embeddings.",
        "example": '{"new_query_vectors": [[1.0, 0.0]], "legacy_doc_vectors": [[0.9, 0.1]], "ground_truth_relevance_pairs": [[0, 0]]}',
        "category": "update",
    },
    "vec-upd-lazy-reembedding": {
        "id": "ALGO-VEC-UPD-152",
        "algo_name": "VectorUpdateAlgoLazyReembedding",
        "plain_name": "Lazy Re-Embedding",
        "what_it_does": "Migrates vectors to new model versions gradually upon read or update access.",
        "example": '{"accessed_records": [{"id": "r1", "model_version": "v1.0"}], "target_model_version": "v2.0"}',
        "category": "update",
    },
    "vec-upd-codebook-retraining": {
        "id": "ALGO-VEC-UPD-153",
        "algo_name": "VectorUpdateAlgoCodebookRetraining",
        "plain_name": "Codebook Retraining",
        "what_it_does": "Re-trains Product Quantization codebooks when data distribution shifts.",
        "example": '{"sample_vectors": [[0.1, 0.2, 0.3, 0.4]], "num_subvectors_m": 2, "centroids_per_subvector_k": 2}',
        "category": "update",
    },
    "vec-upd-metadata-index-maintenance": {
        "id": "ALGO-VEC-UPD-154",
        "algo_name": "VectorUpdateAlgoMetadataIndexMaintenance",
        "plain_name": "Metadata Index Maintenance",
        "what_it_does": "Synchronizes secondary categorical and range indexes with vector mutations.",
        "example": '{"current_inverted_index": {}, "mutation_type": "UPSERT", "record_id": "r1", "metadata": {"tenant": "t1"}}',
        "category": "update",
    },
    "vec-upd-atomic-commit": {
        "id": "ALGO-VEC-UPD-155",
        "algo_name": "VectorUpdateAlgoAtomicCommit",
        "plain_name": "Atomic Vector & Metadata Commit",
        "what_it_does": "Two-phase atomic commit ensuring vectors and security ACLs become visible simultaneously.",
        "example": '{"vector": [0.1, 0.2], "metadata": {"tenant_id": "t1", "acl": ["read"]}}',
        "category": "update",
    },
    "vec-obs-recall-at-k": {
        "id": "ALGO-VEC-OBS-156",
        "algo_name": "VectorObservabilityAlgoRecallAtK",
        "plain_name": "Recall@K Metric",
        "what_it_does": "Measures retrieval recall against exact brute-force ground truth IDs.",
        "example": '{"retrieved_ids": ["a", "b", "c"], "ground_truth_ids": ["a", "b", "d"], "k": 3}',
        "category": "observability",
    },
    "vec-obs-ground-truth-sampling": {
        "id": "ALGO-VEC-OBS-157",
        "algo_name": "VectorObservabilityAlgoGroundTruthSampling",
        "plain_name": "Ground Truth Sampling",
        "what_it_does": "Samples queries and performs exact brute-force search to generate ground-truth baselines.",
        "example": '{"sample_queries": [[0.1, 0.2]], "corpus_vectors": [{"id": "d1", "vector": [0.1, 0.2]}], "ground_truth_k": 1}',
        "category": "observability",
    },
    "vec-obs-precision-at-k": {
        "id": "ALGO-VEC-OBS-158",
        "algo_name": "VectorObservabilityAlgoPrecisionAtK",
        "plain_name": "Precision@K Metric",
        "what_it_does": "Measures the fraction of top-K retrieved items that are relevant.",
        "example": '{"retrieved_ids": ["a", "b", "c"], "relevant_ids": ["a", "c"], "k": 3}',
        "category": "observability",
    },
    "vec-obs-mrr": {
        "id": "ALGO-VEC-OBS-159",
        "algo_name": "VectorObservabilityAlgoMrr",
        "plain_name": "Mean Reciprocal Rank (MRR)",
        "what_it_does": "Calculates reciprocal rank of the first relevant result in retrieved ranking.",
        "example": '{"retrieved_ids": ["x", "a", "b"], "relevant_ids": ["a"]}',
        "category": "observability",
    },
    "vec-obs-ndcg": {
        "id": "ALGO-VEC-OBS-160",
        "algo_name": "VectorObservabilityAlgoNdcg",
        "plain_name": "Normalized Discounted Cumulative Gain (NDCG)",
        "what_it_does": "Calculates position-discounted graded relevance score for ranking quality.",
        "example": '{"retrieved_ids": ["a", "b"], "relevance_scores": {"a": 2.0, "b": 1.0}, "k": 2}',
        "category": "observability",
    },
    "vec-obs-hit-rate": {
        "id": "ALGO-VEC-OBS-161",
        "algo_name": "VectorObservabilityAlgoHitRate",
        "plain_name": "Hit Rate Metric",
        "what_it_does": "Evaluates the proportion of queries with at least one relevant retrieved candidate.",
        "example": '{"query_evaluations": [{"retrieved_ids": ["a"], "relevant_ids": ["a"]}]}',
        "category": "observability",
    },
    "vec-obs-relative-distance-error": {
        "id": "ALGO-VEC-OBS-162",
        "algo_name": "VectorObservabilityAlgoRelativeDistanceError",
        "plain_name": "Relative Distance Error (RDE)",
        "what_it_does": "Quantifies distance distortion introduced by approximate nearest neighbor or quantization.",
        "example": '{"approximate_distances": [1.05, 1.1], "exact_distances": [1.0, 1.0]}',
        "category": "observability",
    },
    "vec-obs-llm-as-judge": {
        "id": "ALGO-VEC-OBS-163",
        "algo_name": "VectorObservabilityAlgoLlmAsJudge",
        "plain_name": "LLM-as-a-Judge Evaluation",
        "what_it_does": "Aggregates and audits LLM judge relevance scores for retrieved context.",
        "example": '{"evaluations": [{"query_id": "q1", "judge_score": 0.9, "reasoning": "Accurate"}], "pass_threshold": 0.7}',
        "category": "observability",
    },
    "vec-obs-golden-query-regression": {
        "id": "ALGO-VEC-OBS-164",
        "algo_name": "VectorObservabilityAlgoGoldenQueryRegression",
        "plain_name": "Golden Query Regression Testing",
        "what_it_does": "Detects regressions between current search results and golden baseline sets.",
        "example": '{"current_results": {"q1": ["d1", "d2"]}, "golden_results": {"q1": ["d1", "d2"]}, "k": 2}',
        "category": "observability",
    },
    "vec-obs-online-implicit-feedback": {
        "id": "ALGO-VEC-OBS-165",
        "algo_name": "VectorObservabilityAlgoOnlineImplicitFeedback",
        "plain_name": "Online Implicit Feedback Tracking",
        "what_it_does": "Analyzes CTR and dwell-time implicit signals to score retrieval relevance.",
        "example": '{"retrieved_ids": ["d1", "d2"], "clicked_ids": ["d1"], "dwell_times_sec": {"d1": 25.0}}',
        "category": "observability",
    },
    "vec-obs-interleaving-experiments": {
        "id": "ALGO-VEC-OBS-166",
        "algo_name": "VectorObservabilityAlgoInterleavingExperiments",
        "plain_name": "Team Draft Interleaving",
        "what_it_does": "Performs unbiased online A/B comparison between two retrieval rankings.",
        "example": '{"list_a": ["a1", "a2"], "list_b": ["b1", "b2"], "clicked_ids": ["a1"]}',
        "category": "observability",
    },
    "vec-obs-faithfulness-groundedness": {
        "id": "ALGO-VEC-OBS-167",
        "algo_name": "VectorObservabilityAlgoFaithfulnessGroundedness",
        "plain_name": "Faithfulness & Groundedness Metric",
        "what_it_does": "Calculates overlap and hallucination risk between generated claims and context.",
        "example": '{"answer_claims": ["Vectors are numeric arrays"], "retrieved_context_chunks": ["Vectors are arrays of numbers"]}',
        "category": "observability",
    },
    "vec-obs-centroid-shift": {
        "id": "ALGO-VEC-OBS-168",
        "algo_name": "VectorObservabilityAlgoCentroidShift",
        "plain_name": "Centroid Shift Drift Detection",
        "what_it_does": "Computes cosine and Euclidean shift between baseline and current dataset centroids.",
        "example": '{"baseline_vectors": [[1.0, 0.0]], "current_vectors": [[0.99, 0.01]]}',
        "category": "observability",
    },
    "vec-obs-mmd": {
        "id": "ALGO-VEC-OBS-169",
        "algo_name": "VectorObservabilityAlgoMmd",
        "plain_name": "Maximum Mean Discrepancy (MMD)",
        "what_it_does": "Non-parametric kernel test for high-dimensional vector distribution drift.",
        "example": '{"sample_p": [[1.0, 0.0]], "sample_q": [[1.0, 0.0]], "gamma": 1.0}',
        "category": "observability",
    },
    "vec-obs-psi-ks-drift": {
        "id": "ALGO-VEC-OBS-170",
        "algo_name": "VectorObservabilityAlgoPsiKsDrift",
        "plain_name": "PSI & KS-Test Drift",
        "what_it_does": "Measures Population Stability Index and Kolmogorov-Smirnov distance on scalar distributions.",
        "example": '{"baseline_distribution": [0.1, 0.2, 0.3], "current_distribution": [0.12, 0.22, 0.32]}',
        "category": "observability",
    },
    "vec-obs-similarity-score-distribution": {
        "id": "ALGO-VEC-OBS-171",
        "algo_name": "VectorObservabilityAlgoSimilarityScoreDistribution",
        "plain_name": "Similarity Score Distribution Profiler",
        "what_it_does": "Monitors similarity score quantiles and detects score compression or degradation.",
        "example": '{"similarity_scores": [0.95, 0.88, 0.82, 0.79]}',
        "category": "observability",
    },
    "vec-obs-vector-norm-distribution": {
        "id": "ALGO-VEC-OBS-172",
        "algo_name": "VectorObservabilityAlgoVectorNormDistribution",
        "plain_name": "Vector Norm Distribution Monitor",
        "what_it_does": "Audits L2 norm consistency to flag unnormalized vectors or embedding anomalies.",
        "example": '{"vectors": [[1.0, 0.0], [0.707, 0.707]], "expected_norm": 1.0}',
        "category": "observability",
    },
    "vec-obs-partition-cluster-balance": {
        "id": "ALGO-VEC-OBS-173",
        "algo_name": "VectorObservabilityAlgoPartitionClusterBalance",
        "plain_name": "Partition & Cluster Balance Monitor",
        "what_it_does": "Calculates entropy, Gini coefficient, and imbalance ratios across index partitions.",
        "example": '{"cluster_sizes": [100, 105, 98, 102]}',
        "category": "observability",
    },
    "vec-obs-hubness-measurement": {
        "id": "ALGO-VEC-OBS-174",
        "algo_name": "VectorObservabilityAlgoHubnessMeasurement",
        "plain_name": "Hubness Measurement",
        "what_it_does": "Quantifies k-occurrence skewness and detects problematic hub vectors in high dimensions.",
        "example": '{"nearest_neighbor_graph": {"q1": ["d1", "d2"], "q2": ["d1", "d3"]}, "total_queries": 2}',
        "category": "observability",
    },
    "vec-obs-intrinsic-dimension": {
        "id": "ALGO-VEC-OBS-175",
        "algo_name": "VectorObservabilityAlgoIntrinsicDimension",
        "plain_name": "Intrinsic Dimension Estimation (Two-NN)",
        "what_it_does": "Estimates true manifold dimensionality using the Two-NN algorithm.",
        "example": '{"vectors": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}',
        "category": "observability",
    },
    "vec-obs-outlier-detection": {
        "id": "ALGO-VEC-OBS-176",
        "algo_name": "VectorObservabilityAlgoOutlierDetection",
        "plain_name": "Vector Outlier Detection",
        "what_it_does": "Identifies isolated anomaly vectors based on local neighbor density and centroid distance.",
        "example": '{"vectors": [[1.0, 1.0], [1.1, 0.9], [10.0, 10.0]]}',
        "category": "observability",
    },
    "vec-obs-query-ood-detection": {
        "id": "ALGO-VEC-OBS-177",
        "algo_name": "VectorObservabilityAlgoQueryOodDetection",
        "plain_name": "Query Out-of-Distribution Detection",
        "what_it_does": "Flags out-of-distribution queries exceeding reference corpus distance radii.",
        "example": '{"query_vector": [5.0, 5.0], "centroid": [0.0, 0.0], "reference_radii": [1.0, 1.5, 2.0]}',
        "category": "observability",
    },
    "vec-obs-latency-histograms": {
        "id": "ALGO-VEC-OBS-178",
        "algo_name": "VectorObservabilityAlgoLatencyHistograms",
        "plain_name": "Latency Histograms & Percentiles",
        "what_it_does": "Computes p50, p90, p95, p99 latencies and verifies SLO compliance.",
        "example": '{"latencies_ms": [10.0, 15.0, 20.0, 45.0, 80.0], "slo_p95_ms": 50.0}',
        "category": "observability",
    },
    "vec-obs-red-use-methods": {
        "id": "ALGO-VEC-OBS-179",
        "algo_name": "VectorObservabilityAlgoRedUseMethods",
        "plain_name": "RED & USE Observability Framework",
        "what_it_does": "Synthesizes Rate, Errors, Duration (RED) and Utilization, Saturation, Errors (USE) health metrics.",
        "example": '{"requests_per_sec": 500.0, "error_rate": 0.002, "duration_p99_ms": 35.0, "utilization_pct": 65.0, "saturation_pct": 10.0, "system_errors": 0}',
        "category": "observability",
    },
    "vec-obs-slo-error-budget-burn": {
        "id": "ALGO-VEC-OBS-180",
        "algo_name": "VectorObservabilityAlgoSloErrorBudgetBurn",
        "plain_name": "SLO & Error Budget Burn Rate",
        "what_it_does": "Tracks error budget consumption and triggers multi-window burn rate alerts.",
        "example": '{"slo_target_percentage": 99.9, "error_budget_window_hours": 720.0, "measured_error_rate": 0.001}',
        "category": "observability",
    },
    "vec-obs-distributed-tracing": {
        "id": "ALGO-VEC-OBS-181",
        "algo_name": "VectorObservabilityAlgoDistributedTracing",
        "plain_name": "Distributed Tracing Span Processor",
        "what_it_does": "Audits trace span hierarchies, bottleneck durations, and dependency graphs.",
        "example": '{"trace_id": "tr-101", "spans": [{"span_id": "s1", "name": "vector_search", "start_time_ms": 0.0, "end_time_ms": 25.0}]}',
        "category": "observability",
    },
    "vec-obs-freshness-lag": {
        "id": "ALGO-VEC-OBS-182",
        "algo_name": "VectorObservabilityAlgoFreshnessLag",
        "plain_name": "Index Freshness & Replication Lag",
        "what_it_does": "Calculates mean and max elapsed duration between upstream mutations and index availability.",
        "example": '{"source_commit_timestamps_sec": [100.0, 110.0], "index_indexed_timestamps_sec": [102.0, 113.0]}',
        "category": "observability",
    },
    "vec-obs-graph-index-health": {
        "id": "ALGO-VEC-OBS-183",
        "algo_name": "VectorObservabilityAlgoGraphIndexHealth",
        "plain_name": "Graph Index Connectivity & Health",
        "what_it_does": "Audits HNSW/Vamana graph connectivity, isolated nodes, and average node degrees.",
        "example": '{"adjacency_list": {"0": ["1", "2"], "1": ["0"], "2": ["0"]}}',
        "category": "observability",
    },
    "vec-obs-tombstone-ratio": {
        "id": "ALGO-VEC-OBS-184",
        "algo_name": "VectorObservabilityAlgoTombstoneRatio",
        "plain_name": "Tombstone Accumulation Monitor",
        "what_it_does": "Calculates deleted tombstone ratio and triggers index vacuuming recommendations.",
        "example": '{"total_indexed_records": 10000, "active_tombstones": 2500, "compaction_threshold_ratio": 0.20}',
        "category": "observability",
    },
    "vec-obs-cache-hit-ratio-memory": {
        "id": "ALGO-VEC-OBS-185",
        "algo_name": "VectorObservabilityAlgoCacheHitRatioMemory",
        "plain_name": "Cache Hit Ratio & Memory Monitor",
        "what_it_does": "Monitors vector search caching efficiency and memory footprint bounds.",
        "example": '{"cache_hits": 850, "cache_misses": 150, "allocated_memory_bytes": 1073741824, "max_memory_capacity_bytes": 2147483648}',
        "category": "observability",
    },
    "vec-obs-capacity-planning-littles-law": {
        "id": "ALGO-VEC-OBS-186",
        "algo_name": "VectorObservabilityAlgoCapacityPlanningLittlesLaw",
        "plain_name": "Capacity Planning via Little's Law",
        "what_it_does": "Calculates required concurrent worker capacity and thread provisioning from throughput and latency.",
        "example": '{"target_throughput_qps": 500.0, "average_latency_seconds": 0.02, "peak_load_safety_multiplier": 1.5}',
        "category": "observability",
    },
    "vec-obs-consumer-lag": {
        "id": "ALGO-VEC-OBS-187",
        "algo_name": "VectorObservabilityAlgoConsumerLag",
        "plain_name": "Streaming Ingestion Consumer Lag",
        "what_it_does": "Monitors Kafka/Pulsar ingestion topic partition offsets and consumer group lag.",
        "example": '{"partition_offsets": {"p0": {"log_end_offset": 5000, "current_offset": 4800}}}',
        "category": "observability",
    },
    "vec-obs-cardinality-safe-labels": {
        "id": "ALGO-VEC-OBS-188",
        "algo_name": "VectorObservabilityAlgoCardinalitySafeLabels",
        "plain_name": "Cardinality-Safe Metric Labeling",
        "what_it_does": "Redacts high-cardinality metadata keys from metric tags to prevent Prometheus explosion.",
        "example": '{"labels": {"tenant_id": "tenant-1", "user_uuid": "123e4567-e89b-12d3-a456-426614174000"}, "allowed_cardinality_keys": ["tenant_id", "status"]}',
        "category": "observability",
    },
    "vec-obs-metric-anomaly-detection": {
        "id": "ALGO-VEC-OBS-189",
        "algo_name": "VectorObservabilityAlgoMetricAnomalyDetection",
        "plain_name": "Metric Anomaly Detection",
        "what_it_does": "Detects statistical anomalies and sudden spikes in observability time series using Z-score.",
        "example": '{"time_series_values": [10.0, 11.0, 10.5, 10.2, 50.0], "z_threshold": 2.5}',
        "category": "observability",
    },
    "vec-obs-quantile-sketches": {
        "id": "ALGO-VEC-OBS-190",
        "algo_name": "VectorObservabilityAlgoQuantileSketches",
        "plain_name": "Streaming Quantile Estimation",
        "what_it_does": "Computes memory-efficient approximate percentiles over high-throughput streaming observations.",
        "example": '{"raw_stream_values": [12.0, 15.0, 18.0, 22.0, 30.0], "requested_quantiles": [0.5, 0.9, 0.99]}',
        "category": "observability",
    },
    "vec-obs-query-explain": {
        "id": "ALGO-VEC-OBS-191",
        "algo_name": "VectorObservabilityAlgoQueryExplain",
        "plain_name": "Vector Query Execution Explain",
        "what_it_does": "Produces structured explanation of execution stages, candidate counts, and index parameters.",
        "example": '{"query_text": "cloud storage", "applied_filters": {"region": "us-west"}, "candidate_count": 120, "returned_count": 10}',
        "category": "observability",
    },
    "vec-obs-retrieval-trace-logging": {
        "id": "ALGO-VEC-OBS-192",
        "algo_name": "VectorObservabilityAlgoRetrievalTraceLogging",
        "plain_name": "Retrieval Trace Sampling & Logging",
        "what_it_does": "Performs privacy-preserving, sampled logging of retrieval queries, docs, and scores.",
        "example": '{"query_id": "q-99", "query_text": "confidential user query", "retrieved_doc_ids": ["doc-1"], "score_list": [0.88], "sample_rate": 1.0, "anonymize_text": true}',
        "category": "observability",
    },
    "vec-obs-embedding-visualization": {
        "id": "ALGO-VEC-OBS-193",
        "algo_name": "VectorObservabilityAlgoEmbeddingVisualization",
        "plain_name": "Embedding Space 2D/3D Projection",
        "what_it_does": "Projects high-dimensional embeddings to 2D coordinates for UI visualization via PCA.",
        "example": '{"vectors": [[1.0, 0.5, 0.2], [0.2, 0.8, 0.1]], "target_dimensions": 2}',
        "category": "observability",
    },
    "vec-obs-failure-clustering": {
        "id": "ALGO-VEC-OBS-194",
        "algo_name": "VectorObservabilityAlgoFailureClustering",
        "plain_name": "Retrieval Failure Clustering",
        "what_it_does": "Clusters zero-result or low-relevance queries into semantic topic groups for diagnostics.",
        "example": '{"failed_queries": [{"query_id": "q1", "text": "error 500", "embedding": [0.9, 0.1]}, {"query_id": "q2", "text": "http 500 crash", "embedding": [0.88, 0.12]}]}',
        "category": "observability",
    },
    "vec-obs-canary-probes": {
        "id": "ALGO-VEC-OBS-195",
        "algo_name": "VectorObservabilityAlgoCanaryProbes",
        "plain_name": "Synthetic Canary Probing",
        "what_it_does": "Evaluates synthetic probe execution results to monitor end-to-end vector search availability.",
        "example": '{"probe_results": [{"probe_id": "p1", "success": true, "latency_ms": 12.0}], "max_tolerable_error_rate": 0.0}',
        "category": "observability",
    },
    "vec-obs-shadow-traffic-comparison": {
        "id": "ALGO-VEC-OBS-196",
        "algo_name": "VectorObservabilityAlgoShadowTrafficComparison",
        "plain_name": "Shadow Traffic Differential Analysis",
        "what_it_does": "Compares primary production results vs shadow pipeline candidate results in real time.",
        "example": '{"primary_results": ["d1", "d2"], "shadow_results": ["d1", "d2"], "primary_latency_ms": 15.0, "shadow_latency_ms": 12.0}',
        "category": "observability",
    },
    "vec-obs-data-lineage": {
        "id": "ALGO-VEC-OBS-197",
        "algo_name": "VectorObservabilityAlgoDataLineage",
        "plain_name": "Vector & Chunk Lineage Audit",
        "what_it_does": "Traces complete provenance chain from source document to chunk, embedding model, and index.",
        "example": '{"record_id": "vec-100", "lineage_events": [{"event_type": "EMBED", "model_version": "text-embedding-3-small"}]}',
        "category": "observability",
    },
    "vec-obs-reconciliation-checks": {
        "id": "ALGO-VEC-OBS-198",
        "algo_name": "VectorObservabilityAlgoReconciliationChecks",
        "plain_name": "Source-to-Vector Reconciliation Audit",
        "what_it_does": "Audits consistency between primary relational/document store records and vector index vectors.",
        "example": '{"source_id_list": ["doc-1", "doc-2"], "vector_index_id_list": ["doc-1"]}',
        "category": "observability",
    },
    "vec-obs-cost-accounting": {
        "id": "ALGO-VEC-OBS-199",
        "algo_name": "VectorObservabilityAlgoCostAccounting",
        "plain_name": "Vector Storage & Query Cost Accounting",
        "what_it_does": "Calculates infrastructure cost per tenant based on vector memory footprint and query load.",
        "example": '{"indexed_vectors_count": 1000000, "dimension": 1536, "monthly_query_count": 5000000}',
        "category": "observability",
    },
    "vec-obs-feedback-improvement-loop": {
        "id": "ALGO-VEC-OBS-200",
        "algo_name": "VectorObservabilityAlgoFeedbackImprovementLoop",
        "plain_name": "Automated Quality Improvement Loop",
        "what_it_does": "Synthesizes observability signals to trigger automated index tuning or re-embedding workflows.",
        "example": '{"low_performing_queries": ["query-alpha"], "average_recall_score": 0.75, "tombstone_ratio": 0.25}',
        "category": "observability",
    },
}
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

try:
    from src.features.audit.service.audit_service import AuditService
    from src.features.refactor.service.refactor_service import RefactorService
    from src.features.policy_sync.service.policy_sync_service import PolicySyncService
    from src.features.rag.service.rag_service import RAGService
    from src.features.rag.types.rag_types import RAGQueryRequest
    from src.features.agent.service.agent_service import AgentService
    from src.features.agent.types.agent_types import AgentExecutionRequest
except ImportError:
    AuditService = None
    RefactorService = None
    PolicySyncService = None
    RAGService = None
    RAGQueryRequest = None
    AgentService = None
    AgentExecutionRequest = None
from src.infra.adapters.knowledge.policy_rules_loader import PolicyRulesMarkdownLoader
from src.infra.adapters.vector.in_memory_vector_adapter import InMemoryCosineVectorAdapter
from src.infra.adapters.llm.openai_compatible_adapter import OpenAICompatibleAdapter
from src.infra.adapters.llm.mock_llm_adapter import MockLLMAdapter
from src.features.code_engine.service.code_engine_service import CodeEngineService
from src.features.code_engine.registry.algorithm_catalog import BUILTIN_ALGORITHM_CONTRACTS, BUILTIN_TYPE_ADAPTERS
from src.features.code_engine.service.algorithm_composer_service import AlgorithmComposerService
from src.infra.adapters.database.in_memory_algorithm_registry_adapter import InMemoryAlgorithmRegistryAdapter
from src.infra.adapters.database.migration_runner import DatabaseMigrationRunner



def _init_rag_and_agent(rules_dir: str, backend: str):
    knowledge_source = PolicyRulesMarkdownLoader(base_rules_dir=rules_dir)
    vector_store = InMemoryCosineVectorAdapter()

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for openai backend")
        llm = OpenAICompatibleAdapter(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )
    elif backend == "ollama":
        llm = OpenAICompatibleAdapter(
            api_key="ollama",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model_name=os.environ.get("OLLAMA_MODEL", "llama3"),
        )
    else:
        llm = MockLLMAdapter()

    rag_service = RAGService(knowledge_source, vector_store, llm)
    agent_service = AgentService(rag_service, llm)
    return rag_service, agent_service


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
        "contracts", "compose", "audit", "rag", "agent", "refactor", "policy-check", "migrate",
        "serve", "completion", "algo", "scaffold-feature", "scaffold-package", "create-file",
        "link-file", "impact", "feature-map", "package-summary", "feature-files", "graph-scan",
        "file-lineage"
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


def handle_audit_command(args: argparse.Namespace) -> int:
    audit_service = AuditService()
    summary = audit_service.run_full_audit(args.root)

    if args.json:
        print(json.dumps(asdict(summary), indent=2))
    else:
        print("\n" + "=" * 80)
        print("REPOSITORY INVARIANT & ARCHITECTURE AUDIT SUMMARY")
        print("=" * 80)
        print(f"Total Rules Checked : {summary.total_rules_checked}")
        print(f"Passed Rules        : {summary.passed_rules}")
        print(f"Failed Rules        : {summary.failed_rules}")
        print(f"Total Violations    : {summary.total_violations}")
        print(f"Audit Status        : {summary.status.upper()}")
        print("=" * 80)
        if summary.failed_rules > 0:
            print("\nRule Breakdown:")
            for report in summary.rule_reports:
                if report.status == "fail":
                    print(f"\n❌ [{report.rule_id}] {report.rule_name}")
                    for finding in report.findings[:5]:
                        print(f"   • {finding.file_path}:{finding.line_number} - {finding.message}")
        print("=" * 80 + "\n")
    return 0 if summary.failed_rules == 0 else 1


def handle_rag_command(args: argparse.Namespace) -> int:
    rag_service, _ = _init_rag_and_agent(args.rules_dir, args.backend)
    if args.action == "index":
        indexed = rag_service.index_policies()
        print(f"✅ Successfully indexed {indexed} policy rule sections into vector memory.")
        return 0
    elif args.action == "search":
        if not args.query:
            print("Error: --query is required for rag search")
            return 1
        rag_service.index_policies()
        request = RAGQueryRequest(query=args.query, category=args.category, top_k=args.top_k)
        response = rag_service.query_policies(request)
        if args.json:
            print(json.dumps(asdict(response), indent=2))
        else:
            print(f"\n{'='*75}")
            print(f"🔍 RAG RETRIEVAL: {len(response.results)} matches for '{args.query}'")
            print(f"{'='*75}\n")
            for i, match in enumerate(response.results, 1):
                print(f"[{i}] {match.rule_title} (Similarity: {match.similarity_score:.3f})")
                print(f"    Source: {match.source_file}")
                print(f"    Content Preview: {match.content[:150]}...\n")
        return 0
    return 0


def handle_agent_command(args: argparse.Namespace) -> int:
    _, agent_service = _init_rag_and_agent(args.rules_dir, args.backend)
    req = AgentExecutionRequest(task_prompt=args.prompt, target_directory=args.target_dir, max_steps=args.max_steps)
    res = agent_service.execute_task(req)
    if args.json:
        print(json.dumps(asdict(res), indent=2))
    else:
        print(f"\n{'='*75}")
        print(f"🤖 AGENT WORKFLOW COMPLETED: Status = {res.status.upper()}")
        print(f"{'='*75}")
        print(f"Summary: {res.summary}\n")
        print("Reasoning Trajectory:")
        for step in res.trajectory:
            print(f"  Step {step.step_number}: {step.action_type} -> Tool: {step.tool_name}")
            print(f"    Thought: {step.thought}")
            if step.tool_output:
                preview = str(step.tool_output)[:120].replace('\n', ' ')
                print(f"    Result: {preview}...")
        print(f"{'='*75}\n")
    return 0 if res.status == "success" else 1


def handle_refactor_command(args: argparse.Namespace) -> int:
    service = RefactorService()
    extensions = [ext.strip() for ext in args.ext.split(",") if ext.strip()]
    res = service.execute_batch_refactor(
        root_dir=args.root,
        find_pattern=args.find,
        replace_text=args.replace,
        allowed_extensions=extensions,
        is_regex=args.regex,
        dry_run=not args.apply,
    )
    if args.json:
        print(json.dumps(asdict(res), indent=2))
    else:
        mode = "APPLIED" if args.apply else "DRY-RUN (Simulated)"
        print(f"\n{'='*75}")
        print(f"🔄 BATCH REFACTOR SUMMARY [{mode}]")
        print(f"{'='*75}")
        print(f"Files Scanned   : {res.total_files_scanned}")
        print(f"Files Modified  : {res.total_files_modified}")
        print(f"Total Matches   : {res.total_occurrences_replaced}")
        print(f"{'='*75}\n")
    return 0


def handle_policy_check_command(args: argparse.Namespace) -> int:
    sync_service = PolicySyncService()
    violations = sync_service.audit_contract_file(args.path)
    if args.json:
        print(json.dumps([asdict(v) for v in violations], indent=2))
    else:
        print(f"\n{'='*75}")
        print(f"📋 POLICY CONTRACT AUDIT: {len(violations)} Violations in {args.path}")
        print(f"{'='*75}")
        if not violations:
            print("✅ All algorithm entries strictly conform to the engineering contract standard.")
        else:
            for v in violations:
                print(f"  ❌ Line {v.line_number}: [{v.rule_id}] {v.message}")
        print(f"{'='*75}\n")
    return 0 if len(violations) == 0 else 1


def handle_migrate_command(args: argparse.Namespace) -> int:
    db_target = getattr(args, "db", "sqlite")
    if db_target == "sqlite":
        db_url = f"sqlite:///{getattr(args, 'sqlite_path', 'policy_registry.db')}"
    else:
        db_url = getattr(args, "postgres_url", "postgresql://postgres:postgres@localhost:5432/postgres")

    runner = DatabaseMigrationRunner(db_url)
    action = getattr(args, "action", "run")

    if action == "run":
        runner.run_migrations()
        count = runner.seed_algorithm_catalog()
        print(f"✅ Migration successful on {db_target}: Applied schema migrations, Seeded {count} algorithm contracts & type adapters.")
        return 0
    elif action == "rollback":
        res = runner.rollback_migrations()
        print(f"🔄 Rollback successful on {db_target}: {res}")
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


def handle_scaffold_feature_command(args: argparse.Namespace) -> int:
    from src.features.folder_scaffold.types.folder_scaffold_types import ScaffoldFeatureCommand
    from src.shared.types.envelope_types import TraceContext

    feature_name = getattr(args, "feature_name", None) or getattr(args, "name_flag", None) or getattr(args, "name", None)
    if not feature_name:
        print("Error: Feature name is required.")
        return 1

    service = get_folder_scaffold_service(args)
    base_dir = getattr(args, "base_dir", "src/features")
    package_root = getattr(args, "package_root", None)
    cmd = ScaffoldFeatureCommand(
        feature_name=feature_name,
        base_dir=base_dir,
        package_root=package_root,
    )
    ctx = TraceContext(
        trace_id="cli-scaffold-feature",
        request_id="cli-req-scaffold-feature",
        correlation_id="cli-corr-scaffold-feature",
    )
    result = service.scaffoldFeature(ctx, cmd)
    if result.is_err:
        print(f"Error scaffolding feature: {result.error()}")
        return 1
    res = result.unwrap()
    if getattr(args, "json", False):
        print(json.dumps(res.model_dump(), indent=2))
    else:
        print("================================================================================")
        print("FEATURE DIRECTORIES SCAFFOLDED")
        print("================================================================================")
        print(f"Feature:              {res.target_name}")
        print(f"Target Directory:     {res.target_path}")
        print(f"Directories Created:  {res.total_directories}")
        print(f"Neo4j Knowledge Graph Synced (Label: Feature)")
        print("================================================================================")
    return 0


def handle_scaffold_package_command(args: argparse.Namespace) -> int:
    from src.features.folder_scaffold.types.folder_scaffold_types import ScaffoldPackageCommand
    from src.shared.types.envelope_types import TraceContext

    package_name = getattr(args, "package_name", None) or getattr(args, "name_flag", None) or getattr(args, "name", None)
    if not package_name:
        print("Error: Package name is required.")
        return 1

    service = get_folder_scaffold_service(args)
    base_dir = getattr(args, "base_dir", ".")
    cmd = ScaffoldPackageCommand(
        package_name=package_name,
        base_dir=base_dir,
    )
    ctx = TraceContext(
        trace_id="cli-scaffold-package",
        request_id="cli-req-scaffold-package",
        correlation_id="cli-corr-scaffold-package",
    )
    result = service.scaffoldPackage(ctx, cmd)
    if result.is_err:
        print(f"Error scaffolding package: {result.error()}")
        return 1
    res = result.unwrap()
    if getattr(args, "json", False):
        print(json.dumps(res.model_dump(), indent=2))
    else:
        print("================================================================================")
        print("PACKAGE DIRECTORIES SCAFFOLDED")
        print("================================================================================")
        print(f"Package:              {res.target_name}")
        print(f"Target Root:          {res.target_path}")
        print(f"Directories Created:  {res.total_directories}")
        print(f"Neo4j Knowledge Graph Synced (Label: PackageRoot)")
        print("================================================================================")
    return 0



def handle_impact_command(args: argparse.Namespace) -> int:
    from src.features.file_structure.service.file_structure_service import FileStructureDomainService

    adapter = resolve_graph_adapter(args)
    service = FileStructureDomainService(graph_store=adapter)
    target_id = getattr(args, "target_id", None) or getattr(args, "target", None)
    if not target_id:
        print("Error: Target ID or file path is required.")
        return 1

    direction = getattr(args, "direction", "UPSTREAM").upper()
    max_depth = int(getattr(args, "max_depth", 6))

    service.graph_sync.sync_feature_dag(target_id)

    report = service.analyze_node_impact(
        target_id=target_id,
        direction=direction,
        max_depth=max_depth,
    )
    if getattr(args, "json", False):
        out = {
            "target_id": report.target_id,
            "target_label": report.target_label,
            "direction": report.direction,
            "blast_radius_score": report.blast_radius_score,
            "risk_level": report.risk_level,
            "impacted_nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "path": n.path,
                    "depth": n.depth,
                    "relationship": n.relationship,
                    "risk_factor": n.risk_factor,
                }
                for n in report.impacted_nodes
            ],
            "affected_layers": report.affected_layers,
            "potential_breaking_risks": report.potential_breaking_risks,
            "required_verification_commands": report.required_verification_commands,
            "recommended_mitigation_recipe": report.recommended_mitigation_recipe,
        }
        print(json.dumps(out, indent=2))
    else:
        print("================================================================================")
        print("ARCHITECTURAL IMPACT & BLAST RADIUS ANALYSIS REPORT")
        print("================================================================================")
        print(f"Target Node: {report.target_id} (Label: {report.target_label})")
        print(f"Direction: {report.direction} | Max Depth: {max_depth}")
        print(f"Risk Classification: {report.risk_level} | Blast Radius Score: {report.blast_radius_score}")
        print("--------------------------------------------------------------------------------")
        print(f"Impacted Nodes ({len(report.impacted_nodes)} total):")
        for node in report.impacted_nodes:
            print(f"  [{node.relationship}] Depth {node.depth}: {node.label} ({node.id}) -> {node.path} [Risk: {node.risk_factor}]")
        print("--------------------------------------------------------------------------------")
        print("Required Verification Commands:")
        for cmd in report.required_verification_commands:
            print(f"  $ {cmd}")
        print("--------------------------------------------------------------------------------")
        print(f"Mitigation: {report.recommended_mitigation_recipe}")
    return 0


def handle_feature_map_command(args: argparse.Namespace) -> int:
    from src.features.file_structure.service.file_structure_service import FileStructureDomainService

    adapter = resolve_graph_adapter(args)
    service = FileStructureDomainService(graph_store=adapter)
    feature_name = getattr(args, "feature_name", None) or getattr(args, "name", None)
    if not feature_name:
        print("Error: Feature name is required.")
        return 1

    service.graph_sync.sync_feature_dag(feature_name)
    f_map = service.get_feature_map(feature_name)

    if getattr(args, "json", False):
        print(json.dumps(f_map, indent=2))
    else:
        print("================================================================================")
        print(f"ARCHITECTURAL MAP FOR FEATURE: {f_map['feature']}")
        print("================================================================================")
        print(f"Context Boundary: {f_map.get('context', {}).get('file_path')}")
        print(f"Public Facade:    {f_map.get('facade', {}).get('file_path')}")
        print(f"Domain Types:     {f_map.get('types', {}).get('file_path')}")
        print(f"Schema ACL:       {f_map.get('schema', {}).get('file_path')}")
        print(f"REST Router:      {f_map.get('router', {}).get('file_path')} (Prefix: {f_map.get('router', {}).get('route_prefix')})")
        print(f"Route Rules:      {f_map.get('route_rules', {}).get('file_path')}")
        print(f"REST Handler:     {f_map.get('handler', {}).get('file_path')}")
        print(f"Domain Service:   {f_map.get('service', {}).get('file_path')}")
        print(f"Business Rules:   {f_map.get('rules', {}).get('file_path')}")
        print(f"State Machine:    {f_map.get('machine', {}).get('file_path')}")
        print(f"Workflow DAG:     {f_map.get('workflow', {}).get('file_path')}")
        print(f"Repository Port:  {f_map.get('repository', {}).get('port')}")
        print(f"Repository Impl:  {f_map.get('repository', {}).get('adapter')}")
        print(f"Named SQL:        {f_map.get('repository', {}).get('queries')}")
    return 0


def handle_package_summary_command(args: argparse.Namespace) -> int:
    from src.features.file_structure.service.file_structure_service import FileStructureDomainService

    adapter = resolve_graph_adapter(args)
    service = FileStructureDomainService(graph_store=adapter)
    package_name = getattr(args, "package_name", None) or getattr(args, "name_flag", None) or getattr(args, "name", None)
    if not package_name:
        print("Error: Package name is required.")
        return 1

    summary = service.get_package_summary(package_name)
    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2))
    else:
        print("================================================================================")
        print(f"PACKAGE PORTFOLIO SUMMARY: {summary['package']}")
        print("================================================================================")
        print(f"Total Features Registered: {summary['total_features']}")
        print("--------------------------------------------------------------------------------")
        for idx, feat in enumerate(summary.get("features", []), start=1):
            print(f"  {idx}. Feature: {feat['feature_name']}")
            print(f"     Status:     {feat['status']}")
            print(f"     Owner:      {feat['owner']}")
            print(f"     Flag:       {feat['flags']}")
            print(f"     Migration:  {feat['migrations']}")
            print(f"     Contract:   {feat.get('contract_version', 'v1')}")
        print("================================================================================")
    return 0


def handle_feature_files_command(args: argparse.Namespace) -> int:
    from src.features.file_structure.service.file_structure_service import FileStructureDomainService

    adapter = resolve_graph_adapter(args)
    service = FileStructureDomainService(graph_store=adapter)
    feature_name = getattr(args, "feature_name", None) or getattr(args, "name_flag", None) or getattr(args, "name", None)
    if not feature_name:
        print("Error: Feature name is required.")
        return 1

    package_name = getattr(args, "package_name", None) or getattr(args, "package", None)
    res = service.get_feature_files(feature_name, package_name)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        print("================================================================================")
        print(f"FEATURE FILE MANIFEST: {res['feature']} (Total Files: {res['total_files']})")
        print("================================================================================")
        for f in res.get("files", []):
            print(f"  [{f['layer']}] {f['role']}: {f['path']}")
        print("================================================================================")
    return 0


def handle_graph_scan_command(args: argparse.Namespace) -> int:
    from src.features.file_structure.service.file_structure_service import FileStructureDomainService

    adapter = resolve_graph_adapter(args)
    service = FileStructureDomainService(graph_store=adapter)
    root_dir = getattr(args, "root_dir", None) or getattr(args, "dir", ".")

    res = service.scan_and_sync_workspace(root_dir=root_dir)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        print(f"Workspace Scan Complete: {res.get('status')}")
        print(f"Current State: {res.get('state')}")
        stats = res.get("stats", {})
        print(f"Indexed Files: {stats.get('indexed_files', 0)}")
    return 0


def handle_file_lineage_command(args: argparse.Namespace) -> int:
    from src.features.file_structure.service.file_structure_service import FileStructureDomainService

    adapter = resolve_graph_adapter(args)
    service = FileStructureDomainService(graph_store=adapter)
    file_path = getattr(args, "file_path", None) or getattr(args, "file", None)
    if not file_path:
        print("Error: Target file path or node ID is required.")
        return 1

    try:
        max_depth = getattr(args, "max_depth", 5) or 5
        lineage = service.get_file_lineage(file_path, max_depth=max_depth)
    except Exception as exc:
        print(f"Error resolving lineage: {exc}")
        return 1

    if getattr(args, "json", False):
        print(json.dumps(lineage, indent=2))
    else:
        t = lineage["target"]
        s = lineage["summary"]
        print("================================================================================")
        print("ARCHITECTURAL FILE LINEAGE & DEPENDENCY NEXUS")
        print("================================================================================")
        print(f"Target Node:     {t['node_id']}")
        print(f"Role / Label:    {t['label']} [{t['layer']}]")
        print(f"File Path:       {t['file_path']}")
        if t.get("feature_name"):
            print(f"Feature:         {t['feature_name']} (Package: {t.get('package_name', 'root')})")
        print("--------------------------------------------------------------------------------")
        print(f"Direct Upstream:   {s['direct_upstream_count']} callers/parents")
        print(f"Direct Downstream: {s['direct_downstream_count']} dependencies/children")
        print(f"Blast Radius:      {s['transitive_upstream_count']} transitive callers | {s['transitive_downstream_count']} transitive dependents")
        print("--------------------------------------------------------------------------------")
        print("DIRECT UPSTREAM (Who depends on or invokes this file):")
        if lineage["direct_upstream"]:
            for u in lineage["direct_upstream"]:
                print(f"  <- [{u['relationship']}] {u['label']} ({u['file_path']})")
        else:
            print("  (None - this is a root ingress or standalone entry point)")

        print("--------------------------------------------------------------------------------")
        print("DIRECT DOWNSTREAM (What this file depends on or calls):")
        if lineage["direct_downstream"]:
            for d in lineage["direct_downstream"]:
                print(f"  -> [{d['relationship']}] {d['label']} ({d['file_path']})")
        else:
            print("  (None - this is a terminal leaf node)")
        print("================================================================================")
    return 0


def get_folder_scaffold_service(args: argparse.Namespace):
    from src.infra.config.config_loader import AppConfigLoader
    from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
    from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
    from src.features.folder_scaffold.repository.folder_scaffold_repository import AlloyDBFolderScaffoldRepository
    from src.features.folder_scaffold.service.folder_scaffold_service import FolderScaffoldDomainService
    from src.features.folder_scaffold.types.folder_scaffold_deps import FolderScaffoldServiceDeps

    db_url = AppConfigLoader.get_database_url()
    kg_cfg = AppConfigLoader.get_knowledge_graph_config()
    repo = AlloyDBFolderScaffoldRepository(db_url=db_url)
    
    store = getattr(args, "store", None)
    neo4j_flag = getattr(args, "neo4j", False)
    if neo4j_flag or store == "neo4j" or os.environ.get("GRAPH_STORE") == "neo4j":
        graph_store = Neo4jGraphAdapter(
            uri=kg_cfg.get("uri"),
            user=kg_cfg.get("user"),
            password=kg_cfg.get("password"),
            database=kg_cfg.get("database", "neo4j"),
        )
    else:
        graph_store = Neo4jGraphAdapter(
            uri=kg_cfg.get("uri"),
            user=kg_cfg.get("user"),
            password=kg_cfg.get("password"),
            database=kg_cfg.get("database", "neo4j"),
        )
    scaffold_cfg = AppConfigLoader.get_scaffold_config()
    deps = FolderScaffoldServiceDeps(
        repository=repo,
        graph_store=graph_store,
        scaffold_config=scaffold_cfg,
    )
    return FolderScaffoldDomainService(deps=deps)




def handle_create_file_command(args: argparse.Namespace) -> int:
    from src.features.folder_scaffold.types.folder_scaffold_types import CreateFileCommand
    from src.shared.types.envelope_types import TraceContext

    service = get_folder_scaffold_service(args)

    file_path = getattr(args, "path", None) or getattr(args, "file_path", None)
    role = getattr(args, "role", None)
    rel_type = getattr(args, "rel_type", None) or getattr(args, "rel", None)
    target = getattr(args, "target", None)
    direction = getattr(args, "direction", "outgoing")
    package_name = getattr(args, "package_root", None)
    feature_name = getattr(args, "feature_name", None)

    if not file_path or not role or not rel_type or not target:
        print("Error: --path, --role, --rel, and --target are all REQUIRED parameters to create an architecturally linked file.")
        return 1

    try:
        cmd = CreateFileCommand(
            path=file_path,
            role=role,
            rel_type=rel_type,
            target=target,
            direction=direction,
            feature_name=feature_name,
            package_name=package_name,
        )
        ctx = TraceContext(
            trace_id="cli-create-file",
            request_id="cli-req-create-file",
            correlation_id="cli-corr-create-file",
        )
        result = service.createFile(ctx, cmd)
        if result.is_err:
            print(f"Error creating file: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("================================================================================")
            print("FILE CREATED & ARCHITECTURALLY LINKED")
            print("================================================================================")
            print(f"File Path:    {res.node.file_path}")
            print(f"Role / Layer: {res.node.role} [{res.node.layer}]")
            rel = res.relationship
            print(f"Relationship: {rel.source_id} -[:{rel.rel_type}]-> {rel.target_id}")
            print(f"AlloyDB:      Persisted ({res.status})")
            print(f"Neo4j:        Synced (Constraints Enforced)")
            print("================================================================================")
        return 0
    except Exception as exc:
        print(f"Error creating file: {exc}")
        return 1


def handle_link_file_command(args: argparse.Namespace) -> int:
    from src.features.folder_scaffold.types.folder_scaffold_types import LinkFilesCommand
    from src.shared.types.envelope_types import TraceContext

    service = get_folder_scaffold_service(args)

    source = getattr(args, "source", None)
    rel_type = getattr(args, "rel_type", None) or getattr(args, "rel", None)
    target = getattr(args, "target", None)

    if not source or not rel_type or not target:
        print("Error: --source, --rel, and --target are all REQUIRED parameters to link files.")
        return 1

    try:
        cmd = LinkFilesCommand(source=source, rel_type=rel_type, target=target)
        ctx = TraceContext(
            trace_id="cli-link-file",
            request_id="cli-req-link-file",
            correlation_id="cli-corr-link-file",
        )
        result = service.linkFiles(ctx, cmd)
        if result.is_err:
            print(f"Error linking files: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            rel = res.relationship
            print(f"Successfully linked: {rel.source_id} -[:{rel.rel_type}]-> {rel.target_id}")
            print(f"AlloyDB & Neo4j synchronized.")
        return 0
    except Exception as exc:
        print(f"Error linking files: {exc}")
        return 1



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
    audit_parser = subparsers.add_parser("audit", help="Run multi-vector invariant scan")
    audit_parser.add_argument("--root", default=".", help="Root directory")
    audit_parser.add_argument("--json", action="store_true", help="Output findings as JSON")

    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations and seed algorithm contracts")
    migrate_parser.add_argument("action", choices=["run", "status", "rollback", "seed", "parity"], default="run", nargs="?", help="Migration action")
    migrate_parser.add_argument("--db", choices=["sqlite", "alloydb", "postgres"], default="sqlite", help="Target database")
    migrate_parser.add_argument("--sqlite-path", default="policy_registry.db", help="SQLite database path")
    migrate_parser.add_argument("--postgres-url", default="postgresql://postgres:postgres@localhost:5432/postgres", help="PostgreSQL connection string")


    rag_parser = subparsers.add_parser("rag", help="Retrieve or index grounded policy rules")
    rag_parser.add_argument("action", choices=["search", "index"], help="RAG action")
    rag_parser.add_argument("--query", default="", help="Search query")
    rag_parser.add_argument("--category", default=None, help="Policy category filter")
    rag_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    rag_parser.add_argument("--rules-dir", default="../rules", help="Path to rules folder")
    rag_parser.add_argument("--backend", default="mock", choices=["mock", "openai", "ollama"], help="LLM backend")
    rag_parser.add_argument("--json", action="store_true", help="Output as JSON")

    agent_parser = subparsers.add_parser("agent", help="Run autonomous AI policy agent")
    agent_parser.add_argument("prompt", help="Task prompt for the agent")
    agent_parser.add_argument("--target-dir", default=".", help="Directory to analyze")
    agent_parser.add_argument("--max-steps", type=int, default=8, help="Maximum reasoning steps")
    agent_parser.add_argument("--rules-dir", default="../rules", help="Path to rules folder")
    agent_parser.add_argument("--backend", default="mock", choices=["mock", "openai", "ollama"], help="LLM backend")
    agent_parser.add_argument("--json", action="store_true", help="Output as JSON")

    refactor_parser = subparsers.add_parser("refactor", help="Execute safe batch search-and-replace")
    refactor_parser.add_argument("--root", default=".", help="Root directory")
    refactor_parser.add_argument("--find", required=True, help="Pattern to find")
    refactor_parser.add_argument("--replace", required=True, help="Replacement string")
    refactor_parser.add_argument("--ext", default=".go,.ts,.js,.py,.sql", help="Extensions")
    refactor_parser.add_argument("--regex", action="store_true", help="Treat pattern as regex")
    refactor_parser.add_argument("--apply", action="store_true", help="Apply mutations")
    refactor_parser.add_argument("--json", action="store_true", help="Output summary as JSON")

    policy_parser = subparsers.add_parser("policy-check", help="Audit markdown contracts")
    policy_parser.add_argument("--path", default="policies/rules/edgeCases/algos/agent-operating-contract.md", help="Contract path")
    policy_parser.add_argument("--json", action="store_true", help="Output report as JSON")

    # Scaffolding & Knowledge Graph Subcommands (Modularized in scaffold_cli.py)
    from src.api.cli.scaffold_cli import (
        register_scaffold_subparsers,
        handle_scaffold_feature,
        handle_scaffold_package,
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
        handle_scaffold_feature,
        handle_scaffold_package,
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
        "audit": handle_audit_command,
        "migrate": handle_migrate_command,
        "rag": handle_rag_command,
        "agent": handle_agent_command,
        "refactor": handle_refactor_command,
        "policy-check": handle_policy_check_command,
        "scaffold-feature": handle_scaffold_feature,
        "feature-scaffold": handle_scaffold_feature,
        "scaffold-package": handle_scaffold_package,
        "package-scaffold": handle_scaffold_package,
        "impact": handle_impact_command,
        "impact-analysis": handle_impact_command,
        "feature-map": handle_feature_map_command,
        "map": handle_feature_map_command,
        "package-summary": handle_package_summary_command,
        "pkg-summary": handle_package_summary_command,
        "feature-files": handle_feature_files_command,
        "feat-files": handle_feature_files_command,
        "graph-scan": handle_graph_scan_command,
        "graph-sync": handle_graph_scan_command,
        "file-lineage": handle_file_lineage_command,
        "file-relations": handle_file_lineage_command,
        "file-deps": handle_file_lineage_command,
        "lineage": handle_file_lineage_command,
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

