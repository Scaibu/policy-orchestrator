"""
================================================================================
BLUEPRINT: CODE SEARCH ALGORITHM & PATTERN RULES
================================================================================
1. OVERVIEW:
   Implementation of core text search algorithms:
   - Boyer-Moore-Horspool: Precomputes bad-character shift table. On mismatch,
     skips ahead by shift value based on mismatched character.
   - Aho-Corasick Automaton: Builds Trie with output nodes and BFS failure links.
     Executes multi-pattern search in a single O(N) pass.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All algorithm blueprints and shift mechanics are documented here.
================================================================================
"""
from typing import Dict, List, Tuple, Set
from collections import deque


def build_bad_character_table(pattern: str) -> Dict[str, int]:
    m = len(pattern)
    table: Dict[str, int] = {}
    for i in range(m - 1):
        table[pattern[i]] = m - 1 - i
    return table


def boyer_moore_horspool_search(text: str, pattern: str) -> List[int]:
    n = len(text)
    m = len(pattern)
    if m == 0 or n < m:
        return []

    shift_table = build_bad_character_table(pattern)
    matches: List[int] = []
    i = 0

    while i <= n - m:
        j = m - 1
        while j >= 0 and text[i + j] == pattern[j]:
            j -= 1
        if j < 0:
            matches.append(i)
            i += shift_table.get(text[i + m - 1], m)
        else:
            mismatch_char = text[i + m - 1]
            i += shift_table.get(mismatch_char, m)

    return matches


class AhoCorasickNode:
    def __init__(self) -> None:
        self.children: Dict[str, AhoCorasickNode] = {}
        self.failure: "AhoCorasickNode" = None  # type: ignore
        self.outputs: List[str] = []


class AhoCorasickAutomaton:
    def __init__(self, patterns: List[str]) -> None:
        self.root = AhoCorasickNode()
        self._build_trie(patterns)
        self._build_failure_links()

    def _build_trie(self, patterns: List[str]) -> None:
        for pat in patterns:
            if not pat:
                continue
            curr = self.root
            for ch in pat:
                if ch not in curr.children:
                    curr.children[ch] = AhoCorasickNode()
                curr = curr.children[ch]
            curr.outputs.append(pat)

    def _build_failure_links(self) -> None:
        queue = deque()
        for ch, child in self.root.children.items():
            child.failure = self.root
            queue.append(child)

        while queue:
            curr = queue.popleft()
            for ch, child in curr.children.items():
                fail = curr.failure
                while fail is not None and ch not in fail.children:
                    fail = fail.failure
                child.failure = fail.children[ch] if fail and ch in fail.children else self.root
                child.outputs.extend(child.failure.outputs)
                queue.append(child)

    def search(self, text: str) -> List[Tuple[int, str]]:
        results: List[Tuple[int, str]] = []
        curr = self.root
        for idx, ch in enumerate(text):
            while curr is not None and ch not in curr.children:
                curr = curr.failure
            if curr is None:
                curr = self.root
                continue
            curr = curr.children[ch]
            for pat in curr.outputs:
                results.append((idx - len(pat) + 1, pat))
        return results
