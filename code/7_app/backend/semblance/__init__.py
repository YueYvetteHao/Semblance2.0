"""Vendored Semblance core (from the standalone Semblance repo, `core/`) + a standalone BioLORD
embedder. Self-contained: no MCP, no LLM, no `config` module — just parse -> signatures ->
similarity + `embed`. Import submodules directly (e.g. `from semblance.signatures import
build_signatures`) so the server never pays for pandas unless it parses a table.
"""
