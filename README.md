# Live Provenance

A chat where Claude answers a data question with tools, and a 3D scene that builds the provenance graph (question → plan → sources → operations → findings → answer) in real time from what the model actually does.

## How the graph becomes real (not scripted)

- **Every tool result gets an id** (`q3`, `p7`, `src_overdose_2023`…) that is returned to the model inside the tool result text.
- The model must call `record_finding(text, from_ids)` and `record_claim(text, evidence_ids)` to produce findings and answer sentences. The ids it cites become the edges. No post-hoc guessing of dependencies.
- `query_table` emits a source node for each table it touches (once), and an operation node with real `rows in → rows out`.
- `search_sources` emits a source node per document hit and an operation node per passage, each with its own id.
- Caveats are findings with `kind="caveat"`; the system prompt asks the model to record interpretation choices there.

Everything the browser shows is a replay of events the server emitted during the run; `server.py` has the event vocabulary at the top.

## Run it

```bash
cd provenance-live
python -m venv .venv && source .venv/bin/activate
pip install anthropic fastapi "uvicorn[standard]" duckdb
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn server:app --reload --port 8000
```

Open http://localhost:8000 and press **Ask**.

Optional: `export CLAUDE_MODEL=claude-opus-5-5` (default is `claude-sonnet-5-5`).

## Data

- `data/overdose_2023.csv` — **partial** table: only the CDC 2023 age-adjusted rates that were verified while building this (WV, DC, SD, NE, US). Replace it with a full export from CDC WONDER or the NCHS Health E-Stat table and the ranking becomes complete. Any `*.csv` dropped into `data/` becomes a queryable table named after the file.
- `data/sources/*.md` — paraphrased passages from five public documents, each with its URL in a `url:` line. Add more `.md` files to grow the library. The search is a plain keyword match; swap it for embeddings or for Anthropic's server-side web search tool when you want open-ended research.

## Where to take it next

1. **Verification nodes.** After each `record_claim`, check that numbers in the claim appear in the cited tool outputs; emit a caveat node automatically when they don't. This catches the model mis-citing its own evidence.
2. **Streaming.** Switch `client.messages.create` to `client.messages.stream` so the closing text types out and tool calls appear the instant they begin rather than when the turn ends.
3. **Replace the corpus with live tools.** Web search, a SQL warehouse, a BI` semantic layer. The event vocabulary doesn't change; only the tool implementations do.
4. **Persist runs** as the event list so an analyst can reopen yesterday's analysis and scrub through it.
