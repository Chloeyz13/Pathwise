"""
Live provenance server.

Runs a Claude tool-use loop and streams a provenance graph to the browser over a
WebSocket as the model works. Every tool result gets a stable id that is handed
back to the model, so when it records a finding or a claim it cites those ids
and the graph edges come from the model's own references.

Event vocabulary sent to the browser (one JSON object per message):
  {"type":"status","text":...}
  {"type":"node","id":..,"kind":question|plan|source|op|finding|caveat|answer,
        "label":..,"sub":..,"col":int,"lane":int,"detail":str,"rows":int?,"rin":int?}
  {"type":"edge","a":id,"b":id}
  {"type":"say","text":..,"evidence":[ids]}
  {"type":"done"} | {"type":"error","text":..}
"""
import asyncio, json, os, re, glob
from pathlib import Path

import duckdb
from anthropic import AsyncAnthropic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

app = FastAPI()
client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY

# ---------------------------------------------------------------- data access
con = duckdb.connect(database=":memory:")
TABLES = {}
for csv in DATA.glob("*.csv"):
    name = csv.stem
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{csv.as_posix()}')")
    TABLES[name] = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]

SOURCES = {}
for md in sorted(DATA.glob("sources/*.md")):
    text = md.read_text(encoding="utf-8")
    title = re.search(r"^#\s*(.+)$", text, re.M)
    url = re.search(r"^url:\s*(\S+)$", text, re.M)
    SOURCES[md.stem] = {
        "title": title.group(1).strip() if title else md.stem,
        "url": url.group(1) if url else "",
        "text": text,
    }

def schema_text():
    out = []
    for name in TABLES:
        cols = con.execute(f"DESCRIBE {name}").fetchall()
        out.append(f"table {name} ({TABLES[name]} rows): " + ", ".join(f"{c[0]} {c[1]}" for c in cols))
    return "\n".join(out)

# ---------------------------------------------------------------- tools
TOOLS = [
    {"name": "plan",
     "description": "Declare the sub-tasks you will carry out, in order. Call this first, once.",
     "input_schema": {"type": "object", "properties": {
         "subtasks": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
         "required": ["subtasks"]}},
    {"name": "query_table",
     "description": "Run a read-only SQL query (DuckDB dialect) against the available tables. Returns rows and a result id.",
     "input_schema": {"type": "object", "properties": {
         "sql": {"type": "string"},
         "purpose": {"type": "string", "description": "Short label for what this query does, e.g. 'Rank states by 2023 rate'"},
         "subtask": {"type": "integer", "description": "1-based index of the sub-task this serves"}},
         "required": ["sql", "purpose", "subtask"]}},
    {"name": "search_sources",
     "description": "Keyword search over the document library. Returns matching passages, each with a passage id you can cite.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "subtask": {"type": "integer"}},
         "required": ["query", "subtask"]}},
    {"name": "record_finding",
     "description": "Record one intermediate finding and the result/passage ids it was derived from. Use before writing claims.",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"},
         "from_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
         "kind": {"type": "string", "enum": ["finding", "caveat"], "default": "finding"}},
         "required": ["text", "from_ids"]}},
    {"name": "record_claim",
     "description": "Emit one sentence of the final answer, with the finding ids that support it. The answer the analyst reads is made only of these sentences, in order.",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"},
         "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
         "required": ["text", "evidence_ids"]}},
]

SYSTEM = f"""You are a data analyst's assistant. You answer questions ONLY from the tools provided; never from memory.
Work in this order:
1. Call `plan` once with your sub-tasks.
2. Use `query_table` and `search_sources` to gather evidence. Every result you get back has an id.
3. Call `record_finding` for each intermediate result worth keeping, citing the ids it came from.
   Use kind="caveat" for interpretation choices or limitations (e.g. how you read an ambiguous word).
4. Write the final answer as a sequence of `record_claim` calls, each citing finding ids. Keep claims to one sentence.
5. Finish with a one-line plain-text closing.
Available tables:
{schema_text()}
Document library: {', '.join(SOURCES)}.
Be honest about data limits: if a table is partial, say so in a caveat."""

# ---------------------------------------------------------------- agent run
class Run:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.n = 0
        self.lane_of = {}       # node id -> lane
        self.lanes_used = 0
        self.sources_seen = {}  # table/doc name -> node id

    async def emit(self, **ev):
        await self.ws.send_json(ev)

    def new_id(self, prefix):
        self.n += 1
        return f"{prefix}{self.n}"

    def lane_for_subtask(self, subtask):
        key = f"task{subtask}"
        if key not in self.lane_of:
            self.lane_of[key] = self.lanes_used; self.lanes_used += 1
        return self.lane_of[key]

    async def node(self, id, kind, label, col, lane, detail="", sub="", rows=None, rin=None):
        self.lane_of[id] = lane
        await self.emit(type="node", id=id, kind=kind, label=label, sub=sub, col=col, lane=lane,
                        detail=detail, rows=rows, rin=rin)

    async def edge(self, a, b):
        await self.emit(type="edge", a=a, b=b)

    # ---- tool implementations; each returns (text for the model, ids it minted)
    async def t_plan(self, inp):
        await self.node("plan", "plan", f"Plan: {len(inp['subtasks'])} sub-tasks", 1, 0,
                        detail="\n".join(f"{i+1}. {s}" for i, s in enumerate(inp["subtasks"])))
        await self.edge("q", "plan")
        for i, s in enumerate(inp["subtasks"], 1):
            tid = f"task{i}"
            await self.node(tid, "plan", f"Task {i} · {s[:40]}", 1.15, self.lane_for_subtask(i), detail=s)
            await self.edge("plan", tid)
        return "Plan recorded. Sub-task ids: " + ", ".join(f"task{i}" for i in range(1, len(inp["subtasks"]) + 1))

    async def t_query(self, inp):
        sql, lane = inp["sql"], self.lane_for_subtask(inp.get("subtask", 1))
        await self.emit(type="status", text=f"Querying: {inp['purpose']}")
        tables = [t for t in TABLES if re.search(rf"\b{t}\b", sql)]
        rin = sum(TABLES[t] for t in tables) or None
        # source node for each table, once
        for t in tables:
            if t not in self.sources_seen:
                sid = f"src_{t}"; self.sources_seen[t] = sid
                await self.node(sid, "source", t, 2, lane, sub=f"{TABLES[t]} rows", rows=TABLES[t],
                                detail=f"table {t}\n{TABLES[t]} rows\n\n" + "\n".join(
                                    f"{c[0]}: {c[1]}" for c in con.execute(f'DESCRIBE {t}').fetchall()))
                await self.edge(f"task{inp.get('subtask', 1)}", sid)
        try:
            if not re.match(r"^\s*(select|with)\b", sql, re.I):
                raise ValueError("read-only: only SELECT/WITH allowed")
            cur = con.execute(sql); rows = cur.fetchall(); cols = [d[0] for d in cur.description]
        except Exception as e:
            oid = self.new_id("q")
            await self.node(oid, "caveat", f"Query failed: {inp['purpose'][:30]}", 3, lane, detail=f"{sql}\n\nERROR: {e}")
            return f"[{oid}] query error: {e}"
        oid = self.new_id("q")
        preview = [dict(zip(cols, r)) for r in rows[:25]]
        await self.node(oid, "op", inp["purpose"], 3, lane, sub=f"{rin} → {len(rows)} rows", rows=len(rows), rin=rin,
                        detail=sql + "\n\n" + json.dumps(preview, indent=1, default=str))
        for t in tables: await self.edge(self.sources_seen[t], oid)
        return f"[{oid}] {len(rows)} rows. First rows:\n" + json.dumps(preview, default=str)

    async def t_search(self, inp):
        q, lane = inp["query"], self.lane_for_subtask(inp.get("subtask", 1))
        await self.emit(type="status", text=f"Searching sources: {q}")
        terms = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 3]
        hits = []
        for name, doc in SOURCES.items():
            paras = [p.strip() for p in doc["text"].split("\n\n") if p.strip() and not p.startswith(("#", "url:"))]
            for p in paras:
                score = sum(p.lower().count(t) for t in terms)
                if score: hits.append((score, name, p))
        hits.sort(reverse=True)
        oid = self.new_id("s")
        out = []
        for score, name, p in hits[:5]:
            if name not in self.sources_seen:
                sid = f"src_{name}"; self.sources_seen[name] = sid
                # each document gets its own lane under the sub-task's lane so parallel sources fan out
                dl = self.lanes_used; self.lanes_used += 1
                await self.node(sid, "source", SOURCES[name]["title"][:60], 2, dl, sub=name,
                                detail=SOURCES[name]["title"] + "\n" + SOURCES[name]["url"])
                await self.edge(f"task{inp.get('subtask', 1)}", sid)
            pid = self.new_id("p")
            self.lane_of[pid] = self.lane_of[self.sources_seen[name]]
            out.append(f"[{pid}] ({name}) {p}")
            await self.node(pid, "op", f"Passage from {name}", 3, self.lane_of[pid], sub=f"match ×{score}", detail=p)
            await self.edge(self.sources_seen[name], pid)
        if not out:
            return f"[{oid}] no passages matched '{q}'"
        return "\n\n".join(out)

    async def t_finding(self, inp):
        fid = self.new_id("f")
        kind = inp.get("kind", "finding")
        srcs = [i for i in inp["from_ids"] if i in self.lane_of]
        lane = self.lane_of[srcs[0]] if srcs else 0
        await self.node(fid, kind, inp["text"][:48], 4.4, lane, detail=inp["text"])
        for s in srcs: await self.edge(s, fid)
        bad = [i for i in inp["from_ids"] if i not in self.lane_of]
        return f"[{fid}] recorded" + (f" (unknown ids ignored: {bad})" if bad else "")

    async def t_claim(self, inp):
        if "ans" not in self.lane_of:
            await self.node("ans", "answer", "Answer", 6.3, 0, detail="Built from the claims below.")
        ev = [i for i in inp["evidence_ids"] if i in self.lane_of]
        for e in ev: await self.edge(e, "ans")
        await self.emit(type="say", text=inp["text"], evidence=ev + ["ans"])
        bad = [i for i in inp["evidence_ids"] if i not in self.lane_of]
        return "claim recorded" + (f" (unknown ids ignored: {bad})" if bad else "")

    async def run(self, question: str):
        await self.node("q", "question", "Question", 0, 0, detail=question)
        await self.emit(type="status", text="Reading the question")
        messages = [{"role": "user", "content": question}]
        handlers = {"plan": self.t_plan, "query_table": self.t_query, "search_sources": self.t_search,
                    "record_finding": self.t_finding, "record_claim": self.t_claim}
        for _ in range(40):  # hard cap on turns
            resp = await client.messages.create(model=MODEL, max_tokens=2048, system=SYSTEM,
                                                tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                closing = "".join(b.text for b in resp.content if b.type == "text").strip()
                if closing: await self.emit(type="say", text=closing, evidence=["ans"] if "ans" in self.lane_of else [])
                break
            results = []
            for block in resp.content:
                if block.type != "tool_use": continue
                try:
                    text = await handlers[block.name](block.input)
                except Exception as e:
                    text = f"tool error: {e}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
            messages.append({"role": "user", "content": results})
        await self.emit(type="status", text="Done", done=True)
        await self.emit(type="done")

# ---------------------------------------------------------------- transport
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ask":
                try:
                    await Run(ws).run(msg["question"])
                except Exception as e:
                    await ws.send_json({"type": "error", "text": str(e)})
    except WebSocketDisconnect:
        pass

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
