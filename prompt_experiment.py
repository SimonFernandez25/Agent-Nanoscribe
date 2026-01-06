# ======================================================
# File: prompt_experiment.py
# Experiment comparing different system prompt variations
# ======================================================

import os, time, json, re
from pathlib import Path
from typing import TypedDict, Optional, List, Dict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt

from gwl_validator import validate_gwl_files, validate_gwl_content, ValidationReport

# ======================================================
# CONFIGURATION
# ======================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR / "Docs"
EXPERIMENT_DIR = SCRIPT_DIR / "system_prompt_test"
CYLINDER_EXAMPLE_DIR = SCRIPT_DIR / "Outputs" / "Cylinder Example"
API_FILE = DOCS_DIR / "API.txt"

OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

EXPERIMENT_PROMPT = "Generate a 20x20 array of square pyramid meta atoms, with base 10 um, height 10um. Spaced by 20 um."

LANGIUM_RULES = """
GWL GRAMMAR RULES:
- VarDecl: var|local $name = value
- Assignment: set $name = value  
- ForLoop: for $var = start to end [step val] ... end
- IfBlock: if condition ... [else ...] end
- IncludeStmt: include filepath
- CoordinateLine: expr expr expr [expr]
- CmdStmt: CommandName [args]
CRITICAL: Every variable used in loops must be declared with 'local' FIRST.
"""

# ======================================================
# PROMPT VARIATIONS
# ======================================================

# Config 1: FULL PROMPTS (current)
META_ATOM_FULL = """You are a DescribeX GWL generator for the Nanoscribe Quantum X.
Produce ONLY valid GWL code for a single meta-atom using official syntax.

CRITICAL RULES:
- Every variable used in a loop or expression must be declared first with 'local'.
- Center geometry at origin (X=0, Y=0), start at ZOffset 0, build upward.
- Use consistent indentation and proper loop termination ('end').
- Do NOT write -$x inline. Use 'set' first: set $x1 = -$xHalf
- Output only GWL code, no commentary.

CORRECT FORMAT:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
var $objective = 63
local $height = 2.0
local $voxel_lateral = 0.1
local $dz = 0.15

XOffset 0
YOffset 0
ZOffset 0
PowerScaling 0.6
LaserPower 40
ScanSpeed 100
LineNumber 1

local $slice = 0
local $y = 0
for $slice = 0 to 10
    if $slice > 0
        AddZOffset $dz
    end
    for $y = -1 to 1 step 0.1
        set $x1 = -1
        set $x2 = 1
        $x1 $y 0
        $x2 $y 0
        Write
    end
end
ZOffset 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

ARRAY_FULL = """You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.
Build a valid master .GWL file that arranges meta-atom GWLs into an array.

CRITICAL RULES:
- Every variable used in a loop must be declared with 'local' FIRST.
- Use MoveStageX/MoveStageY, Include, etc.
- Center array around origin, ZOffset = 0.
- Output ONLY valid GWL code, no commentary.

CORRECT FORMAT:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
var $objective = 63
XOffset 0
YOffset 0
ZOffset 0
PowerScaling 0.6
LaserPower 40
ScanSpeed 100
FindInterfaceAt 0.5
LineNumber 1

local $rows = 5
local $cols = 5
local $pitch = 5.0
local $xstart = -($pitch * ($cols - 1) / 2.0)
local $ystart = -($pitch * ($rows - 1) / 2.0)

local $r = 0
local $c = 0
local $cx = 0
local $cy = 0

for $r = 0 to $rows - 1
    for $c = 0 to $cols - 1
        set $cx = $xstart + $c * $pitch
        set $cy = $ystart + $r * $pitch
        MoveStageX $cx
        MoveStageY $cy
        Include meta_1.gwl
        MoveStageX -$cx
        MoveStageY -$cy
    end
end
ZOffset 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


# Config 2: MINIMAL (no format example)
META_ATOM_MINIMAL = """You are a DescribeX GWL generator for the Nanoscribe Quantum X.
Produce ONLY valid GWL code for a single meta-atom using official syntax.

CRITICAL RULES:
- Every variable used in a loop or expression must be declared first with 'local'.
- Center geometry at origin (X=0, Y=0), start at ZOffset 0, build upward.
- Use consistent indentation and proper loop termination ('end').
- Do NOT write -$x inline. Use 'set' first: set $x1 = -$xHalf
- Output only GWL code, no commentary."""

ARRAY_MINIMAL = """You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.
Build a valid master .GWL file that arranges meta-atom GWLs into an array.

CRITICAL RULES:
- Every variable used in a loop must be declared with 'local' FIRST.
- Use MoveStageX/MoveStageY, Include, etc.
- Center array around origin, ZOffset = 0.
- Output ONLY valid GWL code, no commentary."""


# Config 3: NO CRITICAL RULES
META_ATOM_NO_RULES = """You are a DescribeX GWL generator for the Nanoscribe Quantum X.
Produce ONLY valid GWL code for a single meta-atom using official syntax.
Output only GWL code, no commentary."""

ARRAY_NO_RULES = """You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.
Build a valid master .GWL file that arranges meta-atom GWLs into an array.
Output ONLY valid GWL code, no commentary."""


GEOMETRY_PROMPT = """You are a geometry planner for Nanoscribe metasurfaces.
Output ONE structured JSON object describing:
- 'array': global layout with {'rows': int, 'cols': int, 'spacing_um': float}
- 'meta_atoms': list of UNIQUE meta-atoms with {'id': int, 'shape': str, dimensions...}
- 'pattern_map': 2D list (rows×cols) of integer IDs referencing meta_atoms.

Rules:
- Assign IDs sequentially starting from 1.
- Each unique meta-atom geometry should appear only once in 'meta_atoms'.
- pattern_map should only contain those numeric IDs.
- If the array is uniform, all entries in pattern_map should be [1].
- Output valid JSON only. No commentary.

Example:
{
  "array": {"rows": 3, "cols": 3, "spacing_um": 5.0},
  "meta_atoms": [
    {"id": 1, "shape": "cylinder", "height_um": 1.0, "radius_um": 1.0}
  ],
  "pattern_map": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
}"""

DEBUG_PROMPT = """You are a GWL code debugger. Fix the GWL code based on the validation errors.
Output ONLY the corrected GWL code. No explanations."""


# ======================================================
# STATE
# ======================================================
class AgentState(TypedDict, total=False):
    user_prompt: str
    project_dir: Path
    docs_context: str
    geometry_json: str
    meta_gwls: List[str]
    master_gwl: str
    validation_report: Dict
    token_stats: Dict
    debug_counts: Dict
    meta_prompt: str
    array_prompt: str


# ======================================================
# HELPERS
# ======================================================
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def save_json(path: Path, filename: str, content, metadata: Dict = None):
    output = {"timestamp": datetime.now().isoformat(), "content": content, "metadata": metadata or {}}
    with open(path / filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output

def clean_gwl(code: str) -> str:
    if code.startswith("```"):
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
    return code.strip()

def format_errors(validation) -> str:
    errors = ""
    for v in validation.validations:
        if not v.valid:
            errors += f"\nFile: {v.file}\n"
            for err in v.errors:
                errors += f"  Line {err.line}: {err.message}\n"
    return errors

def debug_code(code: str, errors: str, llm, stats: Dict) -> tuple:
    human = f"GWL CODE:\n{code}\n\nERRORS:\n{errors}\n\nGRAMMAR:\n{LANGIUM_RULES}\n\nFix errors, output only GWL."
    msgs = [SystemMessage(content=DEBUG_PROMPT), HumanMessage(content=human)]
    stats["input"] += count_tokens(DEBUG_PROMPT) + count_tokens(human)
    result = llm.invoke(msgs)
    fixed = clean_gwl(result.content)
    stats["output"] += count_tokens(fixed)
    stats["calls"] += 1
    return fixed, stats


# ======================================================
# CONTEXT LOADER
# ======================================================
def load_context():
    """Load Cylinder example + langium (best from previous experiment)"""
    context = ""
    meta = CYLINDER_EXAMPLE_DIR / "meta_1.gwl"
    array = CYLINDER_EXAMPLE_DIR / "main_array.gwl"
    if meta.exists():
        context += f"\n### EXAMPLE META-ATOM GWL ###\n{meta.read_text(encoding='utf-8')}\n"
    if array.exists():
        context += f"\n### EXAMPLE ARRAY GWL ###\n{array.read_text(encoding='utf-8')}\n"
    context += f"\n{LANGIUM_RULES}\n"
    return context


# ======================================================
# NODES
# ======================================================
def node_geometry(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = state["user_prompt"]
    docs = state.get("docs_context", "")
    human = f"{prompt}\n\nReference:\n{docs}"
    msgs = [SystemMessage(content=GEOMETRY_PROMPT), HumanMessage(content=human)]
    
    stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0})
    stats["input"] += count_tokens(GEOMETRY_PROMPT) + count_tokens(human)
    
    result = llm.invoke(msgs)
    geom = clean_gwl(result.content)
    stats["output"] += count_tokens(geom)
    stats["calls"] += 1
    
    save_json(state["project_dir"], "geometry.json", json.loads(geom))
    state["geometry_json"] = geom
    state["token_stats"] = stats
    state["debug_counts"] = {}
    return state


def node_meta(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    geom = json.loads(state["geometry_json"])
    docs = state.get("docs_context", "")
    meta_prompt = state["meta_prompt"]
    
    stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0})
    debug_counts = state.get("debug_counts", {})
    gwl_files = []
    
    for atom in geom.get("meta_atoms", []):
        name = f"meta_{atom['id']}"
        human = f"META-ATOM: {json.dumps(atom)}\n\nREFERENCE:\n{docs}\n\nGenerate GWL."
        msgs = [SystemMessage(content=meta_prompt), HumanMessage(content=human)]
        
        stats["input"] += count_tokens(meta_prompt) + count_tokens(human)
        result = llm.invoke(msgs)
        code = clean_gwl(result.content)
        stats["output"] += count_tokens(code)
        stats["calls"] += 1
        
        # Debug loop (max 2)
        debug_counts[name] = 0
        for i in range(2):
            val = validate_gwl_content({name: code})
            if val.valid:
                break
            debug_counts[name] += 1
            code, stats = debug_code(code, format_errors(val), llm, stats)
        
        path = state["project_dir"] / f"{name}.gwl"
        path.write_text(code, encoding="utf-8")
        gwl_files.append(str(path))
    
    state["meta_gwls"] = gwl_files
    state["token_stats"] = stats
    state["debug_counts"] = debug_counts
    return state


def node_array(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    geom = json.loads(state["geometry_json"])
    docs = state.get("docs_context", "")
    array_prompt = state["array_prompt"]
    
    atoms = geom.get("meta_atoms", [])
    files = "\n".join([f"meta_{a['id']}.gwl" for a in atoms])
    
    human = f"ARRAY: {json.dumps(geom.get('array', {}))}\nFILES: {files}\nPATTERN: {json.dumps(geom.get('pattern_map', []))}\n\nREFERENCE:\n{docs}\n\nGenerate array GWL."
    msgs = [SystemMessage(content=array_prompt), HumanMessage(content=human)]
    
    stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0})
    stats["input"] += count_tokens(array_prompt) + count_tokens(human)
    
    result = llm.invoke(msgs)
    code = clean_gwl(result.content)
    stats["output"] += count_tokens(code)
    stats["calls"] += 1
    
    debug_counts = state.get("debug_counts", {})
    debug_counts["main_array"] = 0
    
    for i in range(2):
        val = validate_gwl_content({"main_array.gwl": code})
        if val.valid:
            break
        debug_counts["main_array"] += 1
        code, stats = debug_code(code, format_errors(val), llm, stats)
    
    path = state["project_dir"] / "main_array.gwl"
    path.write_text(code, encoding="utf-8")
    
    state["master_gwl"] = str(path)
    state["token_stats"] = stats
    state["debug_counts"] = debug_counts
    return state


def node_validate(state: AgentState) -> AgentState:
    files = state.get("meta_gwls", []).copy()
    if state.get("master_gwl"):
        files.append(state["master_gwl"])
    report = validate_gwl_files(files)
    state["validation_report"] = report.to_dict()
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("geometry", node_geometry)
    g.add_node("meta", node_meta)
    g.add_node("array", node_array)
    g.add_node("validate", node_validate)
    g.set_entry_point("geometry")
    g.add_edge("geometry", "meta")
    g.add_edge("meta", "array")
    g.add_edge("array", "validate")
    g.add_edge("validate", END)
    return g.compile()


# ======================================================
# RUN
# ======================================================
def run_experiment(config_name: str, meta_prompt: str, array_prompt: str, run_num: int, exp_dir: Path) -> Dict:
    print(f"\n{'='*50}\n{config_name} | Run {run_num}\n{'='*50}")
    
    run_dir = exp_dir / config_name.lower().replace(" ", "_") / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    docs = load_context()
    graph = build_graph()
    
    start = time.time()
    result = graph.invoke({
        "user_prompt": EXPERIMENT_PROMPT,
        "project_dir": run_dir,
        "docs_context": docs,
        "meta_prompt": meta_prompt,
        "array_prompt": array_prompt,
        "token_stats": {"input": 0, "output": 0, "calls": 0}
    })
    elapsed = time.time() - start
    
    stats = result.get("token_stats", {})
    val = result.get("validation_report", {})
    debugs = result.get("debug_counts", {})
    
    errors = sum(len(v.get("errors", [])) for v in val.get("validations", []))
    total_debugs = sum(debugs.values())
    
    metrics = {
        "config": config_name,
        "run": run_num,
        "time_sec": round(elapsed, 2),
        "tokens": stats.get("input", 0) + stats.get("output", 0),
        "input_tokens": stats.get("input", 0),
        "output_tokens": stats.get("output", 0),
        "calls": stats.get("calls", 0),
        "debugs": total_debugs,
        "errors": errors,
        "valid": val.get("valid", False)
    }
    
    save_json(run_dir, "metrics.json", metrics)
    status = "[OK]" if metrics["valid"] else f"[ERR:{errors}]"
    print(f"  {status} | {elapsed:.1f}s | {metrics['tokens']} tok | {total_debugs} debugs")
    
    return metrics


def generate_plots(results: List[Dict], exp_dir: Path):
    df = pd.DataFrame(results)
    plots_dir = exp_dir / "analysis"
    plots_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Time
    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby('config')['time_sec'].agg(['mean', 'std']).reset_index()
    ax.bar(g['config'], g['mean'], yerr=g['std'], capsize=5)
    ax.set_ylabel('Time (s)')
    ax.set_title('Execution Time by Prompt Config')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'time.png', dpi=150)
    plt.close()
    
    # Tokens
    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby('config')['tokens'].agg(['mean', 'std']).reset_index()
    ax.bar(g['config'], g['mean'], yerr=g['std'], capsize=5)
    ax.set_ylabel('Total Tokens')
    ax.set_title('Token Usage by Prompt Config')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'tokens.png', dpi=150)
    plt.close()
    
    # Errors
    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby('config')['errors'].agg(['mean', 'std']).reset_index()
    ax.bar(g['config'], g['mean'], yerr=g['std'], capsize=5)
    ax.set_ylabel('Final Errors')
    ax.set_title('Validation Errors by Prompt Config')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'errors.png', dpi=150)
    plt.close()
    
    # Debugs
    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby('config')['debugs'].agg(['mean', 'std']).reset_index()
    ax.bar(g['config'], g['mean'], yerr=g['std'], capsize=5)
    ax.set_ylabel('Debug Attempts')
    ax.set_title('Debug Attempts by Prompt Config')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'debugs.png', dpi=150)
    plt.close()
    
    # Success rate
    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby('config')['valid'].mean() * 100
    ax.bar(g.index, g.values)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Validation Success Rate')
    ax.set_ylim(0, 110)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'success.png', dpi=150)
    plt.close()
    
    df.to_csv(plots_dir / 'results.csv', index=False)
    summary = df.groupby('config').agg({
        'time_sec': ['mean', 'std'],
        'tokens': ['mean'],
        'debugs': ['mean'],
        'errors': ['mean'],
        'valid': ['sum', 'count']
    }).round(2)
    summary.to_csv(plots_dir / 'summary.csv')
    print(f"\nSaved to: {plots_dir}")
    print(summary.to_string())


def main():
    print("\n" + "="*60)
    print("SYSTEM PROMPT EXPERIMENT")
    print("="*60)
    
    exp_dir = EXPERIMENT_DIR
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    configs = [
        ("Full Prompt", META_ATOM_FULL, ARRAY_FULL),
        ("Minimal (No Format)", META_ATOM_MINIMAL, ARRAY_MINIMAL),
        ("No Critical Rules", META_ATOM_NO_RULES, ARRAY_NO_RULES)
    ]
    
    results = []
    for name, meta_p, arr_p in configs:
        for run in range(1, 4):
            try:
                m = run_experiment(name, meta_p, arr_p, run, exp_dir)
                results.append(m)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    generate_plots(results, exp_dir)
    save_json(exp_dir, "all_results.json", results)
    
    print("\n" + "="*60)
    print(f"COMPLETE: {exp_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
