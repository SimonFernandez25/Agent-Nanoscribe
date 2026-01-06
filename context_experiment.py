# ======================================================
# File: context_experiment.py
# Experiment comparing different context configurations
# Uses EXACT prompts from simple_gwl_agent.py
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

# Import the validator
from gwl_validator import validate_gwl_files, validate_gwl_content, ValidationReport

# ======================================================
# CONFIGURATION
# ======================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR / "Docs"
OUTPUTS_BASE_DIR = SCRIPT_DIR / "Outputs"
EXPERIMENT_DIR = SCRIPT_DIR / "context_experiment_results"
CYLINDER_EXAMPLE_DIR = OUTPUTS_BASE_DIR / "Cylinder Example"
LANGIUM_FILE = SCRIPT_DIR / "gwl (2).langium"
API_FILE = DOCS_DIR / "API.txt"

OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Experiment prompt
EXPERIMENT_PROMPT = "Generate a 20x20 array of square pyramid meta atoms, with base 10 um, height 10um. Spaced by 20 um."

# Langium grammar summary for debug
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
Comments start with %
Variable names must start with $
"""

# ======================================================
# EXACT PROMPTS FROM simple_gwl_agent.py - DO NOT MODIFY
# ======================================================
GEOMETRY_SYSTEM_PROMPT = """You are a geometry planner for Nanoscribe metasurfaces.
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

META_ATOM_SYSTEM_PROMPT = """You are a DescribeX GWL generator for the Nanoscribe Quantum X.
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

ARRAY_SYSTEM_PROMPT = """You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.
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

DEBUG_SYSTEM_PROMPT = """You are a GWL code debugger. Fix the GWL code based on the validation errors.

You will receive:
1. The current GWL code
2. Validation errors with exact line numbers
3. GWL grammar rules

Output ONLY the corrected GWL code. No explanations."""


# ======================================================
# STATE
# ======================================================
class AgentState(TypedDict, total=False):
    user_prompt: str
    project_name: str
    project_dir: Path
    docs_context: str
    geometry_json: str
    meta_gwls: List[str]
    master_gwl: str
    validation_report: Dict
    token_stats: Dict
    debug_counts: Dict  # Track debug iterations per file


# ======================================================
# CONTEXT CONFIGURATIONS
# ======================================================
def load_context_config_1():
    """Config 1: Just GWL and Stitching manual (PDFs)"""
    context = ""
    
    pdf_files = [
        DOCS_DIR / "GWL Command Guide DescribeX.pdf",
        DOCS_DIR / "GWL Stiching Guide.pdf"
    ]
    
    for pdf_path in pdf_files:
        if pdf_path.exists():
            try:
                import fitz
                text = ""
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        text += page.get_text()
                context += f"\n### {pdf_path.name} ###\n{text[:6000]}\n"
                print(f"  Loaded {pdf_path.name} ({len(text)} chars)")
            except ImportError:
                print(f"  [WARN] PyMuPDF not available for {pdf_path.name}")
    
    return context, "pdfs_only"


def load_context_config_2():
    """Config 2: All original docs (TXT files)"""
    context = ""
    
    txt_files = [
        DOCS_DIR / "Lines.txt",
        DOCS_DIR / "Transcript.txt",
        DOCS_DIR / "Woodpile.txt"
    ]
    
    for txt_path in txt_files:
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="ignore")[:6000]
            context += f"\n### {txt_path.name} ###\n{text}\n"
            print(f"  Loaded {txt_path.name} ({len(text)} chars)")
    
    return context, "all_txt_docs"


def load_context_config_3():
    """Config 3: Cylinder example + langium grammar"""
    context = ""
    
    meta_gwl = CYLINDER_EXAMPLE_DIR / "meta_1.gwl"
    array_gwl = CYLINDER_EXAMPLE_DIR / "main_array.gwl"
    
    if meta_gwl.exists():
        context += f"\n### EXAMPLE META-ATOM GWL ###\n{meta_gwl.read_text(encoding='utf-8')}\n"
        print(f"  Loaded meta_1.gwl example")
    
    if array_gwl.exists():
        context += f"\n### EXAMPLE ARRAY GWL ###\n{array_gwl.read_text(encoding='utf-8')}\n"
        print(f"  Loaded main_array.gwl example")
    
    context += f"\n{LANGIUM_RULES}\n"
    print(f"  Loaded langium grammar rules")
    
    return context, "cylinder_example_langium"


# ======================================================
# HELPERS
# ======================================================
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def save_json(project_dir: Path, filename: str, content, metadata: Dict = None):
    output = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "metadata": metadata or {}
    }
    json_path = project_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output


def clean_gwl(code: str) -> str:
    """Remove markdown wrappers from GWL code."""
    if code.startswith("```"):
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
    return code.strip()


def format_errors_for_debug(validation_result) -> str:
    """Format validation errors for debug prompt."""
    errors_text = ""
    for v in validation_result.validations:
        if not v.valid:
            errors_text += f"\nFile: {v.file}\n"
            for err in v.errors:
                errors_text += f"  Line {err.line}: {err.message}\n"
    return errors_text


# ======================================================
# DEBUG FUNCTION
# ======================================================
def debug_gwl_code(gwl_code: str, errors: str, llm, token_stats: Dict) -> tuple:
    """Attempt to fix GWL code based on validation errors. Returns (fixed_code, tokens_used)."""
    
    human = f"""GWL CODE WITH ERRORS:
{gwl_code}

VALIDATION ERRORS:
{errors}

GWL GRAMMAR RULES:
{LANGIUM_RULES}

Fix the errors and output only corrected GWL code."""

    messages = [
        SystemMessage(content=DEBUG_SYSTEM_PROMPT),
        HumanMessage(content=human)
    ]
    
    input_tokens = count_tokens(DEBUG_SYSTEM_PROMPT) + count_tokens(human)
    
    result = llm.invoke(messages)
    fixed_code = clean_gwl(result.content)
    
    output_tokens = count_tokens(fixed_code)
    
    token_stats["input"] += input_tokens
    token_stats["output"] += output_tokens
    token_stats["calls"] += 1
    
    return fixed_code, token_stats


# ======================================================
# AGENT NODES
# ======================================================
def node_geometry(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    prompt = state["user_prompt"]
    project_dir = state["project_dir"]
    docs_text = state.get("docs_context", "")
    
    human = f"{prompt}\n\nReference:\n{docs_text}"
    
    messages = [
        SystemMessage(content=GEOMETRY_SYSTEM_PROMPT),
        HumanMessage(content=human)
    ]
    
    input_tokens = count_tokens(GEOMETRY_SYSTEM_PROMPT) + count_tokens(human)
    
    start_time = time.time()
    result = llm.invoke(messages)
    elapsed = time.time() - start_time
    
    geometry_json = clean_gwl(result.content)
    output_tokens = count_tokens(geometry_json)
    
    save_json(project_dir, "geometry_output.json", json.loads(geometry_json))
    
    token_stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0, "debug_calls": 0})
    token_stats["input"] += input_tokens
    token_stats["output"] += output_tokens
    token_stats["calls"] += 1
    
    print(f"  Geometry: {elapsed:.2f}s")
    
    state["geometry_json"] = geometry_json
    state["token_stats"] = token_stats
    state["debug_counts"] = {}
    return state


def node_meta_gwls(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    geom_data = json.loads(state["geometry_json"])
    project_dir = state["project_dir"]
    docs_text = state.get("docs_context", "")
    
    meta_atoms = geom_data.get("meta_atoms", [])
    if not meta_atoms:
        return state
    
    gwl_files = []
    token_stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0, "debug_calls": 0})
    debug_counts = state.get("debug_counts", {})
    
    for atom in meta_atoms:
        atom_id = atom.get("id", 1)
        atom_name = f"meta_{atom_id}"
        atom_json = json.dumps(atom, indent=2)
        
        human = f"""META-ATOM SPECIFICATION:
{atom_json}

REFERENCE DOCS:
{docs_text}

Generate valid GWL code for this meta-atom."""

        messages = [
            SystemMessage(content=META_ATOM_SYSTEM_PROMPT),
            HumanMessage(content=human)
        ]
        
        input_tokens = count_tokens(META_ATOM_SYSTEM_PROMPT) + count_tokens(human)
        
        start_time = time.time()
        result = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        gwl_code = clean_gwl(result.content)
        output_tokens = count_tokens(gwl_code)
        
        token_stats["input"] += input_tokens
        token_stats["output"] += output_tokens
        token_stats["calls"] += 1
        
        # Validate and debug loop (max 2 iterations)
        debug_counts[atom_name] = 0
        for debug_iter in range(2):
            validation = validate_gwl_content({atom_name: gwl_code})
            if validation.valid:
                break
            
            # Debug attempt
            debug_counts[atom_name] += 1
            token_stats["debug_calls"] = token_stats.get("debug_calls", 0) + 1
            errors_text = format_errors_for_debug(validation)
            print(f"    Debug {atom_name} (attempt {debug_iter + 1}): {len(validation.validations[0].errors)} errors")
            
            gwl_code, token_stats = debug_gwl_code(gwl_code, errors_text, llm, token_stats)
        
        # Save file
        fpath = project_dir / f"{atom_name}.gwl"
        fpath.write_text(gwl_code, encoding="utf-8")
        gwl_files.append(str(fpath))
        
        print(f"  {atom_name}: {elapsed:.2f}s, debug attempts: {debug_counts[atom_name]}")
    
    state["meta_gwls"] = gwl_files
    state["token_stats"] = token_stats
    state["debug_counts"] = debug_counts
    return state


def node_master_gwl(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    project_dir = state["project_dir"]
    geom_data = json.loads(state["geometry_json"])
    user_prompt = state.get("user_prompt", "")
    docs_text = state.get("docs_context", "")
    
    array_info = geom_data.get("array", {})
    pattern_map = geom_data.get("pattern_map", [])
    meta_atoms = geom_data.get("meta_atoms", [])
    include_files = "\n".join([f"meta_{a['id']}.gwl" for a in meta_atoms])
    
    human = f"""USER PROMPT:
{user_prompt}

GEOMETRY JSON:
{json.dumps(geom_data, indent=2)}

AVAILABLE META-ATOM FILES:
{include_files}

PATTERN MAP:
{json.dumps(pattern_map, indent=2)}

REFERENCE DOCS:
{docs_text}

Generate valid GWL array code using the meta-atom files according to pattern_map."""

    messages = [
        SystemMessage(content=ARRAY_SYSTEM_PROMPT),
        HumanMessage(content=human)
    ]
    
    input_tokens = count_tokens(ARRAY_SYSTEM_PROMPT) + count_tokens(human)
    
    start_time = time.time()
    result = llm.invoke(messages)
    elapsed = time.time() - start_time
    
    gwl_code = clean_gwl(result.content)
    output_tokens = count_tokens(gwl_code)
    
    token_stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0, "debug_calls": 0})
    token_stats["input"] += input_tokens
    token_stats["output"] += output_tokens
    token_stats["calls"] += 1
    
    debug_counts = state.get("debug_counts", {})
    debug_counts["main_array"] = 0
    
    # Validate and debug loop (max 2 iterations)
    for debug_iter in range(2):
        validation = validate_gwl_content({"main_array.gwl": gwl_code})
        if validation.valid:
            break
        
        debug_counts["main_array"] += 1
        token_stats["debug_calls"] = token_stats.get("debug_calls", 0) + 1
        errors_text = format_errors_for_debug(validation)
        print(f"    Debug main_array (attempt {debug_iter + 1}): {len(validation.validations[0].errors)} errors")
        
        gwl_code, token_stats = debug_gwl_code(gwl_code, errors_text, llm, token_stats)
    
    fpath = project_dir / "main_array.gwl"
    fpath.write_text(gwl_code, encoding="utf-8")
    
    print(f"  main_array: {elapsed:.2f}s, debug attempts: {debug_counts['main_array']}")
    
    state["master_gwl"] = str(fpath)
    state["token_stats"] = token_stats
    state["debug_counts"] = debug_counts
    return state


def node_validate(state: AgentState) -> AgentState:
    """Final validation of all files."""
    gwl_files = state.get("meta_gwls", []).copy()
    if state.get("master_gwl"):
        gwl_files.append(state["master_gwl"])
    
    report = validate_gwl_files(gwl_files)
    
    # Print errors
    for v in report.validations:
        if not v.valid:
            print(f"  ERRORS in {v.file}:")
            for err in v.errors:
                print(f"    L{err.line}: {err.message}")
    
    state["validation_report"] = report.to_dict()
    return state


# ======================================================
# BUILD GRAPH
# ======================================================
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("geometry", node_geometry)
    builder.add_node("meta_gwls", node_meta_gwls)
    builder.add_node("master_gwl", node_master_gwl)
    builder.add_node("validate", node_validate)
    
    builder.set_entry_point("geometry")
    builder.add_edge("geometry", "meta_gwls")
    builder.add_edge("meta_gwls", "master_gwl")
    builder.add_edge("master_gwl", "validate")
    builder.add_edge("validate", END)
    
    return builder.compile()


# ======================================================
# RUN SINGLE EXPERIMENT
# ======================================================
def run_single_experiment(config_name: str, config_loader, run_number: int, experiment_dir: Path) -> Dict:
    print(f"\n{'='*60}")
    print(f"Config: {config_name} | Run: {run_number}")
    print('='*60)
    
    print("Loading context...")
    docs_text, config_id = config_loader()
    context_tokens = count_tokens(docs_text)
    print(f"Context tokens: {context_tokens}")
    
    run_dir = experiment_dir / config_id / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    graph = build_graph()
    
    start_time = time.time()
    result = graph.invoke({
        "user_prompt": EXPERIMENT_PROMPT,
        "project_name": f"{config_id}_run{run_number}",
        "project_dir": run_dir,
        "docs_context": docs_text,
        "token_stats": {"input": 0, "output": 0, "calls": 0, "debug_calls": 0}
    })
    total_time = time.time() - start_time
    
    token_stats = result.get("token_stats", {})
    validation = result.get("validation_report", {})
    debug_counts = result.get("debug_counts", {})
    
    total_errors = sum(len(v.get("errors", [])) for v in validation.get("validations", []))
    total_debug_attempts = sum(debug_counts.values())
    
    metrics = {
        "config": config_id,
        "run": run_number,
        "total_time_sec": round(total_time, 2),
        "input_tokens": token_stats.get("input", 0),
        "output_tokens": token_stats.get("output", 0),
        "total_tokens": token_stats.get("input", 0) + token_stats.get("output", 0),
        "context_tokens": context_tokens,
        "llm_calls": token_stats.get("calls", 0),
        "debug_calls": token_stats.get("debug_calls", 0),
        "debug_counts": debug_counts,
        "valid": validation.get("valid", False),
        "total_errors": total_errors,
        "total_debug_attempts": total_debug_attempts
    }
    
    save_json(run_dir, "run_metrics.json", metrics)
    
    status = "[OK]" if metrics["valid"] else f"[ERRORS: {total_errors}]"
    print(f"Result: {status} | Time: {metrics['total_time_sec']}s | Tokens: {metrics['total_tokens']} | Debugs: {total_debug_attempts}")
    
    return metrics


# ======================================================
# ANALYSIS
# ======================================================
def generate_analysis(all_results: List[Dict], experiment_dir: Path):
    print("\n" + "="*60)
    print("GENERATING ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    plots_dir = experiment_dir / "analysis"
    plots_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    df_grouped = df.groupby('config')['total_time_sec'].agg(['mean', 'std']).reset_index()
    ax.bar(df_grouped['config'], df_grouped['mean'], yerr=df_grouped['std'], capsize=5)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time by Context Configuration')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'time_comparison.png', dpi=150)
    plt.close()
    
    # 2. Token Usage
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df_grouped))
    width = 0.35
    ax.bar([i - width/2 for i in x], df.groupby('config')['input_tokens'].mean(), width, label='Input')
    ax.bar([i + width/2 for i in x], df.groupby('config')['output_tokens'].mean(), width, label='Output')
    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped['config'], rotation=15)
    ax.set_ylabel('Tokens')
    ax.set_title('Token Usage by Configuration')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'token_comparison.png', dpi=150)
    plt.close()
    
    # 3. Error Count
    fig, ax = plt.subplots(figsize=(10, 6))
    error_data = df.groupby('config')['total_errors'].agg(['mean', 'std']).reset_index()
    ax.bar(error_data['config'], error_data['mean'], yerr=error_data['std'], capsize=5)
    ax.set_ylabel('Error Count')
    ax.set_title('Validation Errors by Configuration')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'error_comparison.png', dpi=150)
    plt.close()
    
    # 4. Debug Attempts
    fig, ax = plt.subplots(figsize=(10, 6))
    debug_data = df.groupby('config')['total_debug_attempts'].agg(['mean', 'std']).reset_index()
    ax.bar(debug_data['config'], debug_data['mean'], yerr=debug_data['std'], capsize=5)
    ax.set_ylabel('Debug Attempts')
    ax.set_title('Debug Attempts by Configuration')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'debug_comparison.png', dpi=150)
    plt.close()
    
    # 5. Validation Success Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    success = df.groupby('config')['valid'].mean() * 100
    ax.bar(success.index, success.values)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Validation Success Rate')
    ax.set_ylim(0, 110)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / 'validation_success.png', dpi=150)
    plt.close()
    
    # Save data
    df.to_csv(plots_dir / 'all_results.csv', index=False)
    
    summary = df.groupby('config').agg({
        'total_time_sec': ['mean', 'std'],
        'total_tokens': ['mean'],
        'context_tokens': 'first',
        'total_errors': ['mean'],
        'total_debug_attempts': ['mean'],
        'valid': ['sum', 'count']
    }).round(2)
    summary.to_csv(plots_dir / 'summary_stats.csv')
    
    print(f"Saved to: {plots_dir}")
    print(summary.to_string())


# ======================================================
# MAIN
# ======================================================
def run_full_experiment():
    print("\n" + "="*60)
    print("CONTEXT EXPERIMENT")
    print(f"Prompt: {EXPERIMENT_PROMPT}")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENT_DIR / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    configs = [
        ("PDFs Only", load_context_config_1),
        ("All TXT Docs", load_context_config_2),
        ("Cylinder + Langium", load_context_config_3)
    ]
    
    all_results = []
    
    for config_name, config_loader in configs:
        for run in range(1, 4):
            try:
                metrics = run_single_experiment(config_name, config_loader, run, experiment_dir)
                all_results.append(metrics)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    generate_analysis(all_results, experiment_dir)
    save_json(experiment_dir, "all_results.json", all_results)
    
    print("\n" + "="*60)
    print(f"COMPLETE: {experiment_dir}")
    print("="*60)
    
    return experiment_dir


if __name__ == "__main__":
    run_full_experiment()
