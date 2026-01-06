# ======================================================
# File: simple_gwl_agent.py
# Simplified GWL Agent with 3-node architecture
# ======================================================

import os, time, json, re
from pathlib import Path
from typing import TypedDict, Optional, List, Dict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import tiktoken

# Import the validator
from gwl_validator import validate_gwl_files, ValidationReport

# ======================================================
# STATE
# ======================================================
class AgentState(TypedDict, total=False):
    user_prompt: str
    project_name: str
    project_dir: Path
    geometry_json: str
    meta_gwls: List[str]
    master_gwl: str
    validation_report: Dict
    errors: Optional[str]
    # Token tracking
    token_stats: Dict


# ======================================================
# CONFIGURATION
# ======================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR / "Docs"
OUTPUTS_BASE_DIR = SCRIPT_DIR / "Outputs"
API_FILE = DOCS_DIR / "API.txt"

OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ======================================================
# HELPERS
# ======================================================
def load_docs_for_context():
    """Load all valid Docs files with boundary markers."""
    context_blobs = []
    
    for p in sorted(DOCS_DIR.glob("*")):
        text = extract_text(p)
        if text:
            # Mark bad examples
            if "Generated__" in p.name or "Debug" in p.name or "Error" in p.name:
                tagged = f"\n### BAD EXAMPLE ({p.name}) ###\n{text[:4000]}\n### END BAD EXAMPLE ###"
            else:
                tagged = f"\n### REF DOC ({p.name}) ###\n{text[:4000]}\n### END REF DOC ###"
            context_blobs.append(tagged)
            print(f"Loaded {p.name} ({len(text)} chars)")
    
    return "\n".join(context_blobs)


def extract_text(file: Path) -> str:
    """Extract text from PDF or TXT files."""
    if not file.exists():
        return ""
    
    try:
        ext = file.suffix.lower()
        
        if ext == ".pdf":
            import fitz
            text = ""
            with fitz.open(file) as doc:
                for page in doc:
                    text += page.get_text()
            return text[:8000]
        
        elif ext == ".txt":
            with open(file, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
            return text[:8000]
        
        else:
            return ""
    
    except Exception as e:
        print(f"[WARN] Failed to read {file.name}: {e}")
        return ""


def generate_project_name(prompt: str) -> str:
    """Generate project name from timestamp and prompt keywords."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    words = re.findall(r'\w+', prompt.lower())
    skip_words = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'from', 'in', 'on', 'at'}
    meaningful = [w for w in words if w not in skip_words][:3]
    prefix = '_'.join(meaningful) if meaningful else 'design'
    return f"{prefix}_{timestamp}"


def create_project_structure(project_name: str) -> Path:
    """Create project output directory."""
    project_dir = OUTPUTS_BASE_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def save_json(project_dir: Path, filename: str, content: any, metadata: Dict = None):
    """Save JSON output with metadata."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "metadata": metadata or {}
    }
    json_path = project_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ======================================================
# NODE 1: GEOMETRY PARSER
# ======================================================
def node_geometry(state: AgentState) -> AgentState:
    """Parse user prompt into structured geometry JSON."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    prompt = state["user_prompt"]
    project_dir = state["project_dir"]
    docs_text = load_docs_for_context()
    
    system = """You are a geometry planner for Nanoscribe metasurfaces.
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

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"{prompt}\n\nReference:\n{docs_text}")
    ]
    
    # Track tokens
    input_tokens = count_tokens(system) + count_tokens(prompt) + count_tokens(docs_text)
    
    start_time = time.time()
    result = llm.invoke(messages)
    elapsed = time.time() - start_time
    
    geometry_json = result.content.strip()
    output_tokens = count_tokens(geometry_json)
    
    # Save output
    save_json(project_dir, "geometry_output.json", json.loads(geometry_json), {
        "elapsed_sec": elapsed,
        "model": "gpt-4o-mini"
    })
    
    print("\n=== GEOMETRY OUTPUT ===")
    print(geometry_json)
    print(f"Time: {elapsed:.2f}s | Tokens: {input_tokens} in / {output_tokens} out")
    print("=======================\n")
    
    # Update token stats
    token_stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0})
    token_stats["input"] += input_tokens
    token_stats["output"] += output_tokens
    token_stats["calls"] += 1
    
    state["geometry_json"] = geometry_json
    state["token_stats"] = token_stats
    return state


# ======================================================
# NODE 2: META-ATOM GWL GENERATOR
# ======================================================
def node_meta_gwls(state: AgentState) -> AgentState:
    """Generate GWL code for each unique meta-atom."""
    
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.1)
    
    geom_data = json.loads(state["geometry_json"])
    project_dir = state["project_dir"]
    docs_text = load_docs_for_context()
    
    meta_atoms = geom_data.get("meta_atoms", [])
    if not meta_atoms:
        print("[WARN] No meta_atoms found")
        return state
    
    system = """You are a DescribeX GWL generator for the Nanoscribe Quantum X.
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

    gwl_files = []
    token_stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0})
    
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
            SystemMessage(content=system),
            HumanMessage(content=human)
        ]
        
        input_tokens = count_tokens(system) + count_tokens(human)
        
        start_time = time.time()
        result = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        gwl_code = result.content.strip()
        # Clean markdown if present
        if gwl_code.startswith("```"):
            gwl_code = re.sub(r'^```\w*\n?', '', gwl_code)
            gwl_code = re.sub(r'\n?```$', '', gwl_code)
        
        output_tokens = count_tokens(gwl_code)
        
        # Save file
        fpath = project_dir / f"{atom_name}.gwl"
        fpath.write_text(gwl_code, encoding="utf-8")
        gwl_files.append(str(fpath))
        
        # Update token stats
        token_stats["input"] += input_tokens
        token_stats["output"] += output_tokens
        token_stats["calls"] += 1
        
        print(f"Generated {atom_name}.gwl ({len(gwl_code)} chars, {elapsed:.2f}s)")
    
    state["meta_gwls"] = gwl_files
    state["token_stats"] = token_stats
    return state


# ======================================================
# NODE 3: ARRAY GWL GENERATOR
# ======================================================
def node_master_gwl(state: AgentState) -> AgentState:
    """Generate the master array GWL file."""
    
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.1)
    
    project_dir = state["project_dir"]
    geom_data = json.loads(state["geometry_json"])
    user_prompt = state.get("user_prompt", "")
    meta_files = state.get("meta_gwls", [])
    docs_text = load_docs_for_context()
    
    # Get array info
    array_info = geom_data.get("array", {})
    rows = array_info.get("rows", 1)
    cols = array_info.get("cols", 1)
    spacing = array_info.get("spacing_um", 5.0)
    pattern_map = geom_data.get("pattern_map", [])
    meta_atoms = geom_data.get("meta_atoms", [])
    
    include_files = "\n".join([f"meta_{a['id']}.gwl" for a in meta_atoms])
    
    system = """You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.
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
        SystemMessage(content=system),
        HumanMessage(content=human)
    ]
    
    input_tokens = count_tokens(system) + count_tokens(human)
    
    start_time = time.time()
    result = llm.invoke(messages)
    elapsed = time.time() - start_time
    
    gwl_code = result.content.strip()
    # Clean markdown if present
    if gwl_code.startswith("```"):
        gwl_code = re.sub(r'^```\w*\n?', '', gwl_code)
        gwl_code = re.sub(r'\n?```$', '', gwl_code)
    
    output_tokens = count_tokens(gwl_code)
    
    # Save file
    fpath = project_dir / "main_array.gwl"
    fpath.write_text(gwl_code, encoding="utf-8")
    
    # Update token stats
    token_stats = state.get("token_stats", {"input": 0, "output": 0, "calls": 0})
    token_stats["input"] += input_tokens
    token_stats["output"] += output_tokens
    token_stats["calls"] += 1
    
    print(f"\nGenerated main_array.gwl ({len(gwl_code)} chars, {elapsed:.2f}s)")
    
    state["master_gwl"] = str(fpath)
    state["token_stats"] = token_stats
    return state


# ======================================================
# NODE 4: VALIDATION
# ======================================================
def node_validate(state: AgentState) -> AgentState:
    """Validate all generated GWL files against langium grammar."""
    
    project_dir = state["project_dir"]
    
    # Collect all GWL files
    gwl_files = state.get("meta_gwls", []).copy()
    if state.get("master_gwl"):
        gwl_files.append(state["master_gwl"])
    
    print("\n=== VALIDATING GWL FILES ===")
    for f in gwl_files:
        print(f"  - {Path(f).name}")
    
    # Run validation
    report = validate_gwl_files(gwl_files)
    
    # Save validation report
    report_dict = report.to_dict()
    save_json(project_dir, "validation_report.json", report_dict)
    
    # Print summary
    print("\n=== VALIDATION REPORT ===")
    for v in report.validations:
        status = "[OK] VALID" if v.valid else "[ERROR] INVALID"
        print(f"  {v.file}: {status}")
        for err in v.errors:
            print(f"    Line {err.line}: {err.message}")
    print("=========================\n")
    
    state["validation_report"] = report_dict
    return state


# ======================================================
# GRAPH DEFINITION
# ======================================================
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

compiled_graph = builder.compile()


# ======================================================
# RUN WRAPPER
# ======================================================
def run_agent(prompt_text: str) -> Dict:
    """Run the full GWL agent pipeline."""
    
    start_time = time.time()
    
    print("\n" + "="*60)
    print("SIMPLE GWL AGENT")
    print("="*60 + "\n")
    
    # Setup project
    project_name = generate_project_name(prompt_text)
    project_dir = create_project_structure(project_name)
    
    print(f"Project: {project_name}")
    print(f"Output: {project_dir}\n")
    
    # Run pipeline
    result = compiled_graph.invoke({
        "user_prompt": prompt_text.strip(),
        "project_name": project_name,
        "project_dir": project_dir,
        "token_stats": {"input": 0, "output": 0, "calls": 0}
    })
    
    total_time = time.time() - start_time
    
    # Save run stats
    token_stats = result.get("token_stats", {})
    run_stats = {
        "total_runtime_sec": round(total_time, 3),
        "input_tokens": token_stats.get("input", 0),
        "output_tokens": token_stats.get("output", 0),
        "total_tokens": token_stats.get("input", 0) + token_stats.get("output", 0),
        "llm_calls": token_stats.get("calls", 0)
    }
    save_json(project_dir, "run_stats.json", run_stats)
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Project: {project_name}")
    print(f"Runtime: {run_stats['total_runtime_sec']}s")
    print(f"Tokens: {run_stats['input_tokens']} in / {run_stats['output_tokens']} out")
    print(f"LLM Calls: {run_stats['llm_calls']}")
    print("-"*60)
    
    # Validation result
    validation = result.get("validation_report", {})
    if validation.get("valid"):
        print("[OK] All GWL files are VALID")
    else:
        print("[ERROR] Validation ERRORS found:")
        for v in validation.get("validations", []):
            if not v.get("valid"):
                print(f"  {v['file']}:")
                for err in v.get("errors", []):
                    print(f"    Line {err['line']}: {err['message']}")
    
    print("-"*60)
    print("Generated Files:")
    print(f"  - geometry_output.json")
    for f in result.get("meta_gwls", []):
        print(f"  - {Path(f).name}")
    if result.get("master_gwl"):
        print(f"  - {Path(result['master_gwl']).name}")
    print(f"  - validation_report.json")
    print(f"  - run_stats.json")
    print("="*60 + "\n")
    
    return result


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    # Read prompt from prompt.txt
    prompt_file = SCRIPT_DIR / "prompt.txt"
    if prompt_file.exists():
        USER_PROMPT = prompt_file.read_text(encoding="utf-8").strip().strip('"').strip("'")
    else:
        USER_PROMPT = "generate a 5x5 array of cylinders, height 5um, radius 1um, spacing 5um"
    
    run_agent(USER_PROMPT)
