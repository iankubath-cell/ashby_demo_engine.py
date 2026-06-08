"""
ASHBY-VIRA DEMO ENGINE v3.4 (WEB SERVER READY)
Includes: Secure Logic, Iterative Cycle Detection, Kahn's Topo Sort, and FastAPI Wrapper.
Deploy this as your main.py to Render.
"""

# --- IMPORTS ---
import json, time, math, threading, os, logging, re, operator
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import deque
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_FILE = "ashby_demo_state.json"
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10
STAGNATION_CLEANUP_INTERVAL = 3600

# --- ASHBY CONFIG ---
class AshbyConfig:
    ALPHA = 0.30
    SEVERITY_PENALTIES = {
        "low": 0.02, "medium": 0.05, "high": 0.10, "critical": 0.20,
        "minor": 0.01, "moderate": 0.04, "serious": 0.12, 
        "life_threatening": 0.25, "systemic": 0.30
    }
    STAGNATION_LIMIT = 99
    RECOVERY_RESET_SCORE = 0.85
    STABILITY_THRESHOLD_STABLE = 0.7
    STABILITY_THRESHOLD_WARNING = 0.2
    MAX_HISTORY_ENTRIES = 1000
    CYCLE_TO_STABLE_MAX = 999
    CYCLE_TO_STABLE_MIN = 0

    @staticmethod
    def sanitize_float(val: Any) -> float:
        try: return float(val)
        except (ValueError, TypeError): return 0.0

    SAFE_OPERATORS = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 
                      'truediv': lambda a, b: a / b if b != 0 else 0.0, 'pow': operator.pow, 
                      'abs': abs, 'max': max, 'min': min}

# --- RATE LIMITING ---
_rate_limit_store: Dict[str, deque] = {}
_rate_limit_lock = threading.Lock()

def check_rate_limit(user_id: str, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS) -> bool:
    now = time.time()
    with _rate_limit_lock:
        if user_id not in _rate_limit_store: _rate_limit_store[user_id] = deque()
        user_history = _rate_limit_store[user_id]
        while user_history and user_history[0] < now - window_seconds: user_history.popleft()
        if not user_history and user_id in _rate_limit_store: del _rate_limit_store[user_id]
        if len(user_history) >= max_requests: return False
        user_history.append(now)
        return True

# --- INPUT VALIDATION ---
VALID_INPUT_TYPES = {"infrastructure": ["bug", "feature_request", "general_feedback", "alert"],
                     "healthcare": ["adverse_event", "protocol_change", "patient_feedback", "drug_interaction"],
                     "finance": ["market_shock", "trade_signal", "risk_alert", "liquidity_crisis"],
                     "history": ["event_insert", "actor_removal", "constraint_change", "timeline_divergence"]}
VALID_SEVERITIES = {"low", "medium", "high", "critical", "minor", "moderate", "serious", "life_threatening", "systemic"}

def validate_input(data: dict, domain: str = None) -> Tuple[bool, Optional[str]]:
    if not isinstance(data, dict): return False, "Input must be a dictionary"
    if "type" not in data: return False, "Missing 'type' field"
    if domain and domain in VALID_INPUT_TYPES:
        if data["type"] not in VALID_INPUT_TYPES[domain]: return False, f"Invalid type '{data['type']}' for domain '{domain}'"
    if "severity" in data and data["severity"] not in VALID_SEVERITIES: return False, f"Invalid severity '{data['severity']}'"
    return True, None

# --- ENUMS ---
class LoopCategory(Enum): CATEGORY_I = "closed"; CATEGORY_II = "open"
class SystemStatus(Enum): STABLE = "stable"; WARNING = "warning"; CRITICAL = "critical"; FROZEN = "frozen"; RECOVERING = "recovering"
class AshbyTalent(Enum): HOMEOSTAT = "homeostat"; VALIDATOR = "validator"; COUNTERFACTUAL = "counterfactual"; MUTATION = "mutation"; STRESS_HEAL = "stress_heal"

DOMAIN_CONFIGS = {
    "infrastructure": {"talents": [AshbyTalent.HOMEOSTAT, AshbyTalent.VALIDATOR, AshbyTalent.MUTATION]},
    "healthcare": {"talents": [AshbyTalent.HOMEOSTAT, AshbyTalent.VALIDATOR, AshbyTalent.COUNTERFACTUAL]},
    "finance": {"talents": [AshbyTalent.HOMEOSTAT, AshbyTalent.COUNTERFACTUAL]},
    "history": {"talents": [AshbyTalent.COUNTERFACTUAL, AshbyTalent.STRESS_HEAL]}
}

def get_domain_config(domain: str) -> dict: return DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["infrastructure"])

# --- HOMEOSTAT LOGIC ---
@dataclass
class StabilityState:
    score: float = 1.0; alpha: float = AshbyConfig.ALPHA; stagnation_count: int = 0
    status: SystemStatus = SystemStatus.STABLE
    history: deque = field(default_factory=lambda: deque(maxlen=AshbyConfig.MAX_HISTORY_ENTRIES))
    last_mutation_blocked: bool = False; last_penalty_time: float = 0.0; noise_ignored_count: int = 0
    decay_cycle_count: int = 0; stagnation_start_time: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, init=False, compare=False)

    def _calculate_status(self) -> SystemStatus:
        if self.last_mutation_blocked: return SystemStatus.FROZEN
        if self.score >= AshbyConfig.STABILITY_THRESHOLD_STABLE: return SystemStatus.STABLE
        elif self.score >= AshbyConfig.STABILITY_THRESHOLD_WARNING: return SystemStatus.RECOVERING if self.stagnation_count == 0 else SystemStatus.WARNING
        else: return SystemStatus.CRITICAL

    def _cycles_to_stable(self) -> int:
        if self.score >= AshbyConfig.STABILITY_THRESHOLD_STABLE: return AshbyConfig.CYCLE_TO_STABLE_MIN
        target_gap = 1.0 - AshbyConfig.STABILITY_THRESHOLD_STABLE
        current_gap = 1.0 - self.score
        if current_gap <= 0.0001: return AshbyConfig.CYCLE_TO_STABLE_MIN
        try:
            ratio = target_gap / current_gap
            if ratio <= 0: return AshbyConfig.CYCLE_TO_STABLE_MAX
            log_base = 1 - self.alpha
            if log_base <= 0 or log_base >= 1: return AshbyConfig.CYCLE_TO_STABLE_MAX
            cycles = math.ceil(math.log(ratio) / math.log(log_base))
            return max(AshbyConfig.CYCLE_TO_STABLE_MIN, min(AshbyConfig.CYCLE_TO_STABLE_MAX, cycles))
        except: return AshbyConfig.CYCLE_TO_STABLE_MAX

    def calculate_weighted_penalty(self, severity: str, trust_score: float = 1.0) -> float:
        if trust_score is None: trust_score = 1.0
        trust_score = max(0.1, min(1.0, trust_score))
        sev_key = severity.lower() if severity else "medium"
        penalty = AshbyConfig.SEVERITY_PENALTIES.get(sev_key, 0.05)
        return penalty * trust_score

    def apply_input(self, severity: str, input_type: str = "bug", timestamp: float = None, trust_score: float = 1.0) -> dict:
        with self._lock:
            ts = timestamp or time.time()
            self.last_penalty_time = ts
            if trust_score is None: trust_score = 1.0
            if (self.stagnation_count >= AshbyConfig.STAGNATION_LIMIT and input_type in ["bug", "adverse_event", "market_shock"] and severity in ["low", "medium", "minor", "moderate"]):
                if self.stagnation_start_time and (ts - self.stagnation_start_time) < STAGNATION_CLEANUP_INTERVAL:
                    self.noise_ignored_count += 1
                    return {"status": "filtered_noise", "reason": "Ignored (Stagnation)", "stability_score": round(self.score, 4), "action": "IGNORE", "cycles_to_stable": self._cycles_to_stable()}
                self.stagnation_start_time = ts
            penalty = self.calculate_weighted_penalty(severity, trust_score)
            if input_type == "feature_request": penalty = 0.05 * trust_score
            self.score = max(0.0, min(1.0, self.score - penalty))
            if penalty > 0:
                self.stagnation_count += 1
                self.stagnation_start_time = ts if self.stagnation_count == AshbyConfig.STAGNATION_LIMIT else self.stagnation_start_time
            else: self.stagnation_count = 0
            self.status = self._calculate_status()
            action = self._decide_action()
            self.history.append({"timestamp": ts, "input": input_type, "severity": severity, "penalty": penalty, "score_after": round(self.score, 4), "action": action["action"]})
            return {"stability_score": round(self.score, 4), "status": self.status.value, "penalty_applied": penalty, "action": action, "cycles_to_stable": self._cycles_to_stable()}

    def apply_decay(self) -> dict:
        with self._lock:
            self.decay_cycle_count += 1
            new_score = self.score + self.alpha * (1.0 - self.score)
            self.score = max(0.0, min(1.0, new_score))
            if self.stagnation_count > 0 and self.score >= AshbyConfig.STABILITY_THRESHOLD_STABLE:
                self.stagnation_count = 0; self.stagnation_start_time = None
            self.status = self._calculate_status()
            return {"stability_score": round(self.score, 4), "status": self.status.value, "recovery_applied": True, "cycles_to_stable": self._cycles_to_stable()}

    def _decide_action(self) -> dict:
        if self.status == SystemStatus.FROZEN: return {"action": "ALERT_HUMAN", "reason": "Vira blocked mutation.", "loop_category": LoopCategory.CATEGORY_II.value}
        if self.status == SystemStatus.CRITICAL: return {"action": "TRIGGER_MUTATION", "reason": f"Critical ({self.score:.2f}).", "loop_category": LoopCategory.CATEGORY_II.value, "mutation_type": "arnoldian_diversion"}
        if self.status == SystemStatus.WARNING: return {"action": "FLAG", "reason": f"Degraded ({self.score:.2f}).", "loop_category": LoopCategory.CATEGORY_II.value}
        return {"action": "ALLOW", "reason": f"Nominal ({self.score:.2f}).", "loop_category": LoopCategory.CATEGORY_I.value}

    def reset_after_mutation(self) -> None:
        with self._lock:
            self.score = AshbyConfig.RECOVERY_RESET_SCORE; self.stagnation_count = 0
            self.last_mutation_blocked = False; self.stagnation_start_time = None
            self.status = self._calculate_status()

# --- VIRA VALIDATOR ---
class ViraValidator:
    @staticmethod
    def has_cycle(graph: Dict[str, List[str]]) -> bool:
        if not isinstance(graph, dict): return True
        visited, rec_stack = set(), set()
        for start_node in graph:
            if start_node in visited: continue
            stack = [(start_node, iter(graph.get(start_node, [])))]
            visited.add(start_node); rec_stack.add(start_node)
            while stack:
                node, neighbors = stack[-1]
                try:
                    neighbor = next(neighbors)
                    if neighbor not in visited:
                        visited.add(neighbor); rec_stack.add(neighbor)
                        stack.append((neighbor, iter(graph.get(neighbor, []))))
                    elif neighbor in rec_stack: return True
                except StopIteration:
                    stack.pop()
                    if rec_stack: rec_stack.discard(node)
        return False

    @staticmethod
    def is_goal_reachable(graph: Dict[str, List[str]], start: str = "Self", goal: str = "Goal") -> bool:
        if not isinstance(graph, dict): return False
        queue, visited = [start], set()
        while queue:
            node = queue.pop(0)
            if node == goal: return True
            if node in visited: continue
            visited.add(node)
            for neighbor in graph.get(node, []):
                if isinstance(neighbor, str): queue.append(neighbor)
        return False

    @staticmethod
    def verify_closure(graph: Dict[str, List[str]], start: str = "Self", goal: str = "Goal") -> dict:
        has_cycle_flag = ViraValidator.has_cycle(graph)
        reachable = ViraValidator.is_goal_reachable(graph, start, goal)
        if has_cycle_flag: return {"closure": LoopCategory.CATEGORY_II.value, "valid": False, "reason": "Cycle detected.", "path": None}
        if not reachable: return {"closure": LoopCategory.CATEGORY_II.value, "valid": False, "reason": "Goal unreachable.", "path": None}
        return {"closure": LoopCategory.CATEGORY_I.value, "valid": True, "reason": "Category I.", "path": [start, goal]}

    @staticmethod
    def validate_mutation(mutation_graph: Dict[str, List[str]], stability: 'StabilityState') -> dict:
        if not isinstance(mutation_graph, dict): return {"approved": False, "reason": "Invalid graph.", "closure": LoopCategory.CATEGORY_II.value, "action": "ALERT_HUMAN"}
        result = ViraValidator.verify_closure(mutation_graph)
        if not result["valid"]:
            with stability._lock: stability.last_mutation_blocked = True; stability.status = SystemStatus.FROZEN
            return {"approved": False, "reason": f"Rejected: {result['reason']}", "closure": result["closure"], "action": "ALERT_HUMAN"}
        return {"approved": True, "reason": "Approved.", "closure": result["closure"], "action": "APPLY_MUTATION"}

# --- COUNTERFACTUAL TALENT ---
class DemoCounterfactual:
    def __init__(self, equations: Dict[str, str], parents: Dict[str, list]):
        self.equations = equations; self.parents = parents; self.abducted_u: Dict[str, float] = {}; self._lock = threading.Lock()

    def _safe_eval(self, expr: str, context: Dict[str, float]) -> float:
        forbidden = ['__import__', 'os.', 'sys.', 'exec(', 'eval(', 'compile(', 'getattr', 'setattr']
        for f in forbidden:
            if f in expr: raise ValueError("Unsafe expression detected.")
        local_env = {"__builtins__": {}, "abs": abs, "max": max, "min": min}; local_env.update(context)
        try: return float(eval(expr, {"__builtins__": None}, local_env))
        except Exception as e: logger.error(f"Eval failed: {e}"); return 0.0

    def _topo_sort_kahn(self) -> List[str]:
        in_degree = {node: 0 for node in self.equations}
        for node, deps in self.parents.items():
            if not isinstance(deps, list): continue
            for parent in deps:
                if parent in self.equations: in_degree[node] = in_degree.get(node, 0) + 1
        queue = deque([node for node, degree in in_degree.items() if degree == 0]); sorted_nodes = []
        while queue:
            node = queue.popleft(); sorted_nodes.append(node)
            for child, deps in self.parents.items():
                if not isinstance(deps, list): continue
                if node in deps:
                    in_degree[child] -= 1
                    if in_degree[child] == 0: queue.append(child)
        if len(sorted_nodes) != len(self.equations): return list(self.equations.keys())
        return sorted_nodes

    def abduction(self, observed: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            self.abducted_u = {}
            sorted_nodes = self._topo_sort_kahn()
            for node in sorted_nodes:
                try:
                    parent_list = self.parents.get(node, [])
                    if not isinstance(parent_list, list): parent_list = []
                    parent_vals = {p: AshbyConfig.sanitize_float(observed.get(p, 0.0)) for p in parent_list}
                    eq_str = self.equations.get(node, "0")
                    base = self._safe_eval(eq_str, parent_vals)
                    obs_val = AshbyConfig.sanitize_float(observed.get(node, 0.0))
                    self.abducted_u[node] = obs_val - base
                except Exception as e: logger.error(f"Abduction error: {e}"); self.abducted_u[node] = 0.0
            return self.abducted_u

    def predict(self, intervention: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            twin = {}; sorted_nodes = self._topo_sort_kahn()
            for node in sorted_nodes:
                if node in intervention: twin[node] = AshbyConfig.sanitize_float(intervention[node]); continue
                try:
                    parent_list = self.parents.get(node, [])
                    if not isinstance(parent_list, list): parent_list = []
                    parent_vals = {p: AshbyConfig.sanitize_float(twin.get(p, 0.0)) for p in parent_list}
                    eq_str = self.equations.get(node, "0")
                    base = self._safe_eval(eq_str, parent_vals)
                    twin[node] = base + AshbyConfig.sanitize_float(self.abducted_u.get(node, 0.0))
                except Exception as e: logger.error(f"Predict error: {e}"); twin[node] = 0.0
            return twin

    def run(self, observed: Dict[str, float], intervention: Dict[str, float], target: str = None) -> Dict[str, Any]:
        try:
            if not isinstance(observed, dict) or not isinstance(intervention, dict): return {"error": "Invalid input types", "success": False}
            self.abduction(observed); result = self.predict(intervention)
            target_val = result.get(target) if target else None
            if target_val is not None and not isinstance(target_val, (int, float)): target_val = 0.0
            summary = f"If {list(intervention.keys())[0]} was {list(intervention.values())[0]}, then {target} would be {target_val}." if target else "Computed."
            return {"abducted_noise": self.abducted_u, "counterfactual_world": result, "target_value": target_val, "summary": summary, "success": True}
        except Exception as e: logger.error(f"CF Error: {e}"); return {"error": str(e), "success": False}

# --- SCENARIOS ---
DEMO_SCENARIOS = {
    "power_temp": {"equations": {"Power": "0", "Temp": "2 * Power", "Shutdown": "1 if Temp > 100 else 0"}, "parents": {"Power": [], "Temp": ["Power"], "Shutdown": ["Temp"]}},
    "history_genghis": {"equations": {"Genghis": "1", "Empire": "100 * Genghis", "Trade": "50 + 20 * Empire", "Plague": "10 if Trade > 100 else 0"}, "parents": {"Genghis": [], "Empire": ["Genghis"], "Trade": ["Empire"], "Plague": ["Trade"]}},
    "healthcare_dosage": {"equations": {"Dosage": "50", "BloodLevel": "2 * Dosage", "Effect": "BloodLevel * 0.5", "Toxicity": "1 if BloodLevel > 200 else 0"}, "parents": {"Dosage": [], "BloodLevel": ["Dosage"], "Effect": ["BloodLevel"], "Toxicity": ["BloodLevel"]}}
}

# --- PERSISTENCE ---
def save_state(state_obj: StabilityState) -> bool:
    try:
        temp_file = STATE_FILE + ".tmp"
        with open(temp_file, "w") as f: json.dump({"score": state_obj.score, "stagnation_count": state_obj.stagnation_count, "history": list(state_obj.history), "last_mutation_blocked": state_obj.last_mutation_blocked, "noise_ignored_count": state_obj.noise_ignored_count, "decay_cycle_count": state_obj.decay_cycle_count, "saved_at": datetime.now().isoformat()}, f)
        os.replace(temp_file, STATE_FILE); return True
    except Exception as e: logger.error(f"Save failed: {e}"); return False

def load_state() -> StabilityState:
    state = StabilityState()
    if not os.path.exists(STATE_FILE): return state
    try:
        with open(STATE_FILE, "r") as f: data = json.load(f)
        state.score = max(0.0, min(1.0, AshbyConfig.sanitize_float(data.get("score", 1.0))))
        state.stagnation_count = int(AshbyConfig.sanitize_float(data.get("stagnation_count", 0)))
        hist_data = data.get("history", [])
        state.history = deque(hist_data, maxlen=AshbyConfig.MAX_HISTORY_ENTRIES) if isinstance(hist_data, list) else deque(maxlen=AshbyConfig.MAX_HISTORY_ENTRIES)
        state.last_mutation_blocked = bool(data.get("last_mutation_blocked", False))
        state.noise_ignored_count = int(AshbyConfig.sanitize_float(data.get("noise_ignored_count", 0)))
        state.decay_cycle_count = int(AshbyConfig.sanitize_float(data.get("decay_cycle_count", 0)))
        state.status = state._calculate_status(); return state
    except Exception as e: logger.error(f"Load failed: {e}"); return StabilityState()

_system_lock = threading.Lock(); system_state = None
def _initialize_global_state(): global system_state
    with _system_lock:
        if system_state is None: system_state = load_state()

_initialize_global_state()

# --- HANDLERS ---
def handle_demo_request(domain: str, event: dict) -> dict:
    result = {}
    try:
        config = get_domain_config(domain)
        if AshbyTalent.HOMEOSTAT in config["talents"]:
            try:
                valid, err = validate_input(event, domain)
                if not valid: result["homeostat"] = {"status": "blocked", "reason": err}
                else:
                    res = system_state.apply_input(event.get("severity", "medium"), event.get("type", "bug"), time.time(), event.get("trust_score", 1.0))
                    result["homeostat"] = res; save_state(system_state)
            except Exception as e: result["homeostat"] = {"status": "error", "reason": str(e)}
        if AshbyTalent.COUNTERFACTUAL in config["talents"] and "counterfactual" in event:
            try:
                cf_data = event["counterfactual"]; scenario = cf_data.get("scenario", "power_temp")
                if scenario not in DEMO_SCENARIOS: result["counterfactual"] = {"error": f"Scenario '{scenario}' not found", "success": False}
                else:
                    eqs_raw = cf_data.get("equations", DEMO_SCENARIOS[scenario]["equations"])
                    if not isinstance(eqs_raw, dict): eqs_raw = DEMO_SCENARIOS[scenario]["equations"]
                    eqs = {k: str(v) for k, v in eqs_raw.items()}; pars = cf_data.get("parents", DEMO_SCENARIOS[scenario]["parents"])
                    engine = DemoCounterfactual(eqs, pars)
                    result["counterfactual"] = engine.run(cf_data.get("observed", {}), cf_data.get("intervention", {}), cf_data.get("target"))
            except Exception as e: result["counterfactual"] = {"error": str(e), "success": False}
        if AshbyTalent.VALIDATOR in config["talents"] and "mutation_graph" in event:
            try:
                res = ViraValidator.validate_mutation(event["mutation_graph"], system_state)
                if res["approved"]:
                    with _system_lock: system_state.reset_after_mutation()
                    save_state(system_state)
                result["validation"] = res
            except Exception as e: result["validation"] = {"approved": False, "reason": str(e)}
        result["domain"] = domain; result["active_talents"] = [t.value for t in config["talents"]]
        with _system_lock: result["system_status"] = system_state.status.value; result["stability_score"] = round(system_state.score, 4)
        result["success"] = True
    except Exception as e: result["success"] = False; result["error"] = str(e)
    return result

def handle_decay_cycle() -> dict:
    with _system_lock: res = system_state.apply_decay(); save_state(system_state)
    return {"status": "decay_applied", "stability_score": res["stability_score"], "system_status": res["status"], "cycles_to_stable": res["cycles_to_stable"], "success": True}

def handle_get_state() -> dict:
    with _system_lock: return {"stability_score": round(system_state.score, 4), "status": system_state.status.value, "stagnation_count": system_state.stagnation_count, "history_length": len(system_state.history), "cycles_to_stable": system_state._cycles_to_stable(), "noise_ignored": system_state.noise_ignored_count}

# ================================================================
# PART 2: FASTAPI WRAPPER (THE WEB SERVER PORTION)
# ================================================================

app = FastAPI(title="Ashby-Vira Demo Engine", version="3.4")

class FeedbackEvent(BaseModel):
    domain: str
    event: dict

@app.post("/feedback")
async def receive_feedback(payload: FeedbackEvent):
    """Receives bug/report and updates stability score."""
    try:
        result = handle_demo_request(payload.domain, payload.event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state():
    """Returns current stability score and status."""
    try:
        return handle_get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heal")
async def heal_system():
    """Triggers decay cycle to recover stability."""
    try:
        return handle_decay_cycle()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Ashby-Vira Demo Engine v3.4 is running", "status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
