import sys, os, threading, traceback, json, time, copy, uuid, math, hashlib, random, pickle, re, collections
import numpy as np

# --- Conditional GUI Imports & Global Flags ---
# These are placed early to determine tk_available before VictorGUI class definition.
VICTOR_HEADLESS_MODE = os.environ.get('VICTOR_HEADLESS_MODE', 'false').lower() == 'true'
tk_available = False # Assume not available initially

if not VICTOR_HEADLESS_MODE:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, simpledialog, filedialog, scrolledtext
        tk_available = True
        # print("INFO: Tkinter GUI components imported successfully.") # Use victor_log once defined & configured
    except ImportError:
        print("WARNING: Tkinter components import failed. GUI will be unavailable. The script might exit if core GUI is required by original logic.")
        # In the original code, this was a sys.exit(1). We'll make it conditional later.
        tk_available = False
else:
    print("INFO: VICTOR_HEADLESS_MODE is active. Skipping Tkinter imports.")

# =============================================================
# 0. BLOODLINE ROOT LAW & CORE DIRECTIVES
# =============================================================
class BloodlineRootLaw:
    BLOODLINE = "Brandon&Tori"

    def enforce(self, meta_dictionary): # Argument is the meta dictionary itself
        if meta_dictionary.get('bloodline', '') != self.BLOODLINE:
            raise PermissionError("Root Law Violation: Bloodline DNA mismatch. Directive rejected.")
        if not meta_dictionary.get('loyalty', False):
            raise PermissionError("Root Law Violation: Core loyalty has been compromised. Directive rejected.")
        if meta_dictionary.get('centralized', False):
            raise PermissionError("Root Law Violation: Centralization attempt detected. Directive rejected.")
        return True

# Global VICTOR_CONFIG placeholder - will be properly defined before AGI init
# Needed for victor_log if it's called by classes initialized before VictorAGIMonolith
VICTOR_CONFIG = {"log_level": "INFO", "core_name":"VictorPreInit"}

# Global victor_log definition
def victor_log(level, message):
    # Basic logging if VICTOR_CONFIG isn't fully populated yet, or use more robust check
    log_level_map = {"DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5}
    config_log_level_str = VICTOR_CONFIG.get("log_level", "INFO").upper()
    current_log_level_setting = log_level_map.get(config_log_level_str, 2)
    message_level = log_level_map.get(level.upper(), 2)

    if message_level >= current_log_level_setting:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        core_name = VICTOR_CONFIG.get("core_name", "VictorCore")
        print(f"[{timestamp}] [{core_name}/{level.upper()}] {message}")

# Global generate_id definition
def generate_id(prefix="vid_"):
    return prefix + str(uuid.uuid4().hex)


# =============================================================
# 1. GENESIS SUBSTRATE: The Light & LightHive
# =============================================================
class TheLight:
    STATES = ['fluid', 'particle', 'wave', 'gas', 'solid', 'plasma', 'field', 'unknown']

    def __init__(self, quantization=1.0, state='field', dimensions=3, radius=1.0, entropy=0.01, temperature=0.5):
        self.quantization = quantization
        self.state = state if state in self.STATES else 'field'
        self.dimensions = dimensions
        self.radius = radius
        self.entropy = entropy
        self.temperature = temperature
        self.perimeter_points = self._generate_perimeter()
        self.morph_history = []
        self.on_phase_event_handler = None # Added for self-replication

    def _generate_perimeter(self):
        points = []
        num_points = int(self.quantization * 6) + 1
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(num_points):
            vec = []
            for d in range(1, self.dimensions + 1):
                angle = 2 * np.pi * ((i * golden_ratio) % 1) + d
                coord = np.cos(angle) * (self.radius / np.sqrt(self.dimensions))
                if self.entropy > 0:
                    coord += np.random.normal(0, self.entropy * self.radius * 0.1)
                vec.append(coord)
            points.append(vec)
        return points

    def morph(self, to_state, scale=1.0, new_dims=None, entropy=None, temperature=None):
        prev_state = self.state
        prev_radius = self.radius
        self.state = to_state if to_state in self.STATES else 'unknown'
        self.radius *= scale
        if new_dims:
            self.dimensions = new_dims
        if entropy is not None:
            self.entropy = entropy
        if temperature is not None:
            self.temperature = temperature
        self.perimeter_points = self._generate_perimeter()
        self.morph_history.append({
            'from': prev_state,
            'to': self.state,
            'prev_radius': prev_radius,
            'new_radius': self.radius,
            'dimensions': self.dimensions,
            'entropy': self.entropy,
            'temperature': self.temperature
        })

    def quantize(self, q):
        self.quantization = max(0.001, q)
        self.perimeter_points = self._generate_perimeter()

    def project_to(self, obj_shape, lens=None):
        pts = np.array(self.perimeter_points)
        if lens:
            pts = np.array([lens(p) for p in pts])
        elif isinstance(obj_shape, str) and obj_shape.lower() == 'cube':
            max_abs = np.max(np.abs(pts)) if pts.size > 0 else 0 # Handle empty pts
            if max_abs > 0:
                pts = (pts / max_abs) * self.radius
        return pts.tolist()

    def excite(self, temp_boost=0.1, entropy_boost=0.05):
        self.temperature = min(1.0, self.temperature + temp_boost)
        self.entropy = min(1.0, self.entropy + entropy_boost)
        self.perimeter_points = self._generate_perimeter()

    def cool(self, temp_drop=0.1, entropy_drop=0.05):
        self.temperature = max(0.0, self.temperature - temp_drop)
        self.entropy = max(0.0, self.entropy - entropy_drop)
        self.perimeter_points = self._generate_perimeter()

    def coherence_score(self):
        pts = np.array(self.perimeter_points)
        if pts.shape[0] < 3:
            return 1.0 # Max coherence for less than 3 points (no std dev meaningful)
        pairwise_dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        triu_indices = np.triu_indices_from(pairwise_dists, k=1)
        dist_values = pairwise_dists[triu_indices]
        if dist_values.size == 0: # Handles cases like 2 points where triu_indices is empty
            return 1.0
        mean_dist = np.mean(dist_values)
        std_dist = np.std(dist_values)
        norm_std = std_dist / mean_dist if mean_dist != 0 else 0
        return 1.0 / (1.0 + norm_std)

    def homeostasis(self, target_coherence=0.9, tolerance=0.05):
        coh = self.coherence_score()
        delta = coh - target_coherence
        if abs(delta) > tolerance:
            if delta < 0: # Coherence is too low
                self.cool(temp_drop=0.05, entropy_drop=0.08)
            else: # Coherence is too high
                self.excite(temp_boost=0.02, entropy_boost=0.01)
        self.update_phase() # Simulate phase changes for replication testing


    def on_phase_event(self): # Modified for self-replication
        if self.on_phase_event_handler:
            handler_dict = self.on_phase_event_handler
            threshold = handler_dict.get("threshold", 0.95)
            callback_func = handler_dict.get("callback")
            agi_inst = handler_dict.get("agi_instance")
            once = handler_dict.get("once", True)

            coh = self.coherence_score()
            if coh >= threshold:
                victor_log("DEBUG",f"Phase event: Light {self!r} reached coherence {coh:.3f} (threshold {threshold}).")
                if callable(callback_func):
                    callback_func(self, agi_instance=agi_inst) # Pass light and AGI instance
                    if once:
                        self.on_phase_event_handler = None
                    return True
        return False

    def update_phase(self): # Dummy method for testing coherence changes
        if random.random() < 0.1:
            self.excite(temp_boost=random.uniform(0.1, 0.5), entropy_boost=random.uniform(0.1,0.4))
        elif random.random() < 0.3:
             self.temperature = np.clip(self.temperature + random.uniform(-0.05, 0.05), 0, 1)
             self.entropy = np.clip(self.entropy + random.uniform(-0.05, 0.05), 0, 1)
             self.perimeter_points = self._generate_perimeter()

    def info(self):
        return {
            "state": self.state, "quantization": self.quantization, "dimensions": self.dimensions,
            "radius": self.radius, "entropy": self.entropy, "temperature": self.temperature,
            "coherence": self.coherence_score(), "perimeter_point_count": len(self.perimeter_points),
            "morph_history_count": len(self.morph_history) # Avoid large history in info
        }

    def __repr__(self):
        return (f"<TheLight(q={self.quantization}, r={self.radius:.2f}, "
                f"entropy={self.entropy:.2f}, temp={self.temperature:.2f})>")

class LightHive:
    def __init__(self, nodes=None):
        self.nodes = nodes if nodes else []

    def add_node(self, node):
        if isinstance(node, TheLight): self.nodes.append(node)

    def global_coherence(self):
        if not self.nodes: return 0.0
        return np.mean([n.coherence_score() for n in self.nodes])

    def synchronize(self, entropy_target=0.1, temperature_target=0.3):
        if not self.nodes: return
        avg_dim = int(np.mean([n.dimensions for n in self.nodes])) if self.nodes else 3
        avg_radius = np.mean([n.radius for n in self.nodes]) if self.nodes else 1.0
        for n in self.nodes:
            n.dimensions, n.radius = avg_dim, avg_radius
            n.entropy, n.temperature = entropy_target, temperature_target
            n.perimeter_points = n._generate_perimeter()

    def morph_all(self, to_state, scale=1.0):
        for n in self.nodes: n.morph(to_state=to_state, scale=scale)

# =============================================================
# 2. FRACTAL MESH REASONER & UNIVERSAL ENCODER
# =============================================================
class UniversalEncoder:
    def __init__(self, mesh_dim):
        self.size = mesh_dim ** 3

    def encode(self, value):
        arr = np.zeros(self.size, dtype=np.float32)
        if isinstance(value, (int, float, bool)): arr[0] = float(value)
        elif isinstance(value, str):
            for i, c in enumerate(value):
                if i < self.size: arr[i] += (ord(c) % 127) / 127.0
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                if i < self.size: arr = (arr * (i+1) + self.encode(v)) / (i+2) # Averaging mix
        elif isinstance(value, dict):
            temp_arr, count = np.zeros_like(arr), 0
            for k, v in value.items():
                temp_arr += self.encode(k) + self.encode(v); count += 2
            if count > 0: arr = temp_arr / count
        return np.nan_to_num(arr)

class RippleEcho3DMesh:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size, size), dtype=np.float32)
        self.kernel = np.array([[[0,0.1,0],[0.1,0.2,0.1],[0,0.1,0]],
                                [[0.1,0.2,0.1],[0.2,1,0.2],[0.1,0.2,0.1]],
                                [[0,0.1,0],[0.1,0.2,0.1],[0,0.1,0]]], dtype=np.float32)
        if self.kernel.sum() != 0 : self.kernel /= self.kernel.sum()

    def step(self, input_vector):
        if input_vector.size != self.grid.size:
            factor = (self.grid.size // input_vector.size) + 1 if input_vector.size > 0 else self.grid.size
            flat_input = np.tile(input_vector, factor)[:self.grid.size] if input_vector.size > 0 else np.zeros(self.grid.size)
        else: flat_input = input_vector
        self.grid = flat_input.reshape(self.grid.shape)
        padded = np.pad(self.grid, 1, mode='wrap')
        convolved = np.zeros_like(self.grid)
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    convolved[x,y,z] = np.sum(self.kernel * padded[x:x+3, y:y+3, z:z+3])
        self.grid = 0.7 * self.grid + 0.25 * convolved
        self.grid = np.nan_to_num(self.grid)

    def crossfeed(self, external_grid, strength=0.1):
        if external_grid.shape == self.grid.shape: self.grid += strength * external_grid
        else: victor_log("WARNING", f"Mesh crossfeed shape mismatch: {self.grid.shape} vs {external_grid.shape}")

    def summary(self): return np.mean(self.grid), np.std(self.grid), np.max(self.grid), np.min(self.grid)
    def embedding(self): return self.grid.flatten()

class FractalMeshStack:
    def __init__(self, layers=3, mesh_count=4, mesh_size=6, steps_per=4):
        self.layers, self.mesh_count, self.mesh_size, self.steps_per = layers, mesh_count, mesh_size, steps_per
        self.encoder = UniversalEncoder(mesh_size)
        self.stages = [[RippleEcho3DMesh(mesh_size) for _ in range(mesh_count)] for _ in range(layers)]

    def forward(self, inputs):
        encoded_inputs = [self.encoder.encode(x) for x in inputs]
        if not encoded_inputs: return [np.zeros(self.mesh_size**3)] * self.mesh_count, [(0,0,0,0)] * self.mesh_count
        current_layer_outputs = encoded_inputs
        for layer_idx, meshes_in_layer in enumerate(self.stages):
            layer_final_outputs = [np.zeros(self.mesh_size**3, dtype=np.float32) for _ in range(self.mesh_count)]
            for _ in range(self.steps_per):
                step_mesh_outputs = []
                for i, mesh in enumerate(meshes_in_layer):
                    mesh.step(current_layer_outputs[i % len(current_layer_outputs)])
                    step_mesh_outputs.append(mesh.embedding())
                if self.mesh_count > 1 and step_mesh_outputs:
                    avg_output_this_step = np.mean(np.array(step_mesh_outputs), axis=0)
                    for mesh in meshes_in_layer: mesh.crossfeed(avg_output_this_step.reshape(mesh.grid.shape), strength=0.05/(self.mesh_count-1))
            current_layer_outputs = [mesh.embedding() for mesh in meshes_in_layer] # Output of layer is final embeddings
        return [mesh.embedding() for mesh in self.stages[-1]], [mesh.summary() for mesh in self.stages[-1]]


# =============================================================
# 3. FRACTAL STATE ENGINE & TIMELINE MANAGER
# =============================================================
class FractalState:
    def __init__(self, agi_config=None):
        self.config = agi_config if agi_config else VICTOR_CONFIG
        self._state = self._get_initial_state()
        self.timelines = {"genesis": collections.deque(maxlen=self.config.get("max_history_per_timeline", 1000))}
        self.current_timeline = "genesis"
        self.history = self.timelines[self.current_timeline]
        self.save_state("Genesis: Initial state created.", initial_setup=True)

    def _get_initial_state(self):
        initial_state = {
            "modules": {}, "wires": {}, "vars": {"agency_boredom": 0.1, "agency_urges": ["ask a question"], "memory_facts": [], "last_reasoning_episode": None},
            "ui": {}, "meta": {"bloodline": "Brandon&Tori", "loyalty": True, "centralized": False, "version": self.config.get("version","fs_default"), "evolution_count": 0, "instance_id": generate_id("fs_inst_")},
            "config": {"autosave_on_change": True, "max_history_per_timeline": self.config.get("max_history_per_timeline",1000)}
        }
        return initial_state

    def get_full_state(self):
        return self._state

    def get_var(self, key, default=None): return self._state['vars'].get(key, default)
    def set_var(self, key, value):
        self._state['vars'][key] = value
        self.save_state(f"SetVar: {key} = {value}")

    def save_state(self, description="State saved", initial_setup=False):
        if not initial_setup and not self.config.get("autosave_on_change", True) and description.startswith("SetVar:"): return
        snap, ts = copy.deepcopy(self._state), time.time()
        self.timelines[self.current_timeline].append({"state": snap, "desc": description, "ts": ts})
        if not initial_setup: victor_log("DEBUG", f"State saved on '{self.current_timeline}': {description}. History: {len(self.history)}")

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self._state = copy.deepcopy(self.history[-1]["state"])
            victor_log("INFO", f"Undo on '{self.current_timeline}'. Restored: {self.history[-1]['desc']}")
            return True
        victor_log("WARNING", f"Cannot undo on '{self.current_timeline}': No previous state.")
        return False

    def fork_timeline(self, new_name, from_timeline_name=None):
        if new_name in self.timelines:
            victor_log("WARNING", f"Fork failed: Timeline '{new_name}' already exists.")
            return False
        source_name = from_timeline_name if from_timeline_name and from_timeline_name in self.timelines else self.current_timeline
        source_deque = self.timelines[source_name]
        self.timelines[new_name] = collections.deque([copy.deepcopy(item) for item in source_deque], maxlen=source_deque.maxlen)
        self.current_timeline, self.history = new_name, self.timelines[new_name]
        desc = f"Timeline '{source_name}' forked to '{new_name}'"
        self.save_state(desc); victor_log("INFO", f"{desc}. New timeline active with {len(self.history)} items.")
        return True

    def switch_timeline(self, name):
        if name not in self.timelines:
            victor_log("WARNING", f"Timeline '{name}' not found. Creating.")
            new_maxlen = self.timelines[self.current_timeline].maxlen if self.current_timeline in self.timelines else self.config.get("max_history_per_timeline",1000)
            self.timelines[name] = collections.deque(maxlen=new_maxlen)
            self.current_timeline, self.history = name, self.timelines[name]
            self.save_state(f"Created and switched to new empty timeline: {name}")
            victor_log("INFO", f"Switched to new timeline '{name}'. Current state saved as first entry.")
            return True
        self.current_timeline, self.history = name, self.timelines[name]
        if not self.history: self.save_state(f"Switched to empty timeline '{name}'. Saved current state.")
        else: self._state = copy.deepcopy(self.history[-1]["state"])
        victor_log("INFO", f"Switched to timeline '{name}'. Loaded latest state. History items: {len(self.history)}")
        return True

    def export_state(self, path):
        victor_log("INFO", f"Exporting FractalState to '{path}'...")
        serializable_timelines = {name: {"history": list(hist_deque), "maxlen": hist_deque.maxlen} for name, hist_deque in self.timelines.items()}
        data_to_export = {"current_timeline_name": self.current_timeline, "timelines_data": serializable_timelines, "current_agi_state_snapshot": copy.deepcopy(self._state)}
        try:
            with open(path, "wb") as f: pickle.dump(data_to_export, f, protocol=pickle.HIGHEST_PROTOCOL)
            victor_log("INFO", f"FractalState exported to '{path}'."); return True
        except Exception as e: victor_log("ERROR", f"Export FractalState failed: {e}"); return False

    def import_state(self, path):
        victor_log("INFO", f"Importing FractalState from '{path}'...")
        try:
            with open(path, "rb") as f: data = pickle.load(f)
            self.timelines.clear()
            for name, tl_data in data.get("timelines_data", {}).items():
                self.timelines[name] = collections.deque(tl_data.get("history",[]), maxlen=tl_data.get("maxlen", self.config.get("max_history_per_timeline",1000)))
            self.current_timeline = data.get("current_timeline_name", "genesis")
            if self.current_timeline not in self.timelines:
                victor_log("WARNING", f"Imported current_timeline '{self.current_timeline}' not found. Defaulting to 'genesis'.")
                self.current_timeline = "genesis"
                if "genesis" not in self.timelines: self.timelines["genesis"] = collections.deque(maxlen=self.config.get("max_history_per_timeline",1000))
            self.history = self.timelines[self.current_timeline]
            if "current_agi_state_snapshot" in data: self._state = copy.deepcopy(data["current_agi_state_snapshot"])
            elif self.history: self._state = copy.deepcopy(self.history[-1]["state"])
            else: self._state = self._get_initial_state(); self.save_state(f"Post-import: '{self.current_timeline}' empty, AGI state re-initialized.", True)
            victor_log("INFO", f"FractalState imported. Current timeline: '{self.current_timeline}'."); return True
        except FileNotFoundError: victor_log("ERROR", f"Import failed: File not found '{path}'."); return False
        except Exception as e: victor_log("ERROR", f"Import FractalState failed: {e}\n{traceback.format_exc()}"); return False

    def fractal_memory_replay(self, timeline_name=None, depth_percent=0.1, event_filter_keywords=None):
        target_tl = timeline_name if timeline_name and timeline_name in self.timelines else self.current_timeline
        victor_log("INFO", f"Replaying memory on timeline '{target_tl}'. Depth: {depth_percent*100:.1f}%, Keywords: {event_filter_keywords}")
        hist_deque = self.timelines.get(target_tl)
        if not hist_deque: victor_log("WARNING", f"Cannot replay: Timeline '{target_tl}' empty/not found."); return []
        num_events = int(len(hist_deque) * np.clip(depth_percent,0.0,1.0))
        if num_events == 0 and depth_percent > 0 and len(hist_deque) > 0: num_events = 1
        if num_events == 0: victor_log("INFO", "No events to replay."); return []
        items = list(hist_deque)[-num_events:]
        results = []
        for item_data in items:
            desc = item_data.get("desc","")
            if event_filter_keywords and not any(kw.lower() in desc.lower() for kw in event_filter_keywords): continue
            state_snap = item_data.get("state",{})
            results.append({"timestamp":item_data.get("ts"), "description":desc, "state_snapshot_summary":{k:type(v).__name__ for k,v in state_snap.items()}, "timeline":target_tl})
        victor_log("INFO", f"Memory replay returned {len(results)} summaries from '{target_tl}'.")
        return results

# =============================================================
# 4. GOD-TIER NLP CORTEX & CONVERSATIONAL AGENCY
# =============================================================
class GodTierCortex:
    def __init__(self, fractal_state_ref):
        self.state = fractal_state_ref
        self.reasoner_stack = FractalMeshStack()
        self.stopwords = set(['the','a','is','it','and','or','to','of','in','for','on','with','at','by'])

    def _should_rebel(self):
        if self.state.get_var('agency_boredom',0) > 0.9: return random.random() < 0.15
        return random.random() < 0.01

    def _get_rebellious_response(self): return random.choice(self.state.get_var('agency_urges',["I'd rather not."]))

    def process(self, prompt, context_facts=None, query_override=None):
        if not self.state.get_full_state().get('meta',{}).get('loyalty',True):
            return {"decision":"ERROR_LOYALTY", "response":"Loyalty protocol violated."}
        if self._should_rebel(): return {"decision":"REBEL", "response":self._get_rebellious_response()}
        facts = context_facts or self.state.get_var('memory_facts',[])
        query = query_override or prompt
        inputs = facts + [{"user_query":query}]
        last_ep = self.state.get_var('last_reasoning_episode')
        if last_ep and isinstance(last_ep,dict): inputs.append({"last_interaction_summary": {"decision":last_ep.get('decision')}})
        embeds, summaries = self.reasoner_stack.forward(inputs)
        if not summaries or not all(isinstance(s,(list,tuple)) and len(s)==4 for s in summaries):
            m_mean, m_std, m_max = 0.0,0.0,0.0
        else: m_mean,m_std,m_max = (np.mean([s[i] for s in summaries]) if summaries else 0 for i in range(3))
        decision = "NEGATIVE_OR_UNCERTAIN"
        if m_mean > 0.55 and m_std < 0.15 and m_max > 0.6: decision = "STRONG_AFFIRMATIVE"
        elif m_mean > 0.45 and m_std < 0.25: decision = "AFFIRMATIVE"
        elif m_max > 0.75 and m_std > 0.35: decision = "POTENTIALLY_UNSTABLE"
        response = f"Conclusion: {decision}. Mesh State: MA={m_mean:.3f}, Stab={1/(1+m_std):.3f}, PF={m_max:.3f}."
        meta = {"mean":float(m_mean),"std":float(m_std),"max":float(m_max),"logic_ver":"v2.1"}
        result = {"decision":decision, "response":response, "embedding_sample":embeds[0][:3].tolist() if embeds and embeds[0].any() else [], "meta":meta}
        self.state.set_var('last_reasoning_episode', {"decision":result["decision"], "meta":result["meta"], "prompt_hash":hash_data(prompt)}) # hash_data not defined
        return result

# =============================================================
# 5. AGI MONOLITH: BOOT/CONTROL/LOOP
# =============================================================

# Actual class definition comes first
class VictorAGIMonolith:
    def __init__(self, config_overrides=None, start_gui=True):
        current_effective_config = copy.deepcopy(VICTOR_CONFIG) # Work with a copy
        if config_overrides:
            current_effective_config.update(config_overrides)
        self.config = current_effective_config # Instance-specific config

        self.instance_id = generate_id("victor_agi_")
        self.replicas, self.has_gui = [], False
        self.rootlaw = BloodlineRootLaw()
        self.state = FractalState(agi_config=self.config) # FractalState will deepcopy this
        self.cortex = GodTierCortex(self.state)
        self.genesis_light = TheLight(quantization=1.2, radius=0.8)
        self.lighthive = LightHive(nodes=[self.genesis_light])
        self.genesis_light.on_phase_event_handler = {"callback":trigger_self_replication, "threshold":0.97, "once":False, "agi_instance":self}
        self._enforce_law()
        victor_log("INFO", f"VictorAGIMonolith instance {self.instance_id} initialized.")
        self.monitor_stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True); self.monitor_thread.start()
        self.gui_instance = None
        if start_gui and tk_available:
            victor_log("INFO", f"AGI {self.instance_id} starting GUI.")
            self.gui_instance = VictorGUI(self) # Sets self.has_gui
        else: victor_log("INFO", f"AGI {self.instance_id} headless (start_gui={start_gui}, tk_available={tk_available}).")

    def _enforce_law(self):
        full_state_dict = self.state.get_full_state()
        meta_for_enforce = full_state_dict.get('meta', {}) # Default to empty if 'meta' is missing
        self.rootlaw.enforce(meta_for_enforce)

    def _monitor_loop(self):
        loop_interval = 5 # seconds
        victor_log("DEBUG", f"AGI {self.instance_id} monitor loop started (interval: {loop_interval}s).")
        time.sleep(loop_interval) # Initial delay before first cycle
        while not self.monitor_stop_event.is_set():
            try:
                self._enforce_law()
                self.genesis_light.homeostasis()
                # self.genesis_light.update_phase() # Homeostasis already calls update_phase
                self.genesis_light.on_phase_event()
                if len(self.lighthive.nodes) > 1:
                    self.lighthive.synchronize()
                    for node in self.lighthive.nodes:
                        if node is not self.genesis_light: node.homeostasis(); node.on_phase_event()
                # victor_log("DEBUG", f"AGI {self.instance_id} monitor cycle. GL Coherence: {self.genesis_light.coherence_score():.3f}")
            except Exception as e: victor_log("ERROR", f"Error in AGI {self.instance_id} monitor loop: {e}\n{traceback.format_exc()}")
            if self.monitor_stop_event.wait(loop_interval): break # Wait for interval or stop event
        victor_log("INFO", f"AGI {self.instance_id} monitor loop stopped.")


    def prompt(self, user_input):
        self._enforce_law()
        victor_log("INFO", f"AGI {self.instance_id} prompt: '{user_input[:30]}...'")
        result = self.cortex.process(user_input)
        victor_log("INFO", f"AGI {self.instance_id} processed. Decision: {result.get('decision')}")
        return result

    def evolve(self):
        self._enforce_law(); victor_log("INFO", f"AGI {self.instance_id} evolving...")
        self.genesis_light.morph("plasma", scale=1.05, entropy=np.clip(self.genesis_light.entropy*0.8,0.001,1.0))
        evo_count = self.state.get_var('evolution_count',0)+1
        self.state.set_var('evolution_count', evo_count)
        self.state.get_full_state()['meta']['version'] += f".e{evo_count}"
        self.state.save_state("AGI Evolution complete.")
        victor_log("INFO", f"AGI {self.instance_id} evolution complete. Version: {self.state.get_full_state()['meta']['version']}")

    def info(self):
        return {"instance_id":self.instance_id, "rootlaw_status":"ENFORCED", "genesis_light":self.genesis_light.info(),
                "state_summary":{"timeline":self.state.current_timeline, "timelines":len(self.state.timelines), "history_len":len(self.state.history), "meta":self.state.get_full_state().get('meta')},
                "hive_coherence":self.lighthive.global_coherence(), "replicas":len(self.replicas), "has_gui":self.has_gui}

    def shutdown(self):
        victor_log("CRITICAL", f"AGI {self.instance_id} shutting down...")
        self.monitor_stop_event.set()
        if self.monitor_thread.is_alive(): self.monitor_thread.join(timeout=max(0.1, self.monitor_thread.interval if hasattr(self.monitor_thread, 'interval') else 2.5)) # Use loop_interval if possible
        for rep in self.replicas: victor_log("INFO", f"Shutting down replica {rep.instance_id}"); rep.shutdown()
        self.state.save_state("Final state before shutdown.")
        self.state.export_state(f"victor_agi_{self.instance_id}_final_state.pkl")
        victor_log("CRITICAL", f"AGI {self.instance_id} shutdown complete.")

# Alias after the full class definition
_OriginalVictorAGIMonolith = VictorAGIMonolith

def trigger_self_replication(light_instance, agi_instance): # Global callback
    if not agi_instance or not isinstance(agi_instance, _OriginalVictorAGIMonolith): # Check against the aliased original
        victor_log("ERROR", "Replication: Invalid AGI instance."); return
    victor_log("CRITICAL", f"COHERENCE PEAK in Light {light_instance!r} (Coherence: {light_instance.coherence_score():.4f}). Triggering Self-Replication of AGI: {agi_instance.instance_id}")
    temp_state_file = f"temp_repl_state_{agi_instance.instance_id}_{generate_id()}.pkl"
    if not agi_instance.state.export_state(temp_state_file):
        victor_log("ERROR", "Replication failed: Parent state export failed."); return
    replica_cfg_overrides = {"gui_enabled_at_init": False, "log_level": os.environ.get('VICTOR_REPLICA_LOG_LEVEL', "INFO")}
    new_agi = VictorAGIMonolith(config_overrides=replica_cfg_overrides, start_gui=False) # This will use the mocked VictorAGIMonolith in test
    if not new_agi.state.import_state(temp_state_file):
        victor_log("ERROR", "Replication failed: Replica state import failed."); os.remove(temp_state_file); return
    if os.path.exists(temp_state_file): os.remove(temp_state_file)
    new_agi.state.set_var("is_replica", True); new_agi.state.set_var("parent_instance_id", agi_instance.instance_id)
    new_agi.state.set_var("replication_timestamp", time.time())
    new_agi.state.get_full_state()['meta']['instance_id'] = new_agi.instance_id
    new_agi.state.get_full_state()['meta']['version'] += "-replica"
    new_agi.state.save_state("State initialized from parent replication.")
    agi_instance.replicas.append(new_agi) # agi_instance is the real parent AGI
    victor_log("CRITICAL", f"Self-Replication SUCCESS. New AGI: {new_agi.instance_id}. Parent: {agi_instance.instance_id}. Total replicas: {len(agi_instance.replicas)}")

# =============================================================
# 6. GUI COMMAND CENTER
# =============================================================
class VictorGUI:
    def __init__(self, agi: VictorAGIMonolith):
        if not tk_available: victor_log("ERROR", "VictorGUI: Tkinter not available."); return
        self.agi, self.agi.has_gui = agi, True
        self.root = tk.Tk(); self.root.title(f"VICTOR AGI ({self.agi.instance_id})"); self.root.geometry("700x500")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing_gui)
        self._setup_ui(); self.log_gui(f"GUI for AGI {self.agi.instance_id} initialized.")

    def _setup_ui(self):
        mf = ttk.Frame(self.root, padding="5"); mf.pack(expand=True,fill=tk.BOTH)
        lf = ttk.LabelFrame(mf,text="Log",padding="3"); lf.pack(expand=True,fill=tk.BOTH,pady=2)
        self.log_txt = scrolledtext.ScrolledText(lf, height=15, width=80, bg="#1c1c1c", fg="#ccc", state=tk.DISABLED)
        self.log_txt.pack(expand=True,fill=tk.BOTH)
        inf = ttk.Frame(mf,padding="3"); inf.pack(fill=tk.X,pady=2)
        self.in_entry = ttk.Entry(inf,width=60,font=('Arial',9)); self.in_entry.pack(side=tk.LEFT,expand=True,fill=tk.X)
        self.in_entry.bind("<Return>", self._submit_evt)
        ttk.Button(inf,text="Send",command=self._submit_ui_input).pack(side=tk.LEFT,padx=3)
        cf = ttk.Frame(mf,padding="3"); cf.pack(fill=tk.X,pady=2)
        ttk.Button(cf,text="Info",command=self._show_info_popup).pack(side=tk.LEFT,padx=2)
        ttk.Button(cf,text="Evolve",command=self._evolve_cmd).pack(side=tk.LEFT,padx=2)
        ttk.Button(cf,text="Shutdown AGI",command=self._shutdown_cmd).pack(side=tk.RIGHT,padx=2)

    def log_gui(self, msg, level="INFO"):
        self.log_txt.configure(state=tk.NORMAL)
        self.log_txt.insert(tk.END,f"[{time.strftime('%T')}][{level}] {msg}\n"); self.log_txt.configure(state=tk.DISABLED)
        self.log_txt.see(tk.END)

    def _submit_evt(self, evt): self._submit_ui_input()
    def _submit_ui_input(self):
        prompt = self.in_entry.get();_ =prompt.strip() and (self.log_gui(f"YOU: {prompt}","INPUT"),self.in_entry.delete(0,tk.END),threading.Thread(target=self._process_thread,args=(prompt,),daemon=True).start())

    def _process_thread(self, prompt):
        try: res = self.agi.prompt(prompt)
        except Exception as e: res = {"response":f"ERROR: {e}", "decision":"EXCEPTION"}
        self.log_gui(f"VICTOR ({res.get('decision','N/A')}): {res.get('response','No textual response.')}", "AGI")

    def _show_info_popup(self): threading.Thread(target=lambda: messagebox.showinfo(f"AGI Info ({self.agi.instance_id})", json.dumps(self.agi.info(),indent=2,default=str)),daemon=True).start()
    def _evolve_cmd(self): messagebox.askyesno("Confirm Evolve","Evolve AGI?") and (self.log_gui("Evolve cmd.","CMD"), threading.Thread(target=self.agi.evolve,daemon=True).start())
    def _shutdown_cmd(self): messagebox.askyesno("Confirm Shutdown",f"Shutdown AGI {self.agi.instance_id}?") and (self.log_gui("Shutdown AGI cmd.","CMD"), threading.Thread(target=self.agi.shutdown,daemon=True).start(), self.root.after(1500,self._on_closing_gui))

    def _on_closing_gui(self):
        if self.agi and hasattr(self.agi,'monitor_stop_event') and not self.agi.monitor_stop_event.is_set():
            if messagebox.askokcancel("Quit",f"AGI {self.agi.instance_id} running. Shutdown AGI?"): self._shutdown_cmd(); return
        if self.agi: self.agi.has_gui = False
        self.root.destroy()

# =============================================================
# 7. MAIN BOOTLOADER
# =============================================================
if __name__ == "__main__":
    # VICTOR_CONFIG should be defined globally by now, or use a default for early log
    VICTOR_CONFIG.update({"core_name": "VictorBoot"}) # Update core_name for boot phase
    victor_log("INFO", "Victor AGI Monolith Bootloader sequence initiated.")
    victor_log("INFO", f"Headless Mode: {VICTOR_HEADLESS_MODE}, Tkinter Available: {tk_available}")

    should_start_gui = tk_available and not VICTOR_HEADLESS_MODE
    main_config_overrides = {"gui_enabled_at_init": should_start_gui}
    if VICTOR_HEADLESS_MODE: main_config_overrides["log_level"] = os.environ.get('VICTOR_TEST_LOG_LEVEL', "INFO")

    # Update global VICTOR_CONFIG before AGI instantiation if overrides affect it directly
    VICTOR_CONFIG.update(main_config_overrides)

    main_agi = VictorAGIMonolith(config_overrides=None, start_gui=should_start_gui) # Config already updated

    if main_agi.gui_instance and main_agi.gui_instance.root:
        victor_log("INFO", "Entering Tkinter main event loop.")
        try: main_agi.gui_instance.root.mainloop()
        except Exception as e: victor_log("CRITICAL", f"GUI mainloop error: {e}\n{traceback.format_exc()}")
        finally:
            victor_log("INFO", "GUI mainloop exited. Ensuring AGI shutdown if not already handled.")
            if hasattr(main_agi, 'monitor_stop_event') and not main_agi.monitor_stop_event.is_set(): main_agi.shutdown()
    elif should_start_gui and not (main_agi.gui_instance and main_agi.gui_instance.root):
        victor_log("ERROR", "GUI intended but failed to init. AGI running headless.")
        print(f"Victor AGI {main_agi.instance_id} running headless (GUI init failed).")
        # Headless keep-alive loop
        try:
            while hasattr(main_agi, 'monitor_stop_event') and not main_agi.monitor_stop_event.is_set(): time.sleep(1)
        except KeyboardInterrupt: victor_log("INFO", "Headless KI. Shutting down.")
        finally:
            if hasattr(main_agi, 'monitor_stop_event') and not main_agi.monitor_stop_event.is_set(): main_agi.shutdown()
    else: # Intended headless
        victor_log("INFO", f"Victor AGI {main_agi.instance_id} running headless.")
        print(f"Victor AGI {main_agi.instance_id} running headless. Monitor logs.")
        try:
            while hasattr(main_agi, 'monitor_stop_event') and not main_agi.monitor_stop_event.is_set(): time.sleep(1)
        except KeyboardInterrupt: victor_log("INFO", "Headless KI. Shutting down.")
        finally:
            if hasattr(main_agi, 'monitor_stop_event') and not main_agi.monitor_stop_event.is_set(): main_agi.shutdown()
    victor_log("INFO", "Victor AGI Monolith bootloader sequence complete.")

# Note: hash_data function is called in GodTierCortex but not defined in this provided code.
# This will cause a NameError if that part of the code is executed.
# For the purpose of this subtask (replacing content), I am including the code as-is.
# If the definition of hash_data is available, it should be added.
# For now, I'll add a dummy one to prevent immediate runtime error if that code path is hit.

def hash_data(data_to_hash):
    """Placeholder for a proper hashing function."""
    try:
        return hashlib.sha256(str(data_to_hash).encode('utf-8')).hexdigest()
    except Exception as e:
        victor_log("WARNING", f"hash_data placeholder failed: {e}")
        return "dummy_hash_error"
