import unittest
import os
import time
import copy
import collections # Added import
import numpy as np

from victor_agi_monolith import (
    FractalState,
    VictorAGIMonolith,
    TheLight,
    trigger_self_replication,
    victor_log,
    generate_id,
    VICTOR_CONFIG,
    _tkinter_available
)

VICTOR_TEST_CONFIG_OVERRIDE = {"log_level": "ERROR"}

def apply_test_config(base_config, test_overrides):
    config = copy.deepcopy(base_config)
    config.update(test_overrides)
    return config

# Define MockAGI at module level for accessibility
class MockAGI:
    def __init__(self, config):
        self.config = config
        self.instance_id = "mock_agi_for_fs_test"
        self._internal_state_for_capture = {
            'timestamp': time.time(),
            'version': self.config.get('version', 'N/A'),
            'config_snapshot': copy.deepcopy(self.config),
            'system_status': 'testing',
            'current_timeline_name': 'genesis',
            'instance_id': self.instance_id,
            'parent_id': None,
            "vars": {}
        }

    def get_full_state_snapshot(self):
        return copy.deepcopy(self._internal_state_for_capture)

    def apply_full_state_snapshot(self, snapshot_data):
        self._internal_state_for_capture = copy.deepcopy(snapshot_data)

    def set_test_var(self, key, value):
            if "vars" not in self._internal_state_for_capture: self._internal_state_for_capture["vars"] = {}
            self._internal_state_for_capture["vars"][key] = value

class TestFractalState(unittest.TestCase):

    def setUp(self):
        self.test_config = apply_test_config(VICTOR_CONFIG, VICTOR_TEST_CONFIG_OVERRIDE)
        self.mock_agi = MockAGI(self.test_config)
        self.fs = FractalState(self.mock_agi, self.mock_agi.get_full_state_snapshot, max_history=1000)
        self.initial_state_desc = "Genesis"

    def test_initial_state_and_history(self):
        self.assertEqual(len(self.fs.history), 1)
        self.assertEqual(self.fs.history[0]["desc"], self.initial_state_desc)
        self.assertEqual(self.fs.current_timeline, "genesis")
        self.assertIn("genesis", self.fs.timelines)
        self.assertEqual(len(self.fs.timelines["genesis"]), 1)
        self.assertEqual(self.fs.timelines["genesis"].maxlen, 1000)

    def test_save_state(self):
        initial_history_len = len(self.fs.history)
        self.mock_agi.set_test_var("test_var", 123)
        self.fs.save_state("Set test_var")

        self.assertEqual(len(self.fs.history), initial_history_len + 1)
        self.assertEqual(self.fs.history[-1]["state"]["vars"]["test_var"], 123)
        self.assertEqual(len(self.fs.timelines[self.fs.current_timeline]), initial_history_len + 1)

    def test_fork_timeline(self):
        self.mock_agi.set_test_var("genesis_var", "present")
        self.fs.save_state("Added genesis_var")

        source_timeline_name = self.fs.current_timeline
        source_history_len = len(self.fs.timelines[source_timeline_name])

        self.assertTrue(self.fs.fork_timeline("branch1"))
        self.assertEqual(self.fs.current_timeline, "branch1")
        self.assertIn("branch1", self.fs.timelines)

        self.assertEqual(len(self.fs.timelines["branch1"]), source_history_len + 1)
        self.assertEqual(len(self.fs.history), source_history_len + 1)

        self.assertIsNot(self.fs.timelines["branch1"], self.fs.timelines[source_timeline_name])
        self.assertEqual(self.fs.timelines["branch1"][-2]["state"]["vars"]["genesis_var"], "present")

        self.mock_agi.set_test_var("branch_var", "only_on_branch")
        self.fs.save_state("Added branch_var")

        self.assertEqual(len(self.fs.timelines[source_timeline_name]), source_history_len)
        self.assertNotIn("branch_var", self.fs.timelines[source_timeline_name][-1]["state"]["vars"])

    def test_switch_timeline(self):
        self.mock_agi.set_test_var("val_genesis", 0)
        self.fs.save_state("genesis save")

        self.fs.fork_timeline("branch1")
        self.mock_agi.set_test_var("val_branch1", 1)
        self.fs.save_state("branch1 save")

        self.assertTrue(self.fs.switch_timeline("genesis"))
        self.assertEqual(self.fs.current_timeline, "genesis")
        self.assertNotIn("val_branch1", self.fs.agi._internal_state_for_capture["vars"])
        self.assertEqual(self.fs.agi._internal_state_for_capture["vars"]["val_genesis"], 0)
        self.assertEqual(len(self.fs.history), 2)
        self.assertEqual(self.fs.history[-1]["state"]["vars"]["val_genesis"], 0)

        self.assertTrue(self.fs.switch_timeline("branch1"))
        self.assertEqual(self.fs.current_timeline, "branch1")
        self.assertEqual(self.fs.agi._internal_state_for_capture["vars"]["val_branch1"], 1)
        self.assertEqual(len(self.fs.history), 4)
        self.assertEqual(self.fs.history[-1]["state"]["vars"]["val_branch1"], 1)

    def test_switch_to_new_timeline(self):
        self.mock_agi.set_test_var("pre_switch_var", "exists")
        self.fs.save_state("Before new timeline switch")

        self.assertTrue(self.fs.switch_timeline("branch_new"))
        self.assertEqual(self.fs.current_timeline, "branch_new")
        self.assertIn("branch_new", self.fs.timelines)
        self.assertEqual(len(self.fs.timelines["branch_new"]), 1)
        self.assertEqual(len(self.fs.history), 1)
        self.assertTrue("Initiated new timeline: branch_new" in self.fs.history[0]["desc"])
        self.assertEqual(self.fs.history[0]["state"]["vars"]["pre_switch_var"], "exists")
        self.assertEqual(self.fs.agi._internal_state_for_capture["vars"]["pre_switch_var"], "exists")

    def test_export_import_state(self):
        test_file = "test_victor_state.pkl"
        self.mock_agi.set_test_var("export_test_var", "export_value")
        self.fs.save_state("PreExportSave1")
        self.fs.fork_timeline("export_branch")
        self.mock_agi.set_test_var("export_branch_var", "export_branch_value")
        self.fs.save_state("ExportBranchSave")

        original_current_timeline = self.fs.current_timeline
        original_timelines_data = {n: list(d) for n, d in self.fs.timelines.items()}
        original_timelines_maxlens = {n: d.maxlen for n, d in self.fs.timelines.items()}
        original_history_data = list(self.fs.history)
        original_history_maxlen = self.fs.history.maxlen
        original_full_state = copy.deepcopy(self.fs.agi._internal_state_for_capture)

        export_result = self.fs.export_state(test_file)
        self.assertTrue(export_result)

        new_mock_agi = MockAGI(self.test_config)
        new_fs = FractalState(new_mock_agi, new_mock_agi.get_full_state_snapshot, max_history=original_history_maxlen)

        self.assertTrue(new_fs.import_state(test_file))

        self.assertEqual(new_fs.current_timeline, original_current_timeline)
        self.assertEqual(new_fs.history.maxlen, original_history_maxlen)

        self.assertEqual(set(new_fs.timelines.keys()), set(original_timelines_data.keys()))
        for tl_name in original_timelines_data:
            self.assertEqual(new_fs.timelines[tl_name].maxlen, original_timelines_maxlens[tl_name])
            self.assertEqual(list(new_fs.timelines[tl_name]), original_timelines_data[tl_name])

        self.assertEqual(list(new_fs.history), original_history_data)
        self.assertEqual(new_fs.agi._internal_state_for_capture, original_full_state)

        if os.path.exists(test_file): os.remove(test_file)

    def test_fractal_memory_replay_basic(self):
        self.mock_agi.set_test_var("event1", "alpha"); self.fs.save_state("alpha event")
        time.sleep(0.001)
        self.mock_agi.set_test_var("event2", "beta"); self.fs.save_state("beta event")

        replayed_events = self.fs.fractal_memory_replay(depth_percent=1.0) # Explicitly ask for all
        self.assertEqual(len(replayed_events), 3)
        self.assertEqual(replayed_events[-1]["description"], "beta event")

    def test_fractal_memory_replay_depth_percent(self):
        for i in range(10):
            self.mock_agi.set_test_var(f"event{i}", i); self.fs.save_state(f"State {i}")
            time.sleep(0.001)

        replayed_events = self.fs.fractal_memory_replay(depth_percent=0.2)
        self.assertEqual(len(replayed_events), 3)
        self.assertEqual(replayed_events[-1]["description"], "State 9")
        self.assertEqual(replayed_events[0]["description"], "State 7")

        replayed_all = self.fs.fractal_memory_replay(depth_percent=1.0)
        self.assertEqual(len(replayed_all), 11)
        self.assertEqual(replayed_all[0]["description"], "Genesis")

        replayed_zero = self.fs.fractal_memory_replay(depth_percent=0.0)
        self.assertEqual(len(replayed_zero), 0)

    def test_fractal_memory_replay_keywords(self):
        self.fs.save_state("UniqueKeyword Alpha event")
        self.fs.save_state("Another event without keyword")
        self.fs.save_state("Event with UniqueKeyword Beta")

        replayed_events = self.fs.fractal_memory_replay(event_filter_keywords=["UniqueKeyword"], depth_percent=1.0)
        self.assertEqual(len(replayed_events), 2)
        self.assertTrue(all("UniqueKeyword" in event["description"] for event in replayed_events))
        self.assertTrue("Alpha" in replayed_events[0]["description"])
        self.assertTrue("Beta" in replayed_events[1]["description"])

    def test_fractal_memory_replay_on_forked_timeline(self):
        self.fs.save_state("genesis_specific_event")
        self.fs.fork_timeline("replay_branch")
        self.fs.save_state("branch_specific_event_1")
        self.fs.save_state("branch_specific_event_2_keyword")

        replayed_branch = self.fs.fractal_memory_replay(event_filter_keywords=["branch_specific"], depth_percent=1.0)
        self.assertEqual(len(replayed_branch), 2)
        self.assertEqual(replayed_branch[0]["description"], "branch_specific_event_1")

        replayed_genesis = self.fs.fractal_memory_replay(timeline_name="genesis", event_filter_keywords=["genesis_specific"], depth_percent=1.0)
        self.assertEqual(len(replayed_genesis), 1)
        self.assertEqual(replayed_genesis[0]["description"], "genesis_specific_event")

    def test_fractal_memory_replay_empty_or_bad_timeline(self):
        self.assertEqual(self.fs.fractal_memory_replay(timeline_name="non_existent_timeline"), [])
        self.fs.timelines["empty_timeline"] = collections.deque(maxlen=100)
        self.assertEqual(self.fs.fractal_memory_replay(timeline_name="empty_timeline"), [])

class TestSelfReplication(unittest.TestCase):

    def setUp(self):
        self.agi_parent = VictorAGIMonolith(config_overrides=VICTOR_TEST_CONFIG_OVERRIDE, start_gui=False)
        self.assertIsNotNone(getattr(self.agi_parent, 'genesis_light', None))
        self.assertIsNotNone(self.agi_parent.genesis_light.on_phase_event_handler)
        self.agi_parent.genesis_light.on_phase_event_handler["threshold"] = 0.97
        self.agi_parent.genesis_light.on_phase_event_handler["fire_once"] = False

    def test_replication_triggered_and_replica_properties(self):
        initial_replica_count = len(self.agi_parent.replicas)

        original_coherence_score_method = self.agi_parent.genesis_light.coherence_score
        self.agi_parent.genesis_light.coherence_score = lambda: 0.99

        self.agi_parent.genesis_light.on_phase_event()

        self.assertEqual(len(self.agi_parent.replicas), initial_replica_count + 1)
        replica_agi = self.agi_parent.replicas[-1]

        self.assertIsInstance(replica_agi, VictorAGIMonolith)
        replica_state = replica_agi.fractal_state_engine._state
        self.assertTrue(replica_state.get("custom_properties", {}).get("is_replica"))
        self.assertEqual(replica_state.get("custom_properties", {}).get("parent_instance_id"), self.agi_parent.instance_id)
        self.assertNotEqual(replica_agi.instance_id, self.agi_parent.instance_id)
        self.assertFalse(replica_agi.has_gui)
        self.assertTrue("-R" in replica_agi.config["core_name"])

        self.agi_parent.genesis_light.coherence_score = original_coherence_score_method

    def test_replica_state_is_independent_copy(self):
        if "custom_properties" not in self.agi_parent.fractal_state_engine._state:
            self.agi_parent.fractal_state_engine._state["custom_properties"] = {}
        self.agi_parent.fractal_state_engine._state["custom_properties"]["original_val"] = "parent_value_123"
        self.agi_parent.fractal_state_engine.save_state("Parent pre-replication save")

        original_coherence_score_method = self.agi_parent.genesis_light.coherence_score
        self.agi_parent.genesis_light.coherence_score = lambda: 0.99

        self.agi_parent.genesis_light.on_phase_event()

        self.assertTrue(len(self.agi_parent.replicas) > 0)
        replica_agi = self.agi_parent.replicas[-1]

        replica_live_state_after_import = replica_agi.fractal_state_engine._state
        self.assertEqual(replica_live_state_after_import.get("custom_properties", {}).get("original_val"), "parent_value_123")

        if "custom_properties" not in replica_agi.fractal_state_engine._state:
            replica_agi.fractal_state_engine._state["custom_properties"] = {}
        replica_agi.fractal_state_engine._state["custom_properties"]["original_val"] = "replica_modified_456"
        replica_agi.fractal_state_engine.save_state("Replica modified val")

        parent_live_state = self.agi_parent.fractal_state_engine._state
        self.assertEqual(parent_live_state.get("custom_properties", {}).get("original_val"), "parent_value_123")

        replica_final_live_state = replica_agi.fractal_state_engine._state
        self.assertEqual(replica_final_live_state.get("custom_properties", {}).get("original_val"), "replica_modified_456")

        self.agi_parent.genesis_light.coherence_score = original_coherence_score_method

    def tearDown(self):
        for item in os.listdir("."):
            if item.startswith("temp_replication_state_") and item.endswith(".pkl"):
                try:
                    os.remove(item)
                    print(f"Cleaned up temp file: {item}")
                except OSError as e:
                    print(f"Error cleaning up temp file {item}: {e}")

if __name__ == "__main__":
    unittest.main()
