import unittest
from unittest import mock # Added import
import os
import time
import copy
import numpy as np # Add this import
from victor_agi_monolith import FractalState, VictorAGIMonolith, TheLight, trigger_self_replication, victor_log, generate_id, VICTOR_CONFIG

# Suppress most victor_log output during tests to keep test output clean
# We will override VICTOR_CONFIG when creating AGI instances for tests.
VICTOR_CONFIG_TEST_OVERRIDE = {"log_level": "ERROR"}


# Mock AGI instance for FractalState tests if real AGI is too heavy
class MockAGIForFractalState:
    def __init__(self, config_overrides=None):
        self.config = copy.deepcopy(VICTOR_CONFIG)
        if config_overrides:
            self.config.update(config_overrides)

        # Mock essential components that FractalState might interact with if it calls back to AGI
        # For the current FractalState, it needs get_full_state_snapshot and apply_full_state_snapshot
        self.memory = mock.MagicMock() # Changed unittest.mock to mock
        self.task_manager = mock.MagicMock() # Changed unittest.mock to mock
        self.emotional_core = mock.MagicMock() # Changed unittest.mock to mock
        self.attention_system = mock.MagicMock() # Changed unittest.mock to mock
        self.instance_id = generate_id("mock_agi_")

        # Mock internal state that might be part of a snapshot
        self._internal_mock_vars = {"initial_mock_var": "mock_value"}

    def get_full_state_snapshot(self):
        # This should return a structure similar to what the real AGI returns
        return {
            "timestamp": time.time(),
            "version": self.config.get("version", "test_version"),
            "config_snapshot": copy.deepcopy(self.config),
            "system_status": "testing",
            "mock_internal_vars_snapshot": copy.deepcopy(self._internal_mock_vars),
            # Add other fields if FractalState's _capture_current_agi_state_snapshot expects them
            # from a real AGI, or if _apply_agi_state_from_snapshot tries to restore them.
        }

    def apply_full_state_snapshot(self, snapshot):
        # This should restore parts of the AGI state from a snapshot
        # For mock, we can just log or store the applied snapshot for inspection
        self.applied_snapshot = snapshot
        if "mock_internal_vars_snapshot" in snapshot:
            self._internal_mock_vars = copy.deepcopy(snapshot["mock_internal_vars_snapshot"])
        victor_log("DEBUG", f"MockAGI apply_full_state_snapshot called with snapshot from timestamp {snapshot.get('timestamp')}", component_name="MockAGI")

    # Helper for tests to simulate changing AGI state that FractalState would capture
    def set_mock_internal_var(self, key, value):
        self._internal_mock_vars[key] = value

    def get_mock_internal_var(self, key):
        return self._internal_mock_vars.get(key)


class TestFractalState(unittest.TestCase):

    def setUp(self):
        self.mock_agi_config = copy.deepcopy(VICTOR_CONFIG)
        self.mock_agi_config.update(VICTOR_CONFIG_TEST_OVERRIDE)
        # FractalState is initialized with a config object
        self.fs = FractalState(agi_config=self.mock_agi_config)
        # The FractalState __init__ already saves "Genesis: Initial state created."
        self.initial_state_desc = "Genesis: Initial state created."

    def test_initial_state_and_history(self):
        self.assertEqual(len(self.fs.history), 1, "History should have one entry after init.")
        self.assertEqual(self.fs.history[0]["desc"], self.initial_state_desc) # Changed "description" to "desc"
        self.assertEqual(self.fs.current_timeline, "genesis")
        self.assertIn("genesis", self.fs.timelines)
        self.assertEqual(len(self.fs.timelines["genesis"]), 1)
        # The state saved by FractalState is its own internal _state, not a snapshot from an AGI object.
        # We need to check for expected keys in self.fs.history[0]["state"]
        self.assertIn("vars", self.fs.history[0]["state"])
        self.assertIn("meta", self.fs.history[0]["state"])
        self.assertEqual(self.fs.history[0]["state"]["meta"]["version"], self.mock_agi_config.get("version", "fs_default"))


    def test_save_state(self):
        initial_history_len = len(self.fs.history)
        # Modify FractalState's internal state directly
        self.fs.set_var("test_var", 123)
        # save_state is called automatically by set_var if autosave is on (default)
        # If we want to test save_state explicitly with a description:
        # self.fs.save_state("State with test_var") # This would be an additional save if autosave is on.
        # For this test, let's rely on set_var's implicit save_state.

        # If autosave is on, set_var already saved. If not, this test needs adjustment
        # Assuming autosave is ON by default in FractalState's config.
        # The description of the auto-save from set_var will be like "SetVar: test_var = 123"

        self.assertEqual(len(self.fs.history), initial_history_len + 1, "History should increment after set_var (autosave).")

        saved_entry = self.fs.history[-1]
        self.assertEqual(saved_entry["desc"], "SetVar: test_var = 123")
        self.assertEqual(saved_entry["state"]["vars"]["test_var"], 123)
        self.assertEqual(len(self.fs.timelines[self.fs.current_timeline]), initial_history_len + 1)

    def test_fork_timeline(self):
        self.fs.set_var("genesis_var", "present_on_genesis") # Autosaved
        # self.fs.save_state("Before fork on genesis") # Explicit save if needed, covered by set_var

        self.assertTrue(self.fs.fork_timeline("branch1"))
        # After forking, current_timeline *is* switched to the new timeline ("branch1") by current FractalState.fork_timeline impl.

        self.assertEqual(self.fs.current_timeline, "branch1")
        self.assertIn("branch1", self.fs.timelines)
        # Fork copies history, then adds a "Timeline ... forked to ..." state.
        # So, length of branch1 will be length of genesis (at time of fork) + 1.
        # Let's find the state *before* the fork on genesis to compare length.
        genesis_history_before_fork_len = 0
        for entry in self.fs.timelines["genesis"]:
            if entry["desc"] == "SetVar: genesis_var = present_on_genesis":
                genesis_history_before_fork_len = self.fs.timelines["genesis"].index(entry) + 1
                break
        # Or, more simply, branch1 starts with genesis's history, then adds "Timeline ... forked..."
        # So, len(branch1) = len(genesis_at_fork_point) + 1.
        # And genesis itself also has the "Timeline ... forked..." state if fork_timeline saves on source too (it does not, only on new)
        # The current `fork_timeline` saves state on the *new* timeline *after* copying.
        # So, len(branch1) should be len(genesis_at_fork_time) + 1 (for the fork event itself)
        # And len(genesis) remains unchanged by the fork operation itself.

        # Get genesis state just before fork was called
        genesis_state_desc_before_fork = "SetVar: genesis_var = present_on_genesis"
        genesis_len_at_fork = 0
        for i, entry in enumerate(self.fs.timelines["genesis"]):
            if entry["desc"] == genesis_state_desc_before_fork:
                genesis_len_at_fork = i + 1
                break
        self.assertGreater(genesis_len_at_fork, 0, "Genesis state before fork not found")

        self.assertEqual(len(self.fs.timelines["branch1"]), genesis_len_at_fork + 1, "Forked timeline should have copied history +1 for fork event.")

        self.assertIsNot(self.fs.timelines["branch1"], self.fs.timelines["genesis"], "History deques should be different objects.")
        # Verify content of the *copied* state on the new branch (second to last entry)
        self.assertEqual(self.fs.timelines["branch1"][-2]["state"]["vars"]["genesis_var"], "present_on_genesis")
        self.assertTrue(self.fs.timelines["branch1"][-1]["desc"].startswith("Timeline 'genesis' forked to 'branch1'"))


        # Modify state while on branch1 and save it
        self.fs.set_var("branch_var", "only_on_branch") # Autosaved on branch1

        # Check branch_var is in branch1's latest state
        self.assertEqual(self.fs.timelines["branch1"][-1]["state"]["vars"]["branch_var"], "only_on_branch")

        # Switch back to genesis and verify branch_var is not there
        self.fs.switch_timeline("genesis")
        # After switch, self.fs._state is loaded from genesis's latest.
        self.assertEqual(self.fs.get_var("genesis_var"), "present_on_genesis")
        self.assertIsNone(self.fs.get_var("branch_var"), "branch_var should not be in state when on genesis.")


    def test_switch_timeline(self):
        self.fs.set_var("val_genesis", 0) # Autosaved on genesis

        self.fs.fork_timeline("branch1") # Switches to branch1
        self.fs.set_var("val_branch1", 1) # Autosaved on branch1

        self.fs.switch_timeline("genesis") # Switch back to genesis
        self.assertEqual(self.fs.current_timeline, "genesis")
        self.assertEqual(self.fs.get_var("val_genesis"), 0)
        self.assertIsNone(self.fs.get_var("val_branch1"), "val_branch1 should not be in state when on genesis.")

        self.fs.switch_timeline("branch1") # Switch back to branch1
        self.assertEqual(self.fs.current_timeline, "branch1")
        self.assertEqual(self.fs.get_var("val_branch1"), 1)
        # self.assertIsNone(self.fs.get_var("val_genesis"), "val_genesis should not be in state when on branch1 (unless inherited and not overwritten).") # This was incorrect
        # Actually, after forking, branch1 will have val_genesis. Let's verify that.
        self.assertEqual(self.fs.get_var("val_genesis"), 0) # val_genesis was on genesis, so it's on branch1 too


    def test_export_import_state(self):
        test_file = "test_victor_fractal_state.pkl"
        self.fs.set_var("export_test_var", "export_value") # Autosaved on genesis timeline

        self.fs.fork_timeline("export_branch") # Switches to export_branch
        self.fs.set_var("export_branch_var", "export_branch_value") # Autosaved on export_branch

        original_current_timeline = self.fs.current_timeline # Should be "export_branch"
        # Get history lengths from actual saved data by FractalState
        # The number of items on genesis before fork:
        num_genesis_items_before_fork = 0
        for i, entry in enumerate(self.fs.timelines["genesis"]):
            if entry["desc"] == "SetVar: export_test_var = export_value":
                 num_genesis_items_before_fork = i + 1
                 break
        self.assertTrue(num_genesis_items_before_fork > 0)

        # The number of items on export_branch (it includes items from genesis + its own + fork event)
        num_export_branch_items = len(self.fs.timelines["export_branch"])


        original_genesis_maxlen = self.fs.timelines["genesis"].maxlen
        original_export_branch_maxlen = self.fs.timelines["export_branch"].maxlen

        export_result = self.fs.export_state(test_file)
        self.assertTrue(export_result)

        # New FractalState for import, using a new config obj
        new_mock_config = copy.deepcopy(VICTOR_CONFIG)
        new_mock_config.update(VICTOR_CONFIG_TEST_OVERRIDE)
        new_fs = FractalState(agi_config=new_mock_config)

        self.assertTrue(new_fs.import_state(test_file))

        self.assertEqual(new_fs.current_timeline, original_current_timeline, "Current timeline name mismatch after import.")
        self.assertIn("export_branch", new_fs.timelines, "Exported branch not found after import.")

        # Compare history lengths carefully
        # Genesis timeline in new_fs should have same content as in old fs's genesis
        self.assertEqual(len(new_fs.timelines["genesis"]), num_genesis_items_before_fork, "Genesis history length mismatch after import.")
        self.assertEqual(len(new_fs.timelines["export_branch"]), num_export_branch_items, "Export_branch history length mismatch after import.")

        # Check content of the latest state of the imported current timeline ("export_branch")
        imported_current_timeline_last_entry = new_fs.timelines[original_current_timeline][-1]
        self.assertEqual(imported_current_timeline_last_entry["state"]["vars"]["export_branch_var"], "export_branch_value")

        # Test history deque properties (maxlen)
        self.assertEqual(new_fs.timelines["genesis"].maxlen, original_genesis_maxlen)
        self.assertEqual(new_fs.timelines["export_branch"].maxlen, original_export_branch_maxlen)
        self.assertEqual(new_fs.history.maxlen, new_fs.timelines[new_fs.current_timeline].maxlen)


        if os.path.exists(test_file):
            os.remove(test_file)

    def test_fractal_memory_replay_basic(self):
        self.fs.set_var("event_marker", "alpha_event") # Autosaved with desc "SetVar: event_marker = alpha_event"
        time.sleep(0.01)
        self.fs.set_var("event_marker", "beta_event")  # Autosaved with desc "SetVar: event_marker = beta_event"

        replayed_events = self.fs.fractal_memory_replay(depth_percent=1.0) # Replay all

        # History: 1 (initial) + 1 (alpha) + 1 (beta) = 3
        self.assertEqual(len(replayed_events), 3)
        self.assertTrue(any("SetVar: event_marker = beta_event" in event["description"] for event in replayed_events))

        # Check snapshot content for "alpha_event"
        alpha_event_snapshot = None
        for event in replayed_events:
            if event["description"] == "SetVar: event_marker = alpha_event":
                # The snapshot is the full state, so check vars within it
                self.assertEqual(event["state_snapshot_summary"].get("vars"), "dict") # Example check, need to dig into actual state if needed
                # To check the actual value, we'd need to not use summary, or make summary more detailed.
                # For now, let's assume the presence of the description is enough.
                # A better check would be to retrieve the actual state from history and check 'vars'
                full_state_of_alpha_event = next(h["state"] for h in self.fs.history if h["desc"] == "SetVar: event_marker = alpha_event")
                self.assertEqual(full_state_of_alpha_event["vars"]["event_marker"], "alpha_event")

    def test_fractal_memory_replay_depth_percent(self):
        # Initial genesis state (1) + 10 saved states = 11 total states
        for i in range(10):
            self.fs.set_var("loop_event", i) # Autosaved
            if i < 9: time.sleep(0.001)

        replayed_events_20_percent = self.fs.fractal_memory_replay(depth_percent=0.2)
        total_events = 11
        expected_min_count = int(total_events * 0.2)
        if expected_min_count == 0 and 0.2 > 0 and total_events > 0: expected_min_count = 1

        self.assertGreaterEqual(len(replayed_events_20_percent), expected_min_count)
        self.assertLessEqual(len(replayed_events_20_percent), expected_min_count + 1 if expected_min_count > 0 else 1) # Allow for ceiling effect or min 1

        # Check if the latest events are present
        last_event_desc = "SetVar: loop_event = 9"
        self.assertTrue(any(last_event_desc in event["description"] for event in replayed_events_20_percent))
        if len(replayed_events_20_percent) > 1:
            second_last_event_desc = "SetVar: loop_event = 8"
            self.assertTrue(any(second_last_event_desc in event["description"] for event in replayed_events_20_percent))


    def test_fractal_memory_replay_keywords(self):
        self.fs.save_state("UniqueKeyword Alpha event replay test") # Explicit save for specific desc
        self.fs.set_var("some_other_var", "value") # Autosave: "SetVar: some_other_var = value"
        self.fs.save_state("Event with UniqueKeyword Beta for replay") # Explicit save

        replayed_events = self.fs.fractal_memory_replay(depth_percent=1.0, event_filter_keywords=["UniqueKeyword"])
        self.assertEqual(len(replayed_events), 2, "Should only find events with 'UniqueKeyword' in description")
        self.assertTrue(all("UniqueKeyword".lower() in event["description"].lower() for event in replayed_events))

    def test_fractal_memory_replay_on_forked_timeline(self):
        self.fs.save_state("genesis_specific_event for replay") # Explicit save
        self.fs.fork_timeline("replay_branch_test") # Switches to replay_branch_test
        self.fs.save_state("branch_specific_event_1 for replay") # Explicit save
        self.fs.save_state("branch_specific_event_2_keyword_replay") # Explicit save

        # Replay on current timeline (replay_branch_test)
        replayed_branch = self.fs.fractal_memory_replay(depth_percent=1.0, event_filter_keywords=["branch_specific"])
        self.assertEqual(len(replayed_branch), 2)
        self.assertTrue(any("event_1 for replay" in event["description"] for event in replayed_branch))
        self.assertTrue(any("event_2_keyword_replay" in event["description"] for event in replayed_branch))

        # Replay on genesis timeline
        replayed_genesis = self.fs.fractal_memory_replay(timeline_name="genesis", depth_percent=1.0, event_filter_keywords=["genesis_specific"])
        # History on genesis: initial + "genesis_specific_event..." = 2
        self.assertEqual(len(replayed_genesis), 1) # Filtered, so only 1
        self.assertTrue(any("genesis_specific_event for replay" in event["description"] for event in replayed_genesis))


class TestSelfReplication(unittest.TestCase):

    def setUp(self):
        # Create a parent AGI for testing replication. Ensure it's headless.
        # The VictorAGIMonolith __init__ needs to be callable without starting a full GUI loop.
        # The main script's if __name__ == "__main__": handles GUI loop.
        # So, direct instantiation should be fine for a headless AGI.
        # VICTOR_CONFIG_TEST_OVERRIDE is {"log_level": "ERROR"}
        # Global VICTOR_CONFIG will be updated by VictorAGIMonolith constructor
        self.agi_parent = VictorAGIMonolith(config_overrides=VICTOR_CONFIG_TEST_OVERRIDE, start_gui=False)


        # Ensure genesis_light and its handler are set up as expected by the monolith's init
        self.assertIsNotNone(getattr(self.agi_parent, 'genesis_light', None))
        self.assertIsNotNone(self.agi_parent.genesis_light.on_phase_event_handler)
        self.assertEqual(self.agi_parent.genesis_light.on_phase_event_handler["callback"], trigger_self_replication)
        self.agi_parent.has_gui = False # Explicitly ensure it's treated as headless for tests

    @mock.patch.object(TheLight, 'coherence_score')
    @mock.patch('victor_agi_monolith.VictorAGIMonolith') # Mock the class to control replica instances
    def test_replication_on_coherence_peak(self, MockVictorAGIReplica, mock_coherence_score):
        initial_replica_count = len(self.agi_parent.replicas)

        mock_coherence_score.return_value = 0.99 # High coherence to trigger replication

        # Configure the mock replica that will be created
        mock_replica_instance = mock.MagicMock(spec=VictorAGIMonolith)
        mock_replica_instance.instance_id = generate_id("replica_mock_")
        mock_replica_instance.state = mock.MagicMock(spec=FractalState) # Mock replica's state
        mock_replica_instance.state.import_state.return_value = True # Simulate successful import
        mock_replica_instance.state.get_full_state.return_value = {"meta": {"instance_id": mock_replica_instance.instance_id, "version":"test-replica"}} # for save_state
        MockVictorAGIReplica.return_value = mock_replica_instance

        # Parent AGI state to be copied
        parent_state_engine = self.agi_parent.state # Actual FractalState instance
        parent_state_engine.set_var("custom_prop_for_test", "parent_value")
        # The set_var above already saved the state. The export will use the latest state.

        # Trigger the event check.
        self.agi_parent.genesis_light.on_phase_event()

        self.assertEqual(len(self.agi_parent.replicas), initial_replica_count + 1, "Replica count should increment.")
        # The actual replica is now a MagicMock, so we check calls on it.
        actual_replica_object_in_list = self.agi_parent.replicas[-1]
        self.assertEqual(actual_replica_object_in_list, mock_replica_instance) # Ensure the mock was added

        # Verify that the replica's import_state was called (implies parent's export was attempted)
        mock_replica_instance.state.import_state.assert_called_once()

        # Verify that the replica's state was updated with replica markers
        # This involves checking calls to set_var on the mocked replica's state object
        expected_set_var_calls = [
            mock.call("is_replica", True),
            mock.call("parent_instance_id", self.agi_parent.instance_id),
            mock.call("replication_timestamp", mock.ANY) # Time is tricky to match exactly
        ]
        mock_replica_instance.state.set_var.assert_has_calls(expected_set_var_calls, any_order=False)

        # Verify the replica's meta info was updated (instance_id, version)
        # This would be a call to replica_instance.state.get_full_state()['meta'].update(...) or similar,
        # then replica_instance.state.save_state("State initialized from parent replication.")
        # The current mock_replica_instance.state.get_full_state() is basic.
        # To test this properly, state.get_full_state()['meta'] would need to be a real dict or more complex mock.
        # For now, let's check that save_state was called on the replica's state.
        mock_replica_instance.state.save_state.assert_called_with("State initialized from parent replication.")

        # Check that the original parent's custom variable is still there and unchanged.
        self.assertEqual(self.agi_parent.state.get_var("custom_prop_for_test"), "parent_value")

        # Clean up mock
        mock_coherence_score.reset_mock()
        MockVictorAGIReplica.reset_mock()

    def tearDown(self):
        if hasattr(self.agi_parent, 'shutdown'):
            self.agi_parent.shutdown() # Removed "test_teardown" argument
        for replica in self.agi_parent.replicas:
            # If replica is a mock, it might not have shutdown, or we can check if shutdown was called.
            if isinstance(replica, mock.MagicMock):
                if 'shutdown' in dir(replica): # Check if shutdown is a mocked method
                    replica.shutdown.assert_called_once_with() # Or appropriate args
            elif hasattr(replica, 'shutdown'):
                replica.shutdown() # Removed "test_teardown_replica"
        # VictorAGIMonolith.instance = None # This was from a singleton pattern not in current code.


if __name__ == "__main__":
    # Need to ensure that if VictorAGIMonolith is instantiated directly, it doesn't try to start a GUI.
    # The main script's __main__ block handles GUI. For tests, we want headless instances.

    # unittest.main() # This will run if the script is executed directly.
    # For use with test runners, this is often omitted or conditional.
    # Using argv and exit=False for compatibility with some environments/runners.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
