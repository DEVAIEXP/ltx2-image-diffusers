import unittest

from custom_blocks.ltx2_image.memory.dynamic_weights import (
    DYNAMIC_WEIGHTS_PRESETS,
    format_dynamic_weights_presets,
    get_dynamic_weights_presets,
    load_dynamic_weights_settings_from_env,
    resolve_dynamic_weights_preset,
)


class DynamicWeightsPresetTests(unittest.TestCase):
    def test_auto_resolves_to_one_shot_fast(self):
        self.assertEqual(resolve_dynamic_weights_preset("auto", running_on_wsl=False), "one_shot_fast")
        self.assertEqual(resolve_dynamic_weights_preset("auto", running_on_wsl=True), "one_shot_fast")

    def test_invalid_preset_reports_canonical_env_name(self):
        with self.assertRaisesRegex(ValueError, "DIFFUSERS_DYNAMIC_WEIGHTS_PRESET"):
            resolve_dynamic_weights_preset("windows_fast", running_on_wsl=False)

    def test_presets_do_not_include_removed_legacy_names(self):
        removed_aliases = {
            "windows_fast",
            "linux_native_fast",
            "linux_safe",
            "planner_balanced",
            "planner_slim_resident",
            "warm_server",
            "low_ram",
            "diffusers_group_offload",
            "group_offload_compat",
            "diffusers_leaf_group_offload",
            "leaf_offload_compat",
        }
        self.assertFalse(removed_aliases.intersection(DYNAMIC_WEIGHTS_PRESETS))

    def test_env_override_wins_over_preset_value(self):
        settings = load_dynamic_weights_settings_from_env(
            running_on_wsl=False,
            environ={
                "DIFFUSERS_DYNAMIC_WEIGHTS_PRESET": "one_shot_fast",
                "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "9",
                "DIFFUSERS_DYNAMIC_WEIGHTS_AVAILABLE_SYSTEM_RAM_GB": "64",
            },
        )
        self.assertEqual(settings.effective_preset, "one_shot_fast")
        self.assertEqual(settings.config.pin_cpu_workers, 9)
        self.assertEqual(settings.config.available_system_ram_gb, 64.0)

    def test_wsl_disables_pinned_memory_by_default(self):
        settings = load_dynamic_weights_settings_from_env(
            running_on_wsl=True,
            environ={
                "DIFFUSERS_DYNAMIC_WEIGHTS_PRESET": "one_shot_fast",
                "DIFFUSERS_DYNAMIC_WEIGHTS_AVAILABLE_SYSTEM_RAM_GB": "64",
            },
        )
        self.assertTrue(settings.requested_pin_cpu_memory)
        self.assertFalse(settings.effective_pin_cpu_memory)

    def test_preset_listing_is_a_copy(self):
        presets = get_dynamic_weights_presets()
        presets["one_shot_fast"]["DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS"] = "99"
        self.assertNotEqual(
            DYNAMIC_WEIGHTS_PRESETS["one_shot_fast"]["DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS"],
            "99",
        )

    def test_format_presets_lists_default_mapping(self):
        formatted = format_dynamic_weights_presets(default_preset="auto", running_on_wsl=False)
        self.assertIn("auto -> one_shot_fast", formatted)
        self.assertIn("one_shot_fast", formatted)
        self.assertIn("diffusers_offload_compat", formatted)


if __name__ == "__main__":
    unittest.main()
