import os
import unittest
from unittest.mock import patch

from harness_code_agent.agent.prompts import (
    SHARED_AGENT_IDENTITY,
    GlobalRulesDoc,
    PromptPrefixBuilder,
)
from harness_code_agent.profiles import (
    PRODUCT_PROFILES,
    PROFILES,
    get_profile,
    list_profiles,
)
from harness_code_agent.profiles.router import (
    ROUTING_MODE_PINNED,
    LlmRouteResult,
    route_profile_for_turn,
)
from harness_code_agent.profiles.terminal import TerminalProfile
from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
from harness_code_agent.runtime.middleware import (
    TerminalShellEditPolicyMiddleware,
)
from harness_code_agent.runtime.tool_registry import tool_schemas_for_profile


class ProfilePromptTests(unittest.TestCase):
    def test_product_registry_hides_eval_only_terminal_profile(self):
        self.assertEqual(
            list(PROFILES),
            [
                "general",
                "coding-agent",
                "app-builder",
                "terminal",
                "plan",
                "review",
            ],
        )
        self.assertEqual(
            list(PRODUCT_PROFILES),
            [
                "general",
                "coding-agent",
                "app-builder",
                "plan",
                "review",
            ],
        )
        self.assertEqual([item["name"] for item in list_profiles()], list(PRODUCT_PROFILES))
        self.assertIsInstance(get_profile("terminal"), TerminalProfile)
        with self.assertRaisesRegex(ValueError, "Unknown profile: swe-bench"):
            get_profile("swe-bench")

    def test_terminal_profile_stays_outside_product_auto_routing(self):
        decision = route_profile_for_turn("please build a web app", current_profile="terminal")

        self.assertEqual(decision.profile_name, "terminal")
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.fallback_reason, "profile is sticky")

    def test_profile_router_prefers_explicit_workflow_contracts_over_similarity(self):
        cases = [
            ("先给我方案，不要修改代码", "plan"),
            ("只审查这个实现，不要改动文件", "review"),
            ("审查后直接修复这个 parser bug", "coding-agent"),
            ("帮我写一个计算器", "coding-agent"),
            ("给我创建一个霜叶转换器", "coding-agent"),
            ("写个排序函数", "coding-agent"),
            ("开发一个潮汐索引器", "coding-agent"),
            ("创建一个响应式网页看板", "app-builder"),
            ("解释这段代码是什么意思", "general"),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                decision = route_profile_for_turn(prompt, current_profile="general")
                self.assertEqual(decision.profile_name, expected)
                self.assertEqual(decision.reason, f"High-precision local contract matched {expected}.")
                self.assertEqual(decision.confidence, 0.98)
                self.assertFalse(decision.llm_called)

    def test_profile_router_keeps_specialized_profile_sticky_for_general_followup(self):
        decision = route_profile_for_turn("help me understand this concept", current_profile="coding-agent")

        self.assertEqual(decision.profile_name, "coding-agent")
        self.assertEqual(decision.action, "direct_answer")
        self.assertEqual(decision.matched_profile, "general")

    def test_pinned_profile_never_changes_from_semantic_similarity(self):
        decision = route_profile_for_turn(
            "先给我一个完整实施方案，不要修改代码",
            current_profile="coding-agent",
            routing_mode=ROUTING_MODE_PINNED,
        )

        self.assertEqual(decision.profile_name, "coding-agent")
        self.assertEqual(decision.action, "stay")
        self.assertEqual(decision.source, "pinned")
        self.assertEqual(decision.decisive_signal, "pinned")

    def test_model_routing_can_jump_between_specialized_profiles(self):
        decision = route_profile_for_turn(
            "take care of the parser task",
            current_profile="plan",
            llm_classifier=lambda **_: LlmRouteResult(
                profile_name="coding-agent",
                confidence=0.94,
                reason="Implementation requested.",
                provider="test",
                model="fast-test",
            ),
        )

        self.assertEqual(decision.profile_name, "coding-agent")
        self.assertEqual(decision.action, "switch_profile")
        self.assertEqual(decision.source, "llm")
        self.assertTrue(decision.llm_called)

    def test_explicit_mode_can_transition_and_pins_at_session_layer(self):
        decision = route_profile_for_turn(
            "切换到编码模式",
            current_profile="plan",
        )

        self.assertEqual(decision.profile_name, "coding-agent")
        self.assertEqual(decision.decisive_signal, "explicit_mode")
        self.assertEqual(decision.action, "switch_profile")

    def test_low_evidence_route_keeps_non_unit_confidence(self):
        decision = route_profile_for_turn(
            "嗯",
            current_profile="general",
            llm_classifier=lambda **_: LlmRouteResult(
                profile_name="general",
                confidence=0.41,
                reason="Ambiguous acknowledgement.",
            ),
        )

        self.assertLess(decision.confidence, 1.0)
        self.assertTrue(decision.fallback_used)

    def test_shared_identity_precedes_profile_contract_and_has_own_hash(self):
        prefix = PromptPrefixBuilder().build(
            profile_prompt="## Role\nA focused test profile.",
            global_rules_docs=[
                GlobalRulesDoc(source="HARNESS.md", content="Use focused checks.")
            ],
        )

        self.assertIn("## Agent Identity and Judgment", prefix.content)
        self.assertIn(SHARED_AGENT_IDENTITY, prefix.content)
        self.assertIn("## Profile Contract", prefix.content)
        self.assertLess(
            prefix.content.index("## Agent Identity and Judgment"),
            prefix.content.index("## Profile Contract"),
        )
        self.assertLess(
            prefix.content.index("## Profile Contract"),
            prefix.content.index("## Global Rules Bundle"),
        )
        self.assertNotIn("Acceptance Criteria", prefix.content)
        self.assertNotIn("acceptance_criteria_hash", prefix.hashes)
        self.assertIn("shared_identity_hash", prefix.hashes)

    def test_each_profile_has_role_working_style_boundaries_and_completion(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                prompt = get_profile(name).main_agent().system_prompt
                self.assertIn("## Role", prompt)
                self.assertIn("## Working Style", prompt)
                self.assertIn("## Boundaries", prompt)
                self.assertIn("## Completion", prompt)

    def test_profile_contracts_keep_their_distinctive_behavior(self):
        prompts = {
            name: get_profile(name).main_agent().system_prompt.lower()
            for name in PROFILES
        }

        self.assertIn("answer-first", prompts["general"])
        self.assertIn("existing design", prompts["coding-agent"])
        self.assertIn("decision-complete", prompts["plan"])
        self.assertIn("findings first", prompts["review"])
        self.assertIn("non-interactive", prompts["terminal"])
        self.assertIn("smallest suitable stack", prompts["app-builder"])

    def test_general_profile_does_not_expose_shell_batch_commands(self):
        cfg = get_profile("general").main_agent()
        tool_names = {
            schema["function"]["name"]
            for schema in tool_schemas_for_profile(
                allowed_permissions=cfg.allowed_tool_permissions,
                include_names=cfg.allowed_tool_names,
                exclude_names=cfg.blocked_tool_names,
                registry=BUILTIN_TOOL_REGISTRY,
            )
        }

        self.assertNotIn("parallel_commands", tool_names)
        self.assertNotIn("delegate_agent", tool_names)
        self.assertNotIn("parallel_agents", tool_names)
        self.assertNotIn("spawn_agent", tool_names)

    def test_execution_profiles_have_no_intervention_middlewares_or_hard_timeout(self):
        for name in ("coding-agent", "app-builder"):
            with self.subTest(profile=name):
                cfg = get_profile(name).main_agent()

                self.assertEqual(cfg.middlewares, [])
                self.assertIsNone(cfg.time_budget)

                prompt = cfg.system_prompt
                self.assertIn("non-trivial multi-step execution", prompt)
                self.assertIn("Do not update it after", prompt)
                self.assertNotIn("tracked mode", prompt)
                self.assertNotIn("acceptance_revision", prompt)

    def test_read_only_profiles_block_todo_tool(self):
        for name in ("general", "review"):
            with self.subTest(profile=name):
                cfg = get_profile(name).main_agent()
                self.assertIn("update_todo", cfg.blocked_tool_names)

        plan_cfg = get_profile("plan").main_agent()
        self.assertIsNone(plan_cfg.time_budget)
        self.assertIn("planning flow (plan.md)", plan_cfg.system_prompt)

    def test_terminal_keeps_shell_policy_and_hard_timeout(self):
        cfg = get_profile("terminal").main_agent()

        self.assertEqual(
            [type(mw) for mw in cfg.middlewares],
            [TerminalShellEditPolicyMiddleware],
        )
        self.assertEqual(cfg.time_budget, 1800)
        self.assertNotIn("tracked", cfg.system_prompt)
        self.assertIn("non-trivial multi-step execution", cfg.system_prompt)

    def test_terminal_profile_resolves_timeout_from_task_name_env(self):
        with patch.dict(os.environ, {"HARNESS_TERMINAL_TASK_NAME": "terminal-bench/overfull-hbox"}):
            timeout = TerminalProfile().resolve_task_timeout("instruction text without task slug")

        self.assertEqual(timeout, 750.0)

    def test_terminal_profile_resolves_task_metadata_from_task_name_env(self):
        with patch.dict(os.environ, {"HARNESS_TERMINAL_TASK_NAME": "terminal-bench/configure-git-webserver"}):
            metadata = TerminalProfile().resolve_task_metadata("workspace is /app")

        self.assertEqual(metadata["task_name"], "configure-git-webserver")
        self.assertEqual(metadata["category"], "system-administration")
        self.assertEqual(metadata["agent_timeout_sec"], 900.0)


if __name__ == "__main__":
    unittest.main()
