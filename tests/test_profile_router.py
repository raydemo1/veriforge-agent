import copy
import json
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from harness_code_agent.profiles import router


def choice_response(choice="review", confidence=1.0):
    return {
        "model": "jev-1.13.0",
        "answers": {"profile": {
            "type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {name: float(name == choice) for name in router._ROUTE_CRITERIA},
        }},
    }


class JevProfileRouterTests(unittest.TestCase):
    def test_fixed_modes_and_terminal_bypass_classifier(self):
        classifier = Mock(side_effect=AssertionError("manual controls must not call Jev"))
        cases = [
            ("implement this plan", "review", "pinned", "review", "stay"),
            ("创建网页", "general", "pinned", "general", "stay"),
            ("修复错误", "plan", "pinned", "plan", "stay"),
            ("审查补丁", "coding-agent", "pinned", "coding-agent", "stay"),
            ("制定方案", "app-builder", "pinned", "app-builder", "stay"),
            ("build a web app", "terminal", "auto", "terminal", "stay"),
        ]
        for prompt, current, mode, expected, action in cases:
            with self.subTest(prompt=prompt):
                result = router.route_profile_for_turn(
                    prompt, current_profile=current, routing_mode=mode, llm_classifier=classifier,
                )
                self.assertEqual((result.profile_name, result.action), (expected, action))
                self.assertFalse(result.llm_called)
        classifier.assert_not_called()

    def test_all_task_intents_reach_classifier_including_negated_mode_commands(self):
        for prompt in ["审查后修复这个 bug", "先给方案，不要改代码", "创建一个网站", "切换到编码模式", "不要切换到编码模式", '解释“切换到编码模式”的含义']:
            classifier = Mock(return_value=router.LlmRouteResult(profile_name="general", confidence=1.0))
            with self.subTest(prompt=prompt):
                result = router.route_profile_for_turn(prompt, current_profile="general", llm_classifier=classifier)
                classifier.assert_called_once()
                self.assertTrue(result.llm_called)
                self.assertEqual(result.source, "llm")

    def test_followup_context_is_forwarded_and_general_answer_preserves_profile(self):
        classifier = Mock(return_value=router.LlmRouteResult(profile_name="general", confidence=0.96))
        result = router.route_profile_for_turn(
            "总结一下", current_profile="coding-agent", previous_user_task="fix parser",
            previous_assistant_text="parser updated", llm_classifier=classifier,
        )
        classifier.assert_called_once_with(
            user_prompt="总结一下", current_profile="coding-agent",
            previous_user_task="fix parser", previous_assistant_text="parser updated",
        )
        self.assertEqual((result.profile_name, result.matched_profile, result.action, result.turn_mode),
                         ("coding-agent", "general", "direct_answer", "direct_answer"))

    def test_failure_and_low_confidence_never_switch(self):
        for classifier, failure in [
            (lambda **_: router.LlmRouteResult(profile_name="coding-agent", confidence=0.59), "low_confidence"),
            (Mock(side_effect=TimeoutError("secret message")), "timeout"),
            (Mock(side_effect=RuntimeError("secret message")), "request_error"),
            (lambda **_: None, "invalid_response"),
            (lambda **_: router.LlmRouteResult(profile_name="terminal", confidence=1.0), "invalid_profile"),
            (lambda **_: router.LlmRouteResult(profile_name="review", confidence=float("nan")), "invalid_confidence"),
        ]:
            with self.subTest(failure=failure):
                result = router.route_profile_for_turn("继续", current_profile="plan", llm_classifier=classifier)
                self.assertEqual((result.profile_name, result.action, result.failure_type), ("plan", "stay", failure))
                self.assertTrue(result.fallback_used)

    def test_confidence_gate_accepts_clear_choices_below_old_threshold(self):
        for confidence in [0.6, 0.68, 0.73, 0.75]:
            with self.subTest(confidence=confidence):
                result = router.route_profile_for_turn(
                    "实现这个模块", current_profile="plan",
                    llm_classifier=lambda confidence=confidence, **_: router.LlmRouteResult(
                        profile_name="coding-agent", confidence=confidence,
                    ),
                )
                self.assertEqual((result.profile_name, result.action), ("coding-agent", "switch_profile"))
                self.assertFalse(result.fallback_used)

    def test_choice_response_validation(self):
        valid = choice_response()
        bad_choice = copy.deepcopy(valid)
        bad_choice["answers"]["profile"]["choice"] = "terminal"
        bad_type = copy.deepcopy(valid)
        bad_type["answers"]["profile"]["type"] = "noul"
        cases = [(None, "invalid_response"), ({"choices": []}, "invalid_response"),
                 (bad_type, "invalid_response"), (bad_choice, "invalid_profile")]
        for value in [True, "1", float("nan"), float("inf"), -0.1, 1.1]:
            invalid = copy.deepcopy(valid)
            invalid["answers"]["profile"]["confidence"] = value
            cases.append((invalid, "invalid_confidence"))
        for probabilities in [{}, {"review": 1}, dict.fromkeys(router._ROUTE_CRITERIA, 1.0),
                              {name: float(name == "plan") for name in router._ROUTE_CRITERIA},
                              {name: True for name in router._ROUTE_CRITERIA}]:
            invalid = copy.deepcopy(valid)
            invalid["answers"]["profile"]["probabilities"] = probabilities
            cases.append((invalid, "invalid_probabilities"))
        for data, failure in cases:
            with self.subTest(data=data):
                result = router._parse_jev_route_result(data, requested_model="jev-latest")
                self.assertEqual(result.failure_type, failure)
        result = router._parse_jev_route_result(valid, requested_model="jev-latest")
        self.assertEqual((result.profile_name, result.model, result.provider), ("review", "jev-1.13.0", "jev"))
        self.assertEqual(result.probabilities["review"], 1.0)

    def test_native_request_uses_independent_credentials_and_reuses_transport(self):
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=choice_response())
        transport_client = httpx.Client(transport=httpx.MockTransport(handle))
        def create(**kwargs):
            return OpenAI(**kwargs, http_client=transport_client)
        classifier = router.JevRouteClassifier()
        with (
            patch.object(router.config, "ROUTER_API_KEY", "router-test-key"),
            patch.object(router.config, "ROUTER_BASE_URL", "https://router.example/v1"),
            patch.object(router.config, "ROUTER_MODEL", "jev-latest"),
            patch.object(router.config, "ROUTER_TIMEOUT_SECONDS", 3.0),
            patch.object(router.config, "API_KEY", "main-test-key"),
            patch.object(router.config, "BASE_URL", "https://main.example/v1"),
            patch.object(router, "OpenAI", side_effect=create) as factory,
        ):
            for _ in range(2):
                result = classifier(user_prompt="审查补丁", current_profile="coding-agent",
                                    previous_user_task="x" * 900, previous_assistant_text="y" * 1400)
            factory.assert_called_once_with(api_key="router-test-key", base_url="https://router.example/v1",
                                            timeout=3.0, max_retries=0)
            classifier.close()
            classifier.close()
            self.assertEqual(classifier(user_prompt="继续", current_profile="review").failure_type, "closed")
        self.assertTrue(transport_client.is_closed)
        self.assertEqual(len(requests), 2)
        self.assertEqual(str(requests[0].url), "https://router.example/v1/systemone")
        self.assertEqual(requests[0].headers["authorization"], "Bearer router-test-key")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "jev-latest")
        self.assertEqual(body["questions"]["profile"]["type"], "choice")
        self.assertEqual(len(body["state"]["previous_user_task"]), 800)
        self.assertEqual(len(body["state"]["previous_assistant_answer"]), 1200)
        self.assertEqual(result.profile_name, "review")

    def test_missing_configuration_does_not_use_main_model(self):
        with patch.object(router.config, "ROUTER_API_KEY", ""), patch.object(router, "OpenAI") as factory:
            result = router.route_profile_for_turn("修复登录错误", current_profile="general")
        factory.assert_not_called()
        self.assertEqual((result.profile_name, result.failure_type), ("general", "missing_configuration"))

    def test_provider_failure_does_not_retry_or_log_secret_response(self):
        for status, failure in [(503, "request_error"), (429, "rate_limit")]:
            with self.subTest(status=status):
                requests = []
                def handle(request, requests=requests, status=status):
                    requests.append(request)
                    return httpx.Response(status, json={"error": {"message": "secret-provider-body"}})
                with (
                    patch.object(router.config, "ROUTER_API_KEY", "test-key"),
                    patch.object(router, "OpenAI", side_effect=lambda **kwargs: OpenAI(
                        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))),
                    self.assertLogs("harness", level="INFO") as logs,
                ):
                    classifier = router.JevRouteClassifier()
                    result = router.route_profile_for_turn("继续", current_profile="review", llm_classifier=classifier)
                    classifier.close()
                self.assertEqual(len(requests), 1)
                self.assertEqual(result.failure_type, failure)
                self.assertNotIn("secret-provider-body", "\n".join(logs.output))

    def test_selected_profile_stays_fixed_until_auto_is_selected(self):
        from harness_code_agent.core.interactive import InteractiveSession
        session = InteractiveSession.__new__(InteractiveSession)
        session.session = Mock(id="test-session")
        session.conversation = Mock()
        session.event_bus = Mock()
        session.profile = Mock()
        session.profile.name.return_value = "review"
        session.routing_mode = "auto"
        session.last_user_task = "审查补丁"
        session.last_assistant_text = "审查中"
        session.profile_router = Mock(return_value=router.LlmRouteResult(
            profile_name="coding-agent", confidence=0.96,
            probabilities={name: float(name == "coding-agent") for name in router._ROUTE_CRITERIA},
        ))
        session.session_store = Mock()
        session._switch_profile = Mock()

        session.switch_profile("review")
        self.assertEqual(session.routing_mode, "pinned")
        self.assertIsNone(session._maybe_auto_route_profile("修复发现的问题"))
        session.profile_router.assert_not_called()
        session.session_store.update_routing_mode.assert_called_once_with("test-session", "pinned")

        session.enable_auto_profile_routing()
        decision = session._maybe_auto_route_profile("修复发现的问题")
        self.assertEqual((decision.profile_name, decision.action), ("coding-agent", "switch_profile"))
        self.assertEqual(session.routing_mode, "auto")
        session.profile_router.assert_called_once()
        session._switch_profile.assert_called_with("coding-agent", reason="auto route")
        session.session_store.update_routing_mode.assert_called_with("test-session", "auto")
        self.assertEqual(session.event_bus.emit.call_args.kwargs["payload"]["probabilities"]["coding-agent"], 1.0)


if __name__ == "__main__":
    unittest.main()
