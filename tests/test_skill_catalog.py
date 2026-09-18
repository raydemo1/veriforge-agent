import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_fake_openai_module() -> None:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

from harness_code_agent.runtime.builtins.filesystem import read_skill_file
from harness_code_agent.skills.registry import SKILLS_DIR, SkillRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SkillCatalogTests(unittest.TestCase):
    def test_registry_separates_user_and_model_invoked_skills(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            model_dir = skills_dir / "diagnosing-bugs"
            user_dir = skills_dir / "triage"
            model_dir.mkdir(parents=True)
            user_dir.mkdir(parents=True)
            (model_dir / "SKILL.md").write_text(
                "---\n"
                "name: diagnosing-bugs\n"
                "description: Diagnose hard bugs.\n"
                "---\n\n"
                "Build a tight red loop.\n",
                encoding="utf-8",
            )
            (user_dir / "SKILL.md").write_text(
                "---\n"
                "name: triage\n"
                "description: Triage an issue.\n"
                "argument-hint: \"<issue>\"\n"
                "disable-model-invocation: true\n"
                "---\n\n"
                "Triage the requested issue.\n",
                encoding="utf-8",
            )

            registry = SkillRegistry(skills_dir)

            self.assertEqual(
                [item["name"] for item in registry.model_catalog],
                ["diagnosing-bugs"],
            )
            self.assertEqual(
                [item["name"] for item in registry.user_commands],
                ["triage"],
            )
            self.assertEqual(registry.user_commands[0]["argument_hint"], "<issue>")
            prompt = registry.build_catalog_prompt()
            self.assertIn("**diagnosing-bugs**", prompt)
            self.assertNotIn("**triage**", prompt)

    def test_user_skill_invocation_includes_body_and_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            skill_dir = skills_dir / "triage"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: triage\n"
                "description: Triage an issue.\n"
                "disable-model-invocation: true\n"
                "---\n\n"
                "Inspect the full issue before deciding.\n",
                encoding="utf-8",
            )
            registry = SkillRegistry(skills_dir)

            invocation = registry.build_user_invocation("/triage 42")

            self.assertIsNotNone(invocation)
            self.assertEqual(invocation.name, "triage")
            self.assertEqual(invocation.arguments, "42")
            self.assertIn("Inspect the full issue before deciding.", invocation.prompt)
            self.assertIn("Arguments: 42", invocation.prompt)

    def test_catalog_points_execution_tracking_to_todo_tool(self):
        prompt = SkillRegistry().build_catalog_prompt()

        self.assertIn("update_todo", prompt)
        self.assertNotIn("update_plan_state", prompt)
        self.assertNotIn("PRD.md", prompt)
        self.assertNotIn("skills/catalog/prd", prompt)
        self.assertNotIn('**vibe-execution-guard**: "', prompt)

    def test_catalog_includes_core_workflow_skills(self):
        prompt = SkillRegistry().build_catalog_prompt()

        for skill_name in [
            "diagnosing-bugs",
            "tdd",
            "domain-modeling",
            "codebase-design",
            "code-review",
            "frontend-debugging",
            "writing-for-agents",
            "pdf",
            "docx",
        ]:
            with self.subTest(skill=skill_name):
                self.assertIn(f"**{skill_name}**", prompt)
        for user_skill in [
            "workflows",
            "triage",
            "handoff",
            "grill-with-docs",
            "to-spec",
            "to-tickets",
            "skill-creator",
            "find-skills",
        ]:
            with self.subTest(user_skill=user_skill):
                self.assertNotIn(f"**{user_skill}**", prompt)

    def test_catalog_paths_can_be_loaded_by_read_skill_file(self):
        registry = SkillRegistry()

        self.assertTrue(registry.skills)
        for metadata in registry.skills:
            skill = metadata.as_catalog_item()
            with self.subTest(skill=skill["name"], path=skill["path"]):
                result = read_skill_file(skill["path"])
                self.assertEqual(result.status, "success", result.output)
                self.assertIn("name:", result.output)

    def test_document_skills_and_references_are_repository_builtin(self):
        registry = SkillRegistry()
        model_names = {item["name"] for item in registry.model_catalog}

        self.assertTrue({"pdf", "docx"} <= model_names)
        for path, marker in [
            ("catalog/pdf/REFERENCE.md", "Capability map"),
            ("catalog/docx/REFERENCE.md", "Content coverage"),
        ]:
            with self.subTest(path=path):
                result = read_skill_file(path)
                self.assertEqual(result.status, "success", result.output)
                self.assertIn(marker, result.output)

        for skill_name in ["pdf", "docx"]:
            text = (SKILLS_DIR / skill_name / "SKILL.md").read_text(encoding="utf-8")
            self.assertNotIn("disable-model-invocation", text)
            self.assertNotIn(".codex/skills", text)

    def test_removed_and_replaced_workflows_are_not_registered(self):
        names = {skill.name for skill in SkillRegistry().skills}

        self.assertTrue({"to-spec", "to-tickets", "skill-creator", "find-skills"} <= names)
        self.assertTrue({"prd", "to-issues", "writing-great-skills"}.isdisjoint(names))

    def test_read_skill_file_rejects_catalog_directories(self):
        result = read_skill_file("catalog/codebase-design")

        self.assertEqual(result.status, "failed")
        self.assertIn("not found", result.output)

    def test_skill_files_do_not_include_scraped_metadata(self):
        forbidden_markers = [
            "Skill score",
            "Health score",
            "skillsbench",
            "Trigger phrases",
            "Privacy",
            "First Seen",
        ]

        skill_files = sorted(SKILLS_DIR.rglob("SKILL.md"))
        self.assertTrue(skill_files)
        for skill_file in skill_files:
            text = skill_file.read_text(encoding="utf-8")
            for marker in forbidden_markers:
                with self.subTest(skill=skill_file, marker=marker):
                    self.assertNotIn(marker, text)

    def test_catalog_rejects_invalid_skill_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "catalog"
            skill_dir = skills_dir / "valid-folder"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: wrong-name\ndescription: Mismatch.\n---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must match folder"):
                SkillRegistry(skills_dir)

    def test_catalog_rejects_duplicate_frontmatter_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "catalog"
            skill_dir = skills_dir / "duplicate"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: duplicate\n"
                "description: First description.\n"
                "description: Second description.\n"
                "---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate frontmatter key"):
                SkillRegistry(skills_dir)

    def test_catalog_rejects_missing_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "catalog"
            skill_dir = skills_dir / "unnamed"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\ndescription: Missing an explicit name.\n---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "has no name"):
                SkillRegistry(skills_dir)


if __name__ == "__main__":
    unittest.main()
