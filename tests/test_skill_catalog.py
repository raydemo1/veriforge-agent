import tempfile
import unittest
from pathlib import Path

from harness_code_agent.runtime.builtins.filesystem import read_skill_file
from harness_code_agent.skills.registry import SKILLS_DIR, SkillRegistry


def _write_skill(root: Path, name: str, description: str, *, user_only: bool = False, body: str = "") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    if user_only:
        lines.append('argument-hint: "<argument>"')
        lines.append("disable-model-invocation: true")
    lines.append("---")
    lines.append("")
    lines.append(body or "Follow the skill instructions.")
    (skill_dir / "SKILL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class SkillCatalogTests(unittest.TestCase):
    def test_catalog_splits_skills_by_invocation_permission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "skills"
            skills_dir.mkdir()
            _write_skill(skills_dir, "diagnosing-bugs", "Diagnose hard bugs. Use when a bug resists explanation.")
            _write_skill(skills_dir, "triage", "Triage an issue. Use when an issue needs classification.", user_only=True)

            registry = SkillRegistry(skills_dir)

            self.assertEqual([item["name"] for item in registry.model_catalog], ["diagnosing-bugs"])
            self.assertEqual([item["name"] for item in registry.user_only], ["triage"])
            self.assertEqual(registry.user_only[0]["argument_hint"], "<argument>")

            listing = registry.catalog_listing()
            self.assertIn("Model-invoked:", listing)
            self.assertIn("User-invoked only:", listing)
            self.assertIn("**diagnosing-bugs**", listing)
            self.assertIn("**triage**", listing)
            self.assertIn("Path: `skills/triage/SKILL.md`", listing)

    def test_catalog_listing_carries_no_generic_policy(self):
        listing = SkillRegistry().catalog_listing()

        self.assertNotIn("update_todo", listing)
        self.assertNotIn("Skill routing", listing)
        self.assertNotIn("plan.md", listing)


    def test_slash_invocation_reaches_every_skill(self):
        registry = SkillRegistry()

        user_invoked = registry.build_user_invocation("/triage  #42 locked")
        self.assertIsNotNone(user_invoked)
        self.assertEqual(user_invoked.name, "triage")
        self.assertEqual(user_invoked.arguments, "#42 locked")

        model_invoked = registry.build_user_invocation("/code-review")
        self.assertIsNotNone(model_invoked)
        self.assertEqual(model_invoked.name, "code-review")
        self.assertEqual(model_invoked.arguments, "")

    def test_slash_invocation_accepts_any_whitespace_between_name_and_arguments(self):
        registry = SkillRegistry()

        for line in ("/triage\t#42", "/triage    #42", "/triage  \n #42", "  /triage \t #42  "):
            with self.subTest(line=line):
                invocation = registry.build_user_invocation(line)
                self.assertIsNotNone(invocation, line)
                self.assertEqual(invocation.arguments, "#42")

    def test_natural_language_is_never_matched_by_the_program(self):
        registry = SkillRegistry()

        for line in ("请用 triage 技能处理 #42", "use the code-review skill", "/unknown-skill now", "帮我看看这段代码"):
            with self.subTest(line=line):
                self.assertIsNone(registry.build_user_invocation(line))

    def test_catalog_refreshes_only_after_a_real_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "catalog"
            skills_dir.mkdir()
            _write_skill(skills_dir, "one", "First skill. Use when testing refresh.")
            registry = SkillRegistry(skills_dir)

            self.assertFalse(registry.refresh_if_changed())

            _write_skill(skills_dir, "two", "Second skill. Use when testing refresh.")
            self.assertTrue(registry.refresh_if_changed())
            self.assertIn("**two**", registry.catalog_listing())
            self.assertFalse(registry.refresh_if_changed())

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

    def test_removed_and_replaced_workflows_are_not_registered(self):
        names = {skill.name for skill in SkillRegistry().skills}

        self.assertTrue(
            {"to-spec", "to-tickets", "skill-creator", "find-skills"} <= names
        )
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
            skills_dir.mkdir()
            _write_skill(skills_dir, "valid-folder", "Mismatch. Use when testing validation.")
            skill_file = skills_dir / "valid-folder" / "SKILL.md"
            skill_file.write_text(
                "---\nname: wrong-name\ndescription: Mismatch. Use when testing validation.\n---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must match folder"):
                SkillRegistry(skills_dir)

    def test_catalog_rejects_duplicate_frontmatter_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "catalog"
            skills_dir.mkdir()
            _write_skill(skills_dir, "duplicate", "First description. Use when testing validation.")
            skill_file = skills_dir / "duplicate" / "SKILL.md"
            skill_file.write_text(
                "---\n"
                "name: duplicate\n"
                "description: First description. Use when testing validation.\n"
                "description: Second description. Use when testing validation.\n"
                "---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate frontmatter key"):
                SkillRegistry(skills_dir)

    def test_catalog_rejects_missing_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "catalog"
            skills_dir.mkdir()
            _write_skill(skills_dir, "unnamed", "Missing an explicit name. Use when testing validation.")
            skill_file = skills_dir / "unnamed" / "SKILL.md"
            skill_file.write_text(
                "---\ndescription: Missing an explicit name. Use when testing validation.\n---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "has no name"):
                SkillRegistry(skills_dir)


if __name__ == "__main__":
    unittest.main()
