from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness_code_agent.agent.context import _messages_to_text
from harness_code_agent.agent.llm_channel import _is_multimodal_rejection
from harness_code_agent.attachments import (
    AttachmentError,
    AttachmentManager,
    ExternalPathConfirmationRequired,
    TurnSubmission,
    build_model_content,
    detect_explicit_file_paths,
)
from harness_code_agent.sessions.events import UserInputEvent
from harness_code_agent.support.llm import _strip_response_only_message_fields
from harness_code_agent.support.tokens import count_tokens


class AttachmentManagerTests(unittest.TestCase):
    def _manager(self, root: Path, mode: str = "text") -> AttachmentManager:
        session = root / ".harness" / "sessions" / "session-1"
        session.mkdir(parents=True)
        return AttachmentManager(root, session, mode)  # type: ignore[arg-type]

    def test_text_model_accepts_documents_and_rejects_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            text_path = root / "notes.md"
            text_path.write_text("hello", encoding="utf-8")
            staged = manager.stage_path(text_path, source="picker")
            self.assertEqual(staged.kind, "text")

            image_path = root / "screen.png"
            image_path.write_bytes(b"not-a-real-png")
            with self.assertRaisesRegex(AttachmentError, "文本模型"):
                manager.stage_path(image_path, source="picker")

            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            self.assertEqual(manager.stage_path(pdf_path, source="picker").kind, "pdf")

            docx = manager.stage_bytes(
                b"docx-package",
                name="brief.docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            self.assertEqual(docx.kind, "docx")

            with self.assertRaisesRegex(AttachmentError, "文本模型"):
                manager.stage_bytes(b"image", name="clipboard.jpg", mime_type="image/jpeg")

    def test_only_images_become_multimodal_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root, "multimodal")
            image = manager.stage_bytes(
                b"image-bytes", name="shot.png", mime_type="image/png"
            )
            pdf = manager.stage_bytes(
                b"%PDF-1.4", name="spec.pdf", mime_type="application/pdf"
            )
            content = build_model_content("compare", (image, pdf))
            self.assertIsInstance(content, list)
            self.assertEqual([item["type"] for item in content], ["text", "image_url"])
            self.assertIn("catalog/pdf/SKILL.md", content[0]["text"])
            self.assertIn(str(Path(pdf.path)), content[0]["text"])
            self.assertNotIn("%PDF", str(content))
            self.assertNotIn("file_data", str(content))
            self.assertNotIn("image-bytes", str(image.public_dict()))
            self.assertNotIn("file_data", str(pdf.public_dict()))

    def test_workspace_path_is_live_and_external_authorization_copies_file(self):
        with tempfile.TemporaryDirectory() as workspace_tmp, tempfile.TemporaryDirectory() as external_tmp:
            root = Path(workspace_tmp)
            manager = self._manager(root)
            inside = root / "inside.txt"
            inside.write_text("inside", encoding="utf-8")
            outside = Path(external_tmp) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")

            prepared = manager.prepare(TurnSubmission(f"read {inside}"))
            self.assertFalse(prepared.attachments[0].cached)

            with self.assertRaises(ExternalPathConfirmationRequired) as raised:
                manager.prepare(TurnSubmission(f'read "{outside}"'))
            self.assertEqual(raised.exception.paths, [str(outside.resolve())])

            authorized = manager.prepare(
                TurnSubmission(f'read "{outside}"', authorized_paths=(str(outside),))
            )
            self.assertTrue(authorized.attachments[0].cached)
            self.assertTrue(Path(authorized.attachments[0].path).is_relative_to(manager.attachments_dir))

    def test_mentions_and_explicit_paths_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            target = root / "README.md"
            target.write_text("read me", encoding="utf-8")
            prepared = manager.prepare(TurnSubmission("inspect @file:README.md and README.md"))
            self.assertEqual(len(prepared.attachments), 1)

    def test_picker_clipboard_mention_and_prompt_path_share_one_attachment_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            target = root / "same.txt"
            target.write_text("same", encoding="utf-8")
            picker = manager.stage_path(target, source="picker")
            clipboard = manager.stage_bytes(b"same", name="clipboard.txt", mime_type="text/plain")
            prepared = manager.prepare(TurnSubmission("@file:same.txt same.txt"))
            self.assertEqual(picker.id, clipboard.id)
            self.assertEqual(prepared.attachments, (picker,))

    def test_text_model_keeps_all_document_entry_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            picker_path = root / "picker.pdf"
            picker_path.write_bytes(b"%PDF-picker")
            mention_path = root / "mention.pdf"
            mention_path.write_bytes(b"%PDF-mention")
            prompt_path = root / "prompt.docx"
            prompt_path.write_bytes(b"docx-prompt")

            picker = manager.stage_path(picker_path, source="picker")
            clipboard = manager.stage_bytes(
                b"docx-clipboard",
                name="clipboard.docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            mentioned = manager.prepare(TurnSubmission("inspect @file:mention.pdf"))
            explicit = manager.prepare(TurnSubmission(f'inspect "{prompt_path}"'))

            self.assertEqual((picker.kind, picker.source), ("pdf", "picker"))
            self.assertEqual((clipboard.kind, clipboard.source), ("docx", "clipboard"))
            self.assertEqual(
                (mentioned.attachments[0].kind, mentioned.attachments[0].source),
                ("pdf", "mention"),
            )
            self.assertEqual(
                (explicit.attachments[0].kind, explicit.attachments[0].source),
                ("docx", "path"),
            )

    def test_detect_explicit_paths_ignores_urls_directories_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.ts").write_text("export {}", encoding="utf-8")
            paths = detect_explicit_file_paths(
                "check ./src/app.ts https://example.com/a.pdf missing.pdf ./src", root
            )
            self.assertEqual(paths, [str((root / "src" / "app.ts").resolve())])

    def test_limits_and_deduplication_apply_before_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / ".harness" / "sessions" / "session-1"
            session.mkdir(parents=True)
            manager = AttachmentManager(
                root, session, "text", max_attachments=1, max_file_bytes=4, max_turn_bytes=4
            )
            first = manager.stage_bytes(b"same", name="a.txt", mime_type="text/plain")
            duplicate = manager.stage_bytes(b"same", name="different.txt", mime_type="text/plain")
            self.assertEqual(first.id, duplicate.id)
            with self.assertRaisesRegex(AttachmentError, "单文件限制"):
                manager.stage_bytes(b"large", name="b.txt", mime_type="text/plain")

    def test_invalid_utf8_and_unknown_binary_are_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            invalid_text = manager.stage_bytes(b"\xff\xfe", name="bad.txt", mime_type="text/plain")
            with self.assertRaisesRegex(AttachmentError, "UTF-8"):
                manager.prepare(TurnSubmission("inspect", attachment_ids=(invalid_text.id,)))
            unknown = root / "archive.bin"
            unknown.write_bytes(b"binary")
            with self.assertRaisesRegex(AttachmentError, "不支持"):
                manager.stage_path(unknown, source="picker")

    def test_attachment_count_and_total_size_limits_are_centralized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / ".harness" / "sessions" / "session-1"
            session.mkdir(parents=True)
            manager = AttachmentManager(
                root, session, "text", max_attachments=1, max_file_bytes=10, max_turn_bytes=6
            )
            first = manager.stage_bytes(b"one", name="one.txt", mime_type="text/plain")
            second = manager.stage_bytes(b"two2", name="two.txt", mime_type="text/plain")
            with self.assertRaisesRegex(AttachmentError, "最多上传"):
                manager.prepare(TurnSubmission("", attachment_ids=(first.id, second.id)))
            manager.max_attachments = 2
            with self.assertRaisesRegex(AttachmentError, "总大小"):
                manager.prepare(TurnSubmission("", attachment_ids=(first.id, second.id)))

    def test_docx_routes_to_builtin_skill_without_local_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            path = root / "brief.docx"
            path.write_bytes(b"opaque-docx-package")
            attachment = manager.stage_path(path, source="picker")
            content = build_model_content("", (attachment,))
            self.assertIn("catalog/docx/SKILL.md", content)
            self.assertIn(str(path.resolve()), content)
            self.assertNotIn("opaque-docx-package", content)

    def test_provider_and_context_never_expose_or_count_base64_as_text(self):
        payload = "A" * 100_000
        metadata = {
            "name": "shot.png",
            "mime_type": "image/png",
            "size": 4096,
            "sha256": "abc123",
        }
        messages = [{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{payload}"},
            "attachment": metadata,
        }]}]
        flattened = _messages_to_text(messages)
        self.assertIn("shot.png", flattened)
        self.assertNotIn(payload[:100], flattened)
        self.assertLess(count_tokens(messages), 2000)
        outbound = _strip_response_only_message_fields(messages)
        self.assertNotIn("attachment", outbound[0]["content"][0])
        self.assertIn(payload, outbound[0]["content"][0]["image_url"]["url"])
        error = RuntimeError("unsupported image content type")
        self.assertTrue(_is_multimodal_rejection(error, messages))
        self.assertFalse(_is_multimodal_rejection(error, [{"role": "user", "content": "text"}]))

    def test_deepseek_adapter_flattens_supported_image_file_fields(self):
        messages = [{"role": "user", "content": [{
            "type": "file",
            "file": {"filename": "screen.png", "file_data": "data:image/png;base64,AAAA"},
            "attachment": {"name": "screen.png", "size": 3},
        }]}]
        outbound = _strip_response_only_message_fields(messages, provider_name="deepseek")
        block = outbound[0]["content"][0]
        self.assertEqual(block["type"], "file")
        self.assertEqual(block["filename"], "screen.png")
        self.assertEqual(block["file_data"], "data:image/png;base64,AAAA")
        self.assertNotIn("file", block)
        self.assertNotIn("attachment", block)

    def test_user_event_records_attachment_metadata_only(self):
        event = UserInputEvent(
            text="inspect",
            attachments=[{"name": "shot.png", "mimeType": "image/png", "size": 42}],
        ).to_event()
        serialized = str(event.payload)
        self.assertIn("shot.png", serialized)
        self.assertNotIn("base64", serialized)

    def test_text_attachment_can_be_externalized_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            path = root / "large.txt"
            path.write_text("x" * 5000, encoding="utf-8")
            attachment = manager.stage_path(path, source="picker")
            content = build_model_content(
                "inspect",
                (attachment,),
                text_transform=lambda _attachment, _content: "[EXTERNALIZED]",
            )
            self.assertIn("[EXTERNALIZED]", content)
            self.assertNotIn("x" * 100, content)


if __name__ == "__main__":
    unittest.main()
