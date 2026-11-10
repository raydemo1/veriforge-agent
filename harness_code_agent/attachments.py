"""Unified attachment staging and model-input preparation."""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import config
from .workspace.paths import is_relative_to

ModelInputMode = Literal["text", "multimodal"]
AttachmentKind = Literal["text", "docx", "image", "pdf"]
AttachmentSource = Literal["clipboard", "picker", "mention", "path"]

IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml", ".html",
    ".css", ".scss", ".less", ".js", ".jsx", ".ts", ".tsx", ".mjs",
    ".cjs", ".py", ".pyi", ".java", ".kt", ".kts", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".rb",
    ".swift", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql",
    ".graphql", ".proto", ".ini", ".cfg", ".conf", ".env", ".gitignore",
}
PDF_SUFFIX = ".pdf"
DOCX_SUFFIX = ".docx"
LEGACY_DOC_SUFFIX = ".doc"

DEFAULT_MAX_ATTACHMENTS = 10
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TURN_BYTES = 50 * 1024 * 1024


class AttachmentError(ValueError):
    """A user-actionable attachment validation failure."""


class ExternalPathConfirmationRequired(AttachmentError):
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        super().__init__("需要确认读取工作区外的文件")


@dataclass(frozen=True)
class Attachment:
    id: str
    name: str
    path: str
    source: AttachmentSource
    mime_type: str
    size: int
    sha256: str
    kind: AttachmentKind
    cached: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "source": self.source,
            "mimeType": self.mime_type,
            "size": self.size,
            "sha256": self.sha256,
            "kind": self.kind,
            "cached": self.cached,
        }


@dataclass(frozen=True)
class TurnSubmission:
    text: str
    attachment_ids: tuple[str, ...] = ()
    authorized_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedTurn:
    text: str
    attachments: tuple[Attachment, ...] = ()
    model_content: str | list[dict[str, Any]] = ""
    session_id: str | None = None
    ui_request_id: str | None = None


@dataclass
class AttachmentManager:
    workspace_root: Path
    session_root: Path
    input_mode: ModelInputMode
    max_attachments: int = DEFAULT_MAX_ATTACHMENTS
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_turn_bytes: int = DEFAULT_MAX_TURN_BYTES
    _attachments: dict[str, Attachment] = field(default_factory=dict, init=False)
    _dedupe: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()
        self.session_root = self.session_root.resolve()
        if self.input_mode not in {"text", "multimodal"}:
            raise ValueError(f"Unsupported model input mode: {self.input_mode}")

    @property
    def attachments_dir(self) -> Path:
        return self.session_root / "attachments"

    def get(self, attachment_id: str) -> Attachment:
        try:
            return self._attachments[attachment_id]
        except KeyError as exc:
            raise AttachmentError(f"附件不存在或已失效：{attachment_id}") from exc

    def remove(self, attachment_id: str) -> bool:
        attachment = self._attachments.pop(attachment_id, None)
        if attachment is None:
            return False
        self._dedupe.pop(attachment.sha256, None)
        if attachment.cached:
            path = Path(attachment.path)
            try:
                if path.exists() and is_relative_to(path.resolve(), self.attachments_dir.resolve()):
                    path.unlink()
                    parent = path.parent
                    if parent != self.attachments_dir and parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
            except OSError:
                pass
        return True

    def stage_path(
        self,
        path: str | Path,
        *,
        source: AttachmentSource,
        copy_to_session: bool | None = None,
    ) -> Attachment:
        resolved = _resolve_candidate(path, self.workspace_root)
        if not resolved.exists() or not resolved.is_file():
            raise AttachmentError(f"文件不存在：{path}")
        external = not is_relative_to(resolved, self.workspace_root)
        should_copy = external if copy_to_session is None else copy_to_session
        size = resolved.stat().st_size
        self._validate_size(size)
        kind, mime_type = classify_attachment(resolved.name, _guess_mime(resolved))
        self._validate_kind(kind)
        digest = _sha256_file(resolved)
        duplicate = self._dedupe.get(digest)
        if duplicate:
            return self._attachments[duplicate]
        stored_path = resolved
        cached = False
        if should_copy:
            stored_path = self._copy_into_session(resolved, resolved.name)
            cached = True
        return self._register(
            name=resolved.name,
            path=stored_path,
            source=source,
            mime_type=mime_type,
            size=size,
            sha256=digest,
            kind=kind,
            cached=cached,
        )

    def stage_bytes(
        self,
        data: bytes,
        *,
        name: str,
        mime_type: str,
        source: AttachmentSource = "clipboard",
    ) -> Attachment:
        self._validate_size(len(data))
        safe_name = _safe_filename(name)
        kind, normalized_mime = classify_attachment(safe_name, mime_type)
        self._validate_kind(kind)
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self._dedupe.get(digest)
        if duplicate:
            return self._attachments[duplicate]
        attachment_id = uuid.uuid4().hex
        target_dir = self.attachments_dir / attachment_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / safe_name
        target.write_bytes(data)
        return self._register(
            attachment_id=attachment_id,
            name=safe_name,
            path=target,
            source=source,
            mime_type=normalized_mime,
            size=len(data),
            sha256=digest,
            kind=kind,
            cached=True,
        )

    def prepare(self, submission: TurnSubmission) -> PreparedTurn:
        attachment_ids = list(dict.fromkeys(submission.attachment_ids))
        authorized = {str(_resolve_candidate(item, self.workspace_root)) for item in submission.authorized_paths}
        mentioned_paths = _file_mention_paths(submission.text)
        detected_paths = detect_explicit_file_paths(submission.text, self.workspace_root)
        path_entries: list[tuple[Path, AttachmentSource]] = []
        seen_paths: set[str] = set()
        for raw, source in [*((item, "mention") for item in mentioned_paths), *((item, "path") for item in detected_paths)]:
            resolved = _resolve_candidate(raw, self.workspace_root)
            key = str(resolved)
            if key in seen_paths or not resolved.exists() or not resolved.is_file():
                continue
            seen_paths.add(key)
            path_entries.append((resolved, source))

        external_unapproved = [
            str(path)
            for path, _source in path_entries
            if not is_relative_to(path, self.workspace_root) and str(path) not in authorized
        ]
        if external_unapproved:
            raise ExternalPathConfirmationRequired(external_unapproved)

        for path, source in path_entries:
            attachment = self.stage_path(
                path,
                source=source,
                copy_to_session=not is_relative_to(path, self.workspace_root),
            )
            attachment_ids.append(attachment.id)

        attachments = tuple(self.get(item) for item in dict.fromkeys(attachment_ids))
        self._validate_turn_limits(attachments)
        self._validate_locally_parsed_files(attachments)
        return PreparedTurn(
            text=submission.text.strip(),
            attachments=attachments,
            model_content="",
        )

    def _validate_kind(self, kind: AttachmentKind) -> None:
        if self.input_mode == "text" and kind == "image":
            raise AttachmentError("当前配置为文本模型，不能上传图片")

    def _validate_size(self, size: int) -> None:
        if size > self.max_file_bytes:
            raise AttachmentError(f"附件超过单文件限制（{_format_bytes(self.max_file_bytes)}）")

    def _validate_turn_limits(self, attachments: tuple[Attachment, ...]) -> None:
        if len(attachments) > self.max_attachments:
            raise AttachmentError(f"每回合最多上传 {self.max_attachments} 个附件")
        total = sum(item.size for item in attachments)
        if total > self.max_turn_bytes:
            raise AttachmentError(f"附件总大小超过限制（{_format_bytes(self.max_turn_bytes)}）")

    def _validate_locally_parsed_files(self, attachments: tuple[Attachment, ...]) -> None:
        for attachment in attachments:
            path = Path(attachment.path)
            if attachment.kind == "text":
                try:
                    path.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise AttachmentError(f"文本附件不是有效的 UTF-8：{attachment.name}") from exc
                except OSError as exc:
                    raise AttachmentError(f"无法读取附件：{attachment.name}") from exc

    def _copy_into_session(self, source: Path, name: str) -> Path:
        attachment_id = uuid.uuid4().hex
        target_dir = self.attachments_dir / attachment_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / _safe_filename(name)
        shutil.copy2(source, target)
        return target

    def _register(
        self,
        *,
        name: str,
        path: Path,
        source: AttachmentSource,
        mime_type: str,
        size: int,
        sha256: str,
        kind: AttachmentKind,
        cached: bool,
        attachment_id: str | None = None,
    ) -> Attachment:
        attachment = Attachment(
            id=attachment_id or (path.parent.name if cached else uuid.uuid4().hex),
            name=name,
            path=str(path.resolve()),
            source=source,
            mime_type=mime_type,
            size=size,
            sha256=sha256,
            kind=kind,
            cached=cached,
        )
        self._attachments[attachment.id] = attachment
        self._dedupe[attachment.sha256] = attachment.id
        return attachment


def model_input_mode() -> ModelInputMode:
    profile = getattr(config, "MODEL_PROFILE", None)
    value = str(getattr(profile, "input_mode", "text") or "text").strip().lower()
    return "multimodal" if value == "multimodal" else "text"


def classify_attachment(name: str, mime_type: str | None) -> tuple[AttachmentKind, str]:
    suffix = Path(name).suffix.lower()
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if suffix == LEGACY_DOC_SUFFIX:
        raise AttachmentError("暂不支持旧版 .doc，请另存为 .docx 后上传")
    if suffix in IMAGE_SUFFIXES or mime in IMAGE_MIME_TYPES:
        normalized = "image/jpeg" if suffix in {".jpg", ".jpeg"} else mime
        if normalized not in IMAGE_MIME_TYPES:
            normalized = mimetypes.types_map.get(suffix, "application/octet-stream")
        if normalized not in IMAGE_MIME_TYPES:
            raise AttachmentError(f"不支持的图片格式：{name}")
        return "image", normalized
    if suffix == PDF_SUFFIX or mime == "application/pdf":
        return "pdf", "application/pdf"
    if suffix == DOCX_SUFFIX or mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix in TEXT_SUFFIXES or mime.startswith("text/") or not suffix:
        return "text", mime if mime.startswith("text/") else "text/plain"
    raise AttachmentError(f"不支持的文件格式：{name}")


def build_model_content(
    text: str,
    attachments: tuple[Attachment, ...],
    *,
    text_transform: Callable[[Attachment, str], str] | None = None,
) -> str | list[dict[str, Any]]:
    text_sections: list[str] = []
    if text:
        text_sections.append(text)
    binary_blocks: list[dict[str, Any]] = []
    for attachment in attachments:
        path = Path(attachment.path)
        if attachment.kind == "text":
            content = path.read_text(encoding="utf-8")
            if text_transform is not None:
                content = text_transform(attachment, content)
            text_sections.append(_render_text_attachment(attachment, content))
        elif attachment.kind == "docx":
            text_sections.append(_render_skill_attachment(attachment, "docx"))
        elif attachment.kind == "image":
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            binary_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{attachment.mime_type};base64,{encoded}"},
                "attachment": _model_attachment_metadata(attachment),
            })
        elif attachment.kind == "pdf":
            text_sections.append(_render_skill_attachment(attachment, "pdf"))
    combined_text = "\n\n".join(section for section in text_sections if section)
    if not binary_blocks:
        return combined_text
    blocks: list[dict[str, Any]] = []
    if combined_text:
        blocks.append({"type": "text", "text": combined_text})
    blocks.extend(binary_blocks)
    return blocks


def detect_explicit_file_paths(text: str, workspace_root: str | Path) -> list[str]:
    root = Path(workspace_root).resolve()
    candidates: list[str] = []
    candidates.extend(match.group(1) for match in re.finditer(r"[\"']([^\"'\r\n]+)[\"']", text))
    candidates.extend(
        match.group(0)
        for match in re.finditer(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|\\\\|\.[\\/])[^\s\"'<>|]+", text)
    )
    candidates.extend(
        token.strip("()[]{}<>,;:!?")
        for token in text.split()
        if not token.startswith(("http://", "https://", "@"))
        and ("/" in token or "\\" in token or Path(token.strip("()[]{}<>,;:!?\"'")).suffix.lower() in _all_supported_suffixes())
    )
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        value = raw.strip().strip("\"'").rstrip(".,;:!?)]}")
        if not value or value.startswith(("http://", "https://", "@file:")):
            continue
        path = _resolve_candidate(value, root)
        key = str(path)
        if key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        resolved.append(key)
    return resolved


def _file_mention_paths(text: str) -> list[str]:
    from .core.mentions import parse_mentions

    return [item.target for item in parse_mentions(text) if item.kind == "file"]


def _resolve_candidate(path: str | Path, workspace_root: Path) -> Path:
    raw = Path(os.path.expandvars(str(path))).expanduser()
    return raw.resolve() if raw.is_absolute() else (workspace_root / raw).resolve()


def _guess_mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(name: str) -> str:
    base = Path(name).name.strip().replace("\x00", "") or "attachment"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", " ", "（", "）"} else "_" for ch in base)
    return safe[:180] or "attachment"


def _all_supported_suffixes() -> set[str]:
    return TEXT_SUFFIXES | IMAGE_SUFFIXES | {PDF_SUFFIX, DOCX_SUFFIX, LEGACY_DOC_SUFFIX}


def _render_text_attachment(attachment: Attachment, content: str) -> str:
    return (
        f"[ATTACHMENT name={attachment.name!r} mime={attachment.mime_type} "
        f"sha256={attachment.sha256}]\n{content}"
    )


def _render_skill_attachment(attachment: Attachment, skill_name: str) -> str:
    return (
        f"[DOCUMENT ATTACHMENT name={attachment.name!r} mime={attachment.mime_type} "
        f"size={attachment.size} sha256={attachment.sha256}]\n"
        f"local_path: {attachment.path}\n"
        f"Before processing this document, load `catalog/{skill_name}/SKILL.md` with "
        "`read_skill_file` and follow its instructions. The document bytes are available only "
        "at local_path; they are not embedded in this model request."
    )


def _model_attachment_metadata(attachment: Attachment) -> dict[str, Any]:
    return {
        "id": attachment.id,
        "name": attachment.name,
        "mime_type": attachment.mime_type,
        "size": attachment.size,
        "sha256": attachment.sha256,
    }


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):g} MB"
