"""Provision the Python 3.14 interpreter used by benchmark agents."""

from __future__ import annotations

import shlex

PYTHON_PREFIX = "/opt/veriforge-python"
CONTAINER_PYTHON = f"{PYTHON_PREFIX}/bin/python3.14"
PYTHON_STANDALONE_TARBALL = "python-3.14.10-x86_64-unknown-linux-gnu.tar.gz"
_RELEASE = "20261110"
_ASSET = f"cpython-3.14.10+{_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
_SHA256 = "c2ff2e97422f5d3896a16e0317fd212ba977bbf2688d93a6f7a03d817506bd30"
_URL = f"https://github.com/astral-sh/python-build-standalone/releases/download/{_RELEASE}/{_ASSET}"
_MIRROR = f"https://registry.npmmirror.com/-/binary/python-build-standalone/{_RELEASE}/{_ASSET}"


def bootstrap_python(vendored_archive: str, *, download_url: str | None = None) -> str:
    """Install a dedicated interpreter without changing the task's system Python."""
    downloads = (download_url,) if download_url else (_URL, _MIRROR)
    fetch = " || ".join(f"fetch {shlex.quote(url)}" for url in downloads)
    verify = f"printf '%s  %s\\n' {_SHA256} \"$archive\" | sha256sum -c -" if not download_url else "true"
    check = f"{CONTAINER_PYTHON} -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 14))'"
    vendor = shlex.quote(vendored_archive)
    return f"""
(
  set -eu
  if {check} >/dev/null 2>&1; then exit 0; fi
  archive=$(mktemp)
  trap 'rm -f "$archive"' EXIT
  fetch() {{ curl -fsSL -o "$archive" "$1" || wget -q -O "$archive" "$1"; }}
  if [ -f {vendor} ]; then
    cp {vendor} "$archive"
    printf '%s  %s\\n' {_SHA256} "$archive" | sha256sum -c -
  else
    ({fetch})
    {verify}
  fi
  mkdir -p {PYTHON_PREFIX}
  tar -xzf "$archive" -C {PYTHON_PREFIX} --strip-components=1
  {check}
)
""".strip()
