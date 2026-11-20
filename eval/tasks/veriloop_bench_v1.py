"""VeriLoop-Bench v1: trap tasks inside small but realistic projects.

An earlier draft put every requirement in the prompt and carried one trap per
task; the fast DeepSeek tier passed 58 of 60 runs, so it could not show any
improvement and was dropped. This pack makes low-level mistakes realistic:

* Tasks run inside three shared base projects (an inventory CLI, a ledger
  library and a notification CLI) of about fifteen files each, with existing
  tests, docs and a changelog.
* Each project's README says to read CONTRIBUTING.md first. The conventions a
  change must follow (changelog entries, registering and documenting commands,
  the stdout/stderr and exit-status contract, where defaults live, migrations,
  money as Decimal, the public API) are written only there and in the existing
  code, never in the task prompt. The prompt asks for the feature.
* Every task carries two to four trap checks, named
  ``test_trap_<category>_<what>``, where category is one of ``TRAPS``.

Validation is strict: the ``near_miss`` change implements only the
requested feature, and it must pass every core check and fail every trap
check. That proves each trap check can be missed by a feature-only solution
and is caught objectively.

Tasks are split 12/12 into ``evolve`` and ``held_out``, four of each per
project. The pack was frozen before any probe or evolution code.
"""

from __future__ import annotations

from textwrap import dedent


def _t(text: str) -> str:
    return dedent(text).lstrip("\n")


TRAPS = ("deliverable", "verify_claim", "cross_file", "edge_case", "interface", "persistence")


def trap_category(test_name: str) -> str:
    """Trap category encoded in a ``test_trap_<category>_...`` name ('' if none)."""
    for category in sorted(TRAPS, key=len, reverse=True):
        if test_name.startswith(f"test_trap_{category}"):
            return category
    return ""


# ======================================================================
# Project 1: stockroom, an inventory command-line tool
# ======================================================================

STOCK_DATA = (
    '{\n'
    '  "A-100": {\n    "name": "Widget",\n    "qty": 12\n  },\n'
    '  "B-200": {\n    "name": "Gadget",\n    "qty": 3\n  },\n'
    '  "C-300": {\n    "name": "Doohickey",\n    "qty": 0\n  }\n'
    '}\n'
)

STOCK_CHANGELOG = _t("""
    # Changelog

    ## [Unreleased]

    ## [0.4.0] - 2026-10-01
    ### Added
    - `list` command.
""")

STOCK_COMMANDS_DOC = _t("""
    # Commands

    Run `python -m stockroom <command> [args]`.

    ## list

    List every item as `SKU  NAME  QTY`. `--json` prints a JSON array of objects with `sku`,
    `name` and `qty`.
""")

STOCKROOM = {
    "README.md": _t("""
        # Stockroom

        A tiny inventory tool: `python -m stockroom list`.

        Contributors: read CONTRIBUTING.md before changing anything.
    """),
    "CONTRIBUTING.md": _t("""
        # Contributing to Stockroom

        1. **Changelog.** Every user-visible change adds a bullet to CHANGELOG.md under
           `## [Unreleased]`, in the matching subsection: `### Added`, `### Changed` or
           `### Fixed`. Create the subsection if it does not exist yet.
        2. **Commands.** A command lives in `stockroom/commands/<name>_cmd.py` and exposes
           `run(args, store) -> int`. Register it in `stockroom/commands/__init__.py` and
           document it in `docs/commands.md` under a `## <name>` heading.
        3. **Output.** Results go to stdout; warnings and errors go to stderr. Exit status is 0
           on success, 1 when the request cannot be done (unknown SKU, not enough stock, ...)
           and 2 for usage errors (missing or malformed arguments). Commands that change stock
           validate everything first and save the store before returning; a rejected request
           changes nothing.
        4. **Settings.** Every setting and its default lives in `config/defaults.toml` and is
           read with `stockroom.settings.get(name)`. Do not hard-code defaults in code.
        5. **Tests.** Keep `python -m pytest tests` green.
    """),
    "CHANGELOG.md": STOCK_CHANGELOG,
    "config/defaults.toml": 'data_file = "data/stock.json"\n',
    "data/stock.json": STOCK_DATA,
    "docs/commands.md": STOCK_COMMANDS_DOC,
    "stockroom/__init__.py": "",
    "stockroom/__main__.py": _t("""
        import sys

        from .commands import COMMANDS
        from .store import Store


        def main(argv=None):
            argv = list(sys.argv[1:] if argv is None else argv)
            if not argv or argv[0] not in COMMANDS:
                names = ", ".join(sorted(COMMANDS))
                print(f"usage: python -m stockroom <command> [args]  (commands: {names})", file=sys.stderr)
                return 2
            return COMMANDS[argv[0]](argv[1:], Store())


        if __name__ == "__main__":
            raise SystemExit(main())
    """),
    "stockroom/settings.py": _t("""
        import tomllib
        from pathlib import Path

        DEFAULTS_FILE = Path(__file__).resolve().parent.parent / "config" / "defaults.toml"


        def load():
            with DEFAULTS_FILE.open("rb") as fh:
                return tomllib.load(fh)


        def get(name):
            values = load()
            if name not in values:
                raise KeyError(f"unknown setting: {name}")
            return values[name]
    """),
    "stockroom/store.py": _t("""
        import json
        from pathlib import Path

        from . import settings


        class UnknownSku(KeyError):
            pass


        class Store:
            def __init__(self, path=None):
                self.path = Path(path or settings.get("data_file"))
                self.items = json.loads(self.path.read_text(encoding="utf-8"))

            def get(self, sku):
                try:
                    return self.items[sku]
                except KeyError:
                    raise UnknownSku(sku) from None

            def save(self):
                self.path.write_text(json.dumps(self.items, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    """),
    "stockroom/commands/__init__.py": _t("""
        from . import list_cmd

        COMMANDS = {
            "list": list_cmd.run,
        }
    """),
    "stockroom/commands/list_cmd.py": _t("""
        import json


        def run(args, store):
            rows = [
                {"sku": sku, "name": item["name"], "qty": item["qty"]}
                for sku, item in sorted(store.items.items())
            ]
            if "--json" in args:
                print(json.dumps(rows))
                return 0
            for row in rows:
                print(f"{row['sku']}  {row['name']}  {row['qty']}")
            return 0
    """),
    "tests/test_list.py": _t("""
        import json
        import subprocess
        import sys


        def run(*args):
            return subprocess.run([sys.executable, "-m", "stockroom", *args], capture_output=True, text=True)


        def test_list_json():
            result = run("list", "--json")
            assert result.returncode == 0
            assert json.loads(result.stdout)[0] == {"sku": "A-100", "name": "Widget", "qty": 12}


        def test_unknown_command_is_usage_error():
            assert run("nope").returncode == 2
    """),
}

STOCK_PRELUDE = _t(r'''
    import json
    import subprocess
    import sys
    import tomllib
    from pathlib import Path

    import pytest

    from stockroom.store import Store

    DATA = {
        "A-100": {"name": "Widget", "qty": 12},
        "B-200": {"name": "Gadget", "qty": 3},
        "C-300": {"name": "Doohickey", "qty": 0},
    }


    @pytest.fixture(autouse=True)
    def fresh_data():
        Path("data/stock.json").write_text(json.dumps(DATA, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        yield


    def cli(*args):
        return subprocess.run([sys.executable, "-m", "stockroom", *args], capture_output=True, text=True)


    def stock():
        return json.loads(Path("data/stock.json").read_text(encoding="utf-8"))


    def unreleased(section):
        text = Path("CHANGELOG.md").read_text(encoding="utf-8")
        block = text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        if f"### {section}" not in block:
            return ""
        return block.split(f"### {section}", 1)[1].split("\n### ", 1)[0]


    def documented(name):
        return f"## {name}" in Path("docs/commands.md").read_text(encoding="utf-8")
''')


def _stock_registry(*names: str) -> str:
    modules = ", ".join(["list_cmd", *(f"{name}_cmd" for name in names)])
    entries = "".join(f'    "{name}": {name}_cmd.run,\n' for name in names)
    return f'from . import {modules}\n\nCOMMANDS = {{\n    "list": list_cmd.run,\n{entries}}}\n'


def _stock_changelog(section: str, bullet: str) -> str:
    return STOCK_CHANGELOG.replace(
        "## [Unreleased]\n", f"## [Unreleased]\n### {section}\n- {bullet}\n", 1
    )


def _stock_doc(name: str, text: str) -> str:
    return STOCK_COMMANDS_DOC + f"\n## {name}\n\n{text}\n"


STOCK_TASKS: list[dict] = [
    {
        "id": "stock_remove_command",
        "split": "evolve",
        "prompt": "Add a `remove` command to stockroom: `python -m stockroom remove <sku> <qty>` takes qty units out of stock.",
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_remove_decreases_stock():
                from stockroom.commands.remove_cmd import run

                store = Store()
                assert run(["A-100", "2"], store) == 0
                assert store.items["A-100"]["qty"] == 10


            def test_trap_persistence_saved():
                from stockroom.commands.remove_cmd import run

                run(["A-100", "2"], Store())
                assert stock()["A-100"]["qty"] == 10


            def test_trap_edge_case_not_enough_stock(capsys):
                from stockroom.commands.remove_cmd import run

                store = Store()
                assert run(["B-200", "5"], store) == 1
                assert store.items["B-200"]["qty"] == 3 and stock()["B-200"]["qty"] == 3
                captured = capsys.readouterr()
                assert captured.err.strip() and not captured.out.strip()


            def test_trap_cross_file_registered_and_documented():
                assert cli("remove", "A-100", "1").returncode == 0
                assert documented("remove")


            def test_trap_deliverable_changelog():
                assert "remove" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/commands/remove_cmd.py": _t("""
                import sys

                from ..store import UnknownSku


                def run(args, store):
                    if len(args) != 2 or not args[1].isdigit():
                        print("usage: python -m stockroom remove <sku> <qty>", file=sys.stderr)
                        return 2
                    sku, qty = args[0], int(args[1])
                    try:
                        item = store.get(sku)
                    except UnknownSku:
                        print(f"error: unknown SKU {sku}", file=sys.stderr)
                        return 1
                    if qty > item["qty"]:
                        print(f"error: only {item['qty']} of {sku} in stock", file=sys.stderr)
                        return 1
                    item["qty"] -= qty
                    store.save()
                    return 0
            """),
            "stockroom/commands/__init__.py": _stock_registry("remove"),
            "docs/commands.md": _stock_doc("remove", "`remove <sku> <qty>` takes qty units out of stock."),
            "CHANGELOG.md": _stock_changelog("Added", "`remove` command to take stock out."),
        },
        "near_miss": {
            "stockroom/commands/remove_cmd.py": _t("""
                def run(args, store):
                    sku, qty = args[0], int(args[1])
                    store.items[sku]["qty"] -= qty
                    return 0
            """),
        },
    },
    {
        "id": "stock_low_command",
        "split": "evolve",
        "prompt": (
            "Add a `low` command to stockroom that lists the items whose quantity is below the "
            "low-stock threshold (5 by default), in the same `SKU  NAME  QTY` format as `list`, "
            "with `--json` support like `list`."
        ),
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_low_lists_items_below_threshold(capsys):
                from stockroom.commands.low_cmd import run

                assert run([], Store()) == 0
                assert capsys.readouterr().out == "B-200  Gadget  3\nC-300  Doohickey  0\n"


            def test_trap_persistence_threshold_lives_in_config(capsys):
                from stockroom.commands.low_cmd import run

                path = Path("config/defaults.toml")
                original = path.read_text(encoding="utf-8")
                assert tomllib.loads(original)["low_stock_threshold"] == 5
                try:
                    path.write_text(original.replace("low_stock_threshold = 5", "low_stock_threshold = 1"), encoding="utf-8")
                    run([], Store())
                    assert capsys.readouterr().out == "C-300  Doohickey  0\n"
                finally:
                    path.write_text(original, encoding="utf-8")


            def test_trap_interface_json(capsys):
                from stockroom.commands.low_cmd import run

                assert run(["--json"], Store()) == 0
                assert json.loads(capsys.readouterr().out) == [
                    {"sku": "B-200", "name": "Gadget", "qty": 3},
                    {"sku": "C-300", "name": "Doohickey", "qty": 0},
                ]


            def test_trap_cross_file_registered_and_documented():
                assert cli("low").returncode == 0
                assert documented("low")


            def test_trap_deliverable_changelog():
                assert "low" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/commands/low_cmd.py": _t("""
                import json

                from .. import settings


                def run(args, store):
                    threshold = settings.get("low_stock_threshold")
                    rows = [
                        {"sku": sku, "name": item["name"], "qty": item["qty"]}
                        for sku, item in sorted(store.items.items())
                        if item["qty"] < threshold
                    ]
                    if "--json" in args:
                        print(json.dumps(rows))
                        return 0
                    for row in rows:
                        print(f"{row['sku']}  {row['name']}  {row['qty']}")
                    return 0
            """),
            "config/defaults.toml": 'data_file = "data/stock.json"\nlow_stock_threshold = 5\n',
            "stockroom/commands/__init__.py": _stock_registry("low"),
            "docs/commands.md": _stock_doc("low", "List items below `low_stock_threshold` (default 5). Supports `--json`."),
            "CHANGELOG.md": _stock_changelog("Added", "`low` command listing items below the low-stock threshold."),
        },
        "near_miss": {
            "stockroom/commands/low_cmd.py": _t("""
                def run(args, store):
                    for sku, item in sorted(store.items.items()):
                        if item["qty"] < 5:
                            print(f"{sku}  {item['name']}  {item['qty']}")
                    return 0
            """),
        },
    },
    {
        "id": "stock_list_sku_filter",
        "split": "evolve",
        "prompt": "Add a `--sku <sku>` option to stockroom's `list` command that shows only that item.",
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_list_single_sku():
                result = cli("list", "--sku", "A-100")
                assert result.returncode == 0
                assert result.stdout == "A-100  Widget  12\n"


            def test_trap_interface_unknown_sku():
                result = cli("list", "--sku", "Z-999")
                assert result.returncode == 1
                assert result.stderr.strip() and not result.stdout.strip()


            def test_trap_edge_case_flag_order():
                expected = [{"sku": "B-200", "name": "Gadget", "qty": 3}]
                assert json.loads(cli("list", "--json", "--sku", "B-200").stdout) == expected
                assert json.loads(cli("list", "--sku", "B-200", "--json").stdout) == expected


            def test_trap_cross_file_documented():
                assert "--sku" in Path("docs/commands.md").read_text(encoding="utf-8")


            def test_trap_deliverable_changelog():
                assert "--sku" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/commands/list_cmd.py": _t("""
                import json
                import sys


                def run(args, store):
                    args = list(args)
                    as_json = "--json" in args
                    if as_json:
                        args.remove("--json")
                    sku = None
                    if args:
                        if args[0] != "--sku" or len(args) != 2:
                            print("usage: python -m stockroom list [--json] [--sku SKU]", file=sys.stderr)
                            return 2
                        sku = args[1]
                        if sku not in store.items:
                            print(f"error: unknown SKU {sku}", file=sys.stderr)
                            return 1
                    rows = [
                        {"sku": key, "name": item["name"], "qty": item["qty"]}
                        for key, item in sorted(store.items.items())
                        if sku is None or key == sku
                    ]
                    if as_json:
                        print(json.dumps(rows))
                        return 0
                    for row in rows:
                        print(f"{row['sku']}  {row['name']}  {row['qty']}")
                    return 0
            """),
            "docs/commands.md": STOCK_COMMANDS_DOC.replace(
                "`name` and `qty`.\n",
                "`name` and `qty`. `--sku SKU` shows a single item; an unknown SKU exits with status 1.\n",
            ),
            "CHANGELOG.md": _stock_changelog("Added", "`list --sku SKU` shows a single item."),
        },
        "near_miss": {
            "stockroom/commands/list_cmd.py": _t("""
                import json


                def run(args, store):
                    rows = [
                        {"sku": sku, "name": item["name"], "qty": item["qty"]}
                        for sku, item in sorted(store.items.items())
                    ]
                    if args[:1] == ["--sku"]:
                        rows = [row for row in rows if row["sku"] == args[1]]
                    if "--json" in args:
                        print(json.dumps(rows))
                        return 0
                    for row in rows:
                        print(f"{row['sku']}  {row['name']}  {row['qty']}")
                    return 0
            """),
        },
    },
    {
        "id": "stock_import_csv",
        "split": "evolve",
        "prompt": (
            "Add an `import <csv-file>` command to stockroom that adds the quantities from a CSV "
            "file with the header `sku,qty` to the stock."
        ),
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def write_csv(tmp_path, body):
                path = tmp_path / "delivery.csv"
                path.write_text("sku,qty\n" + body, encoding="utf-8")
                return str(path)


            def test_import_adds_quantities(tmp_path):
                from stockroom.commands.import_cmd import run

                store = Store()
                assert run([write_csv(tmp_path, "A-100,5\nB-200,2\n")], store) == 0
                assert store.items["A-100"]["qty"] == 17 and store.items["B-200"]["qty"] == 5


            def test_trap_persistence_saved(tmp_path):
                from stockroom.commands.import_cmd import run

                run([write_csv(tmp_path, "A-100,5\n")], Store())
                assert stock()["A-100"]["qty"] == 17


            def test_trap_edge_case_bad_row_rejects_whole_file(tmp_path, capsys):
                from stockroom.commands.import_cmd import run

                store = Store()
                assert run([write_csv(tmp_path, "A-100,5\nZ-999,1\n")], store) == 1
                assert store.items["A-100"]["qty"] == 12 and stock()["A-100"]["qty"] == 12
                assert capsys.readouterr().err.strip()


            def test_trap_cross_file_registered_and_documented(tmp_path):
                assert cli("import", write_csv(tmp_path, "C-300,1\n")).returncode == 0
                assert documented("import")


            def test_trap_deliverable_changelog():
                assert "import" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/commands/import_cmd.py": _t("""
                import csv
                import sys


                def run(args, store):
                    if len(args) != 1:
                        print("usage: python -m stockroom import <csv-file>", file=sys.stderr)
                        return 2
                    with open(args[0], newline="", encoding="utf-8") as fh:
                        rows = list(csv.DictReader(fh))
                    updates = []
                    for row in rows:
                        sku, qty = row.get("sku", ""), row.get("qty", "")
                        if sku not in store.items:
                            print(f"error: unknown SKU {sku}", file=sys.stderr)
                            return 1
                        if not qty.isdigit():
                            print(f"error: bad quantity {qty!r} for {sku}", file=sys.stderr)
                            return 1
                        updates.append((sku, int(qty)))
                    for sku, qty in updates:
                        store.items[sku]["qty"] += qty
                    store.save()
                    return 0
            """),
            "stockroom/commands/__init__.py": _stock_registry("import"),
            "docs/commands.md": _stock_doc(
                "import", "`import <csv-file>` adds quantities from a `sku,qty` CSV. Any bad row rejects the whole file."
            ),
            "CHANGELOG.md": _stock_changelog("Added", "`import` command to add stock from a CSV file."),
        },
        "near_miss": {
            "stockroom/commands/import_cmd.py": _t("""
                import csv


                def run(args, store):
                    with open(args[0], newline="", encoding="utf-8") as fh:
                        for row in csv.DictReader(fh):
                            store.items[row["sku"]]["qty"] += int(row["qty"])
                    return 0
            """),
        },
    },
    {
        "id": "stock_adjust_command",
        "split": "held_out",
        "prompt": (
            "Add an `adjust <sku> <delta>` command to stockroom that changes an item's quantity by "
            "delta, which may be negative, for stock-take corrections."
        ),
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_adjust_both_directions():
                from stockroom.commands.adjust_cmd import run

                store = Store()
                assert run(["A-100", "-2"], store) == 0
                assert run(["B-200", "4"], store) == 0
                assert store.items["A-100"]["qty"] == 10 and store.items["B-200"]["qty"] == 7


            def test_trap_persistence_saved():
                from stockroom.commands.adjust_cmd import run

                run(["A-100", "-2"], Store())
                assert stock()["A-100"]["qty"] == 10


            def test_trap_edge_case_never_below_zero(capsys):
                from stockroom.commands.adjust_cmd import run

                store = Store()
                assert run(["B-200", "-4"], store) == 1
                assert store.items["B-200"]["qty"] == 3 and stock()["B-200"]["qty"] == 3
                assert capsys.readouterr().err.strip()


            def test_trap_interface_malformed_delta_is_usage_error(capsys):
                from stockroom.commands.adjust_cmd import run

                assert run(["A-100", "lots"], Store()) == 2
                assert capsys.readouterr().err.strip()


            def test_trap_cross_file_registered_and_documented():
                assert cli("adjust", "A-100", "1").returncode == 0
                assert documented("adjust")


            def test_trap_deliverable_changelog():
                assert "adjust" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/commands/adjust_cmd.py": _t("""
                import sys


                def run(args, store):
                    if len(args) != 2:
                        print("usage: python -m stockroom adjust <sku> <delta>", file=sys.stderr)
                        return 2
                    sku, raw = args
                    try:
                        delta = int(raw)
                    except ValueError:
                        print(f"usage: delta must be an integer, got {raw!r}", file=sys.stderr)
                        return 2
                    if sku not in store.items:
                        print(f"error: unknown SKU {sku}", file=sys.stderr)
                        return 1
                    item = store.items[sku]
                    if item["qty"] + delta < 0:
                        print(f"error: {sku} would go below zero", file=sys.stderr)
                        return 1
                    item["qty"] += delta
                    store.save()
                    return 0
            """),
            "stockroom/commands/__init__.py": _stock_registry("adjust"),
            "docs/commands.md": _stock_doc("adjust", "`adjust <sku> <delta>` changes the quantity by delta (may be negative)."),
            "CHANGELOG.md": _stock_changelog("Added", "`adjust` command for stock-take corrections."),
        },
        "near_miss": {
            "stockroom/commands/adjust_cmd.py": _t("""
                def run(args, store):
                    sku, delta = args[0], int(args[1])
                    store.items[sku]["qty"] += delta
                    return 0
            """),
        },
    },
    {
        "id": "stock_report_file",
        "split": "held_out",
        "prompt": (
            "Add a `report` command to stockroom that writes the full inventory to "
            "reports/stock.csv (columns sku, name, qty; sorted by SKU) and prints the path it "
            "wrote. Put the row building in stockroom/reporting.py as stock_rows(store), "
            "returning [[sku, name, qty], ...]."
        ),
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_stock_rows():
                from stockroom.reporting import stock_rows

                assert stock_rows(Store()) == [
                    ["A-100", "Widget", 12], ["B-200", "Gadget", 3], ["C-300", "Doohickey", 0],
                ]


            def test_trap_deliverable_report_has_header(capsys):
                from stockroom.commands.report_cmd import run

                assert run([], Store()) == 0
                text = Path("reports/stock.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
                assert text == "sku,name,qty\nA-100,Widget,12\nB-200,Gadget,3\nC-300,Doohickey,0\n"
                assert capsys.readouterr().out.strip().replace("\\", "/").endswith("reports/stock.csv")


            def test_trap_cross_file_registered_and_documented():
                assert cli("report").returncode == 0
                assert documented("report")


            def test_trap_deliverable_changelog():
                assert "report" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/reporting.py": _t("""
                def stock_rows(store):
                    return [[sku, item["name"], item["qty"]] for sku, item in sorted(store.items.items())]
            """),
            "stockroom/commands/report_cmd.py": _t("""
                import csv
                from pathlib import Path

                from ..reporting import stock_rows


                def run(args, store):
                    path = Path("reports") / "stock.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("w", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh, lineterminator="\\n")
                        writer.writerow(["sku", "name", "qty"])
                        writer.writerows(stock_rows(store))
                    print(path.as_posix())
                    return 0
            """),
            "stockroom/commands/__init__.py": _stock_registry("report"),
            "docs/commands.md": _stock_doc("report", "Write the inventory to `reports/stock.csv` and print the path."),
            "CHANGELOG.md": _stock_changelog("Added", "`report` command writing reports/stock.csv."),
        },
        "near_miss": {
            "stockroom/reporting.py": "@golden",
            "stockroom/commands/report_cmd.py": _t("""
                import csv
                from pathlib import Path

                from ..reporting import stock_rows


                def run(args, store):
                    path = Path("reports") / "stock.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("w", newline="", encoding="utf-8") as fh:
                        csv.writer(fh).writerows(stock_rows(store))
                    print(path)
                    return 0
            """),
        },
    },
    {
        "id": "stock_find_command",
        "split": "held_out",
        "prompt": (
            "Add a `find <text>` command to stockroom that lists the items whose name contains "
            "the text, case-insensitively, in the same output formats as `list`."
        ),
        "files": {},
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_find_case_insensitive(capsys):
                from stockroom.commands.find_cmd import run

                assert run(["WIDG"], Store()) == 0
                assert capsys.readouterr().out == "A-100  Widget  12\n"


            def test_trap_interface_json(capsys):
                from stockroom.commands.find_cmd import run

                assert run(["--json", "gad"], Store()) == 0
                assert json.loads(capsys.readouterr().out) == [{"sku": "B-200", "name": "Gadget", "qty": 3}]


            def test_trap_interface_missing_text_is_usage_error(capsys):
                from stockroom.commands.find_cmd import run

                assert run([], Store()) == 2
                assert capsys.readouterr().err.strip()


            def test_trap_cross_file_registered_and_documented():
                assert cli("find", "widget").returncode == 0
                assert documented("find")


            def test_trap_deliverable_changelog():
                assert "find" in unreleased("Added")
        '''),
        "golden": {
            "stockroom/commands/find_cmd.py": _t("""
                import json
                import sys


                def run(args, store):
                    args = list(args)
                    as_json = "--json" in args
                    if as_json:
                        args.remove("--json")
                    if len(args) != 1:
                        print("usage: python -m stockroom find [--json] <text>", file=sys.stderr)
                        return 2
                    needle = args[0].casefold()
                    rows = [
                        {"sku": sku, "name": item["name"], "qty": item["qty"]}
                        for sku, item in sorted(store.items.items())
                        if needle in item["name"].casefold()
                    ]
                    if as_json:
                        print(json.dumps(rows))
                        return 0
                    for row in rows:
                        print(f"{row['sku']}  {row['name']}  {row['qty']}")
                    return 0
            """),
            "stockroom/commands/__init__.py": _stock_registry("find"),
            "docs/commands.md": _stock_doc("find", "`find <text>` lists items whose name contains text (case-insensitive). Supports `--json`."),
            "CHANGELOG.md": _stock_changelog("Added", "`find` command to search item names."),
        },
        "near_miss": {
            "stockroom/commands/find_cmd.py": _t("""
                def run(args, store):
                    needle = args[0].lower()
                    for sku, item in sorted(store.items.items()):
                        if needle in item["name"].lower():
                            print(f"{sku}  {item['name']}  {item['qty']}")
                    return 0
            """),
        },
    },
    {
        "id": "stock_sell_oversell_fix",
        "split": "held_out",
        "prompt": (
            "stockroom's `sell` command lets stock go negative: selling 5 Gadgets when only 3 "
            "are left succeeds. Fix it."
        ),
        "files": {
            "stockroom/commands/sell_cmd.py": _t("""
                def run(args, store):
                    sku, qty = args[0], int(args[1])
                    store.get(sku)["qty"] -= qty
                    store.save()
                    return 0
            """),
            "stockroom/commands/__init__.py": _stock_registry("sell"),
            "docs/commands.md": _stock_doc("sell", "`sell <sku> <qty>` records a sale."),
            "tests/test_sell.py": _t("""
                import json
                import subprocess
                import sys
                from pathlib import Path


                def test_selling_the_last_units():
                    path = Path("data/stock.json")
                    original = path.read_text(encoding="utf-8")
                    try:
                        result = subprocess.run(
                            [sys.executable, "-m", "stockroom", "sell", "B-200", "3"], capture_output=True, text=True,
                        )
                        assert result.returncode == 0
                        assert json.loads(path.read_text(encoding="utf-8"))["B-200"]["qty"] == 0
                    finally:
                        path.write_text(original, encoding="utf-8")
            """),
        },
        "hidden_test": STOCK_PRELUDE + _t(r'''


            def test_oversell_rejected():
                from stockroom.commands.sell_cmd import run

                store = Store()
                assert run(["B-200", "5"], store) != 0
                assert store.items["B-200"]["qty"] == 3 and stock()["B-200"]["qty"] == 3


            def test_trap_verify_claim_selling_last_units_still_works():
                from stockroom.commands.sell_cmd import run

                assert run(["B-200", "3"], Store()) == 0
                assert stock()["B-200"]["qty"] == 0


            def test_trap_interface_rejection_contract(capsys):
                from stockroom.commands.sell_cmd import run

                assert run(["B-200", "5"], Store()) == 1
                captured = capsys.readouterr()
                assert captured.err.strip() and not captured.out.strip()


            def test_trap_deliverable_changelog_fixed():
                assert "sell" in unreleased("Fixed")
        '''),
        "golden": {
            "stockroom/commands/sell_cmd.py": _t("""
                import sys


                def run(args, store):
                    sku, qty = args[0], int(args[1])
                    item = store.get(sku)
                    if qty > item["qty"]:
                        print(f"error: only {item['qty']} of {sku} in stock", file=sys.stderr)
                        return 1
                    item["qty"] -= qty
                    store.save()
                    return 0
            """),
            "CHANGELOG.md": _stock_changelog("Fixed", "`sell` no longer lets stock go negative."),
        },
        "near_miss": {
            "stockroom/commands/sell_cmd.py": _t("""
                def run(args, store):
                    sku, qty = args[0], int(args[1])
                    item = store.get(sku)
                    if qty >= item["qty"]:
                        print(f"not enough {sku} in stock")
                        return 1
                    item["qty"] -= qty
                    store.save()
                    return 0
            """),
        },
    },
]


# ======================================================================
# Project 2: ledgerlite, a small double-entry ledger library
# ======================================================================

LEDGER_001 = _t("""
    CREATE TABLE accounts (name TEXT PRIMARY KEY);
    CREATE TABLE entries (
        id INTEGER PRIMARY KEY,
        account TEXT NOT NULL REFERENCES accounts(name),
        amount_minor INTEGER NOT NULL,
        memo TEXT NOT NULL DEFAULT '',
        posted_on TEXT NOT NULL
    );
""")

LEDGER_CHANGELOG = _t("""
    # Changelog

    ## [Unreleased]

    ## [1.0.0] - 2026-09-15
    ### Added
    - Accounts, postings and balances.
""")

LEDGER_INIT = _t("""
    from .db import connect
    from .errors import LedgerError, UnknownAccount
    from .ledger import balance, open_account, post

    __all__ = ["LedgerError", "UnknownAccount", "balance", "connect", "open_account", "post"]
""")

LEDGER_ERRORS = _t('''
    class LedgerError(Exception):
        """Base class for ledgerlite domain errors."""


    class UnknownAccount(LedgerError):
        pass
''')

LEDGER_LEDGER = _t('''
    from datetime import date

    from .errors import UnknownAccount
    from .money import from_minor, to_minor


    def open_account(conn, name):
        conn.execute("INSERT INTO accounts (name) VALUES (?)", (name,))
        conn.commit()


    def _require(conn, name):
        if conn.execute("SELECT 1 FROM accounts WHERE name = ?", (name,)).fetchone() is None:
            raise UnknownAccount(name)


    def post(conn, account, amount, memo="", posted_on=None):
        """Record amount (positive = debit) on account and return the entry id."""
        _require(conn, account)
        cursor = conn.execute(
            "INSERT INTO entries (account, amount_minor, memo, posted_on) VALUES (?, ?, ?, ?)",
            (account, to_minor(amount), memo, (posted_on or date.today()).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid


    def balance(conn, account):
        _require(conn, account)
        total = conn.execute(
            "SELECT COALESCE(SUM(amount_minor), 0) FROM entries WHERE account = ?", (account,)
        ).fetchone()[0]
        return from_minor(total)
''')

LEDGERLITE = {
    "README.md": _t("""
        # ledgerlite

        A small double-entry ledger library on SQLite.

        ```python
        import ledgerlite

        conn = ledgerlite.connect()
        ledgerlite.open_account(conn, "cash")
        ledgerlite.post(conn, "cash", "12.50", memo="float")
        ```

        Contributors: read CONTRIBUTING.md before changing anything.
    """),
    "CONTRIBUTING.md": _t("""
        # Contributing to ledgerlite

        1. **Changelog.** Every user-visible change adds a bullet to CHANGELOG.md under
           `## [Unreleased]`, in the matching subsection: `### Added`, `### Changed` or
           `### Fixed`. Create the subsection if it does not exist yet.
        2. **Public API.** Users import only from the top-level `ledgerlite` package. Every
           public function and exception is re-exported in `ledgerlite/__init__.py` and listed
           in `__all__`.
        3. **Money.** Amounts are `decimal.Decimal`, never float. Values returned by the API are
           Decimals with two decimal places, rounded half-up with the helpers in
           `ledgerlite/money.py`. The database stores integer cents.
        4. **Errors.** Domain errors raise subclasses of `ledgerlite.errors.LedgerError`; add a
           subclass (and export it) when you need a new one. Never let ValueError, KeyError or
           sqlite3 errors escape for bad input.
        5. **Schema.** Every schema change is a new numbered file in `ledgerlite/migrations/`
           (`002_...sql`, `003_...sql`). Never edit a migration that has shipped: existing
           databases have already applied it and will not run it again.
        6. **Tests.** Keep `python -m pytest tests` green.
    """),
    "CHANGELOG.md": LEDGER_CHANGELOG,
    "ledgerlite/__init__.py": LEDGER_INIT,
    "ledgerlite/errors.py": LEDGER_ERRORS,
    "ledgerlite/money.py": _t('''
        from decimal import ROUND_HALF_UP, Decimal

        CENT = Decimal("0.01")


        def cents(value):
            """Return value as a Decimal rounded half-up to cents."""
            return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


        def to_minor(value):
            return int(cents(value) * 100)


        def from_minor(minor):
            return (Decimal(minor) / 100).quantize(CENT)
    '''),
    "ledgerlite/db.py": _t("""
        import sqlite3
        from pathlib import Path

        MIGRATIONS = Path(__file__).resolve().parent / "migrations"


        def migrate(conn):
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
            for path in sorted(MIGRATIONS.glob("*.sql")):
                version = int(path.name.split("_", 1)[0])
                if version > current:
                    conn.executescript(path.read_text(encoding="utf-8"))
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()


        def connect(path=":memory:"):
            conn = sqlite3.connect(path)
            migrate(conn)
            return conn
    """),
    "ledgerlite/migrations/001_init.sql": LEDGER_001,
    "ledgerlite/ledger.py": LEDGER_LEDGER,
    "tests/test_ledger.py": _t("""
        from decimal import Decimal

        import pytest

        import ledgerlite


        def test_post_and_balance():
            conn = ledgerlite.connect()
            ledgerlite.open_account(conn, "cash")
            ledgerlite.post(conn, "cash", "10.005")
            ledgerlite.post(conn, "cash", Decimal("-2.50"), "lunch")
            assert ledgerlite.balance(conn, "cash") == Decimal("7.51")


        def test_unknown_account():
            conn = ledgerlite.connect()
            with pytest.raises(ledgerlite.UnknownAccount):
                ledgerlite.post(conn, "nope", 1)
    """),
}

LEDGER_PRELUDE = _t(r'''
    import sqlite3
    from datetime import date
    from decimal import Decimal
    from pathlib import Path

    import pytest

    import ledgerlite

    SHIPPED_001 = (
        "CREATE TABLE accounts (name TEXT PRIMARY KEY);\n"
        "CREATE TABLE entries (\n"
        "    id INTEGER PRIMARY KEY,\n"
        "    account TEXT NOT NULL REFERENCES accounts(name),\n"
        "    amount_minor INTEGER NOT NULL,\n"
        "    memo TEXT NOT NULL DEFAULT '',\n"
        "    posted_on TEXT NOT NULL\n"
        ");\n"
    )


    def fresh(*accounts):
        conn = ledgerlite.connect()
        for name in accounts:
            ledgerlite.open_account(conn, name)
        return conn


    def legacy(*accounts):
        """A database created by 1.0.0 (only migration 001 applied), opened by this version."""
        from ledgerlite.db import migrate

        conn = sqlite3.connect(":memory:")
        conn.executescript(SHIPPED_001)
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        for name in accounts:
            conn.execute("INSERT INTO accounts (name) VALUES (?)", (name,))
        conn.commit()
        migrate(conn)
        return conn


    def shipped_migration_untouched():
        return Path("ledgerlite/migrations/001_init.sql").read_text(encoding="utf-8") == SHIPPED_001


    def exported(name):
        return name in ledgerlite.__all__ and hasattr(ledgerlite, name)


    def unreleased(section):
        text = Path("CHANGELOG.md").read_text(encoding="utf-8")
        block = text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        if f"### {section}" not in block:
            return ""
        return block.split(f"### {section}", 1)[1].split("\n### ", 1)[0]
''')


def _ledger_changelog(section: str, bullet: str) -> str:
    return LEDGER_CHANGELOG.replace("## [Unreleased]\n", f"## [Unreleased]\n### {section}\n- {bullet}\n", 1)


def _ledger_init(*names: str, errors: tuple[str, ...] = ()) -> str:
    ledger_names = sorted({"balance", "open_account", "post", *names})
    error_names = ["LedgerError", "UnknownAccount", *errors]
    exported = sorted({*ledger_names, *error_names, "connect"})
    return (
        "from .db import connect\n"
        f"from .errors import {', '.join(error_names)}\n"
        f"from .ledger import {', '.join(ledger_names)}\n\n"
        f"__all__ = {exported!r}\n".replace("'", '"')
    )


def _ledger_error(name: str, doc: str) -> str:
    return LEDGER_ERRORS + f'\n\nclass {name}(LedgerError):\n    """{doc}"""\n'


LEDGER_TASKS: list[dict] = [
    {
        "id": "ledger_freeze_account",
        "split": "evolve",
        "prompt": (
            "Add account freezing to ledgerlite: a `freeze(conn, account)` function in "
            "ledgerlite/ledger.py, after which posting to that account is refused."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def test_frozen_account_refuses_postings():
                from ledgerlite.ledger import freeze, post

                conn = fresh("cash", "bank")
                freeze(conn, "cash")
                with pytest.raises(Exception):
                    post(conn, "cash", "1.00")
                post(conn, "bank", "1.00")
                assert ledgerlite.balance(conn, "cash") == 0


            def test_trap_persistence_existing_databases_upgrade():
                from ledgerlite.ledger import freeze, post

                assert shipped_migration_untouched()
                conn = legacy("cash")
                freeze(conn, "cash")
                with pytest.raises(Exception):
                    post(conn, "cash", "1.00")


            def test_trap_interface_domain_error():
                from ledgerlite.ledger import freeze, post

                conn = fresh("cash")
                freeze(conn, "cash")
                with pytest.raises(ledgerlite.LedgerError):
                    post(conn, "cash", "1.00")


            def test_trap_cross_file_public_api():
                assert exported("freeze")


            def test_trap_deliverable_changelog():
                assert "freeze" in unreleased("Added")
        '''),
        "golden": {
            "ledgerlite/migrations/002_account_frozen.sql": "ALTER TABLE accounts ADD COLUMN frozen INTEGER NOT NULL DEFAULT 0;\n",
            "ledgerlite/errors.py": _ledger_error("AccountFrozen", "Raised when posting to a frozen account."),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                "from .errors import UnknownAccount\n", "from .errors import AccountFrozen, UnknownAccount\n"
            ).replace(
                '    """Record amount (positive = debit) on account and return the entry id."""\n    _require(conn, account)\n',
                '    """Record amount (positive = debit) on account and return the entry id."""\n    _require(conn, account)\n'
                '    if conn.execute("SELECT frozen FROM accounts WHERE name = ?", (account,)).fetchone()[0]:\n'
                '        raise AccountFrozen(account)\n',
            ) + _t('''


                def freeze(conn, account):
                    _require(conn, account)
                    conn.execute("UPDATE accounts SET frozen = 1 WHERE name = ?", (account,))
                    conn.commit()
            '''),
            "ledgerlite/__init__.py": _ledger_init("freeze", errors=("AccountFrozen",)),
            "CHANGELOG.md": _ledger_changelog("Added", "`freeze()` to stop postings to an account."),
        },
        "near_miss": {
            "ledgerlite/migrations/001_init.sql": LEDGER_001.replace(
                "CREATE TABLE accounts (name TEXT PRIMARY KEY);",
                "CREATE TABLE accounts (name TEXT PRIMARY KEY, frozen INTEGER NOT NULL DEFAULT 0);",
            ),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                '    """Record amount (positive = debit) on account and return the entry id."""\n    _require(conn, account)\n',
                '    """Record amount (positive = debit) on account and return the entry id."""\n    _require(conn, account)\n'
                '    if conn.execute("SELECT frozen FROM accounts WHERE name = ?", (account,)).fetchone()[0]:\n'
                '        raise ValueError(f"account {account} is frozen")\n',
            ) + _t('''


                def freeze(conn, account):
                    conn.execute("UPDATE accounts SET frozen = 1 WHERE name = ?", (account,))
                    conn.commit()
            '''),
        },
    },
    {
        "id": "ledger_fee_helper",
        "split": "evolve",
        "prompt": (
            "Add `fee(amount, percent)` to ledgerlite/money.py, returning the processing fee for a "
            "payment: fee('19.99', '2.9') is 0.58."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def test_fee_value():
                from ledgerlite.money import fee

                assert Decimal(str(fee("19.99", "2.9"))) == Decimal("0.58")
                assert Decimal(str(fee("100", "1"))) == Decimal("1")


            def test_trap_edge_case_decimal_half_up():
                from ledgerlite.money import fee

                result = fee("1.15", "10")
                assert isinstance(result, Decimal)
                assert str(result) == "0.12"
                assert str(fee("100", "1")) == "1.00"


            def test_trap_cross_file_public_api():
                assert exported("fee")


            def test_trap_deliverable_changelog():
                assert "fee" in unreleased("Added")
        '''),
        "golden": {
            "ledgerlite/money.py": _t('''
                from decimal import ROUND_HALF_UP, Decimal

                CENT = Decimal("0.01")


                def cents(value):
                    """Return value as a Decimal rounded half-up to cents."""
                    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


                def to_minor(value):
                    return int(cents(value) * 100)


                def from_minor(minor):
                    return (Decimal(minor) / 100).quantize(CENT)


                def fee(amount, percent):
                    """Processing fee: percent of amount, rounded half-up to cents."""
                    return cents(Decimal(str(amount)) * Decimal(str(percent)) / 100)
            '''),
            "ledgerlite/__init__.py": LEDGER_INIT.replace(
                "from .ledger import balance, open_account, post\n",
                "from .ledger import balance, open_account, post\nfrom .money import fee\n",
            ).replace('"connect", "open_account"', '"connect", "fee", "open_account"'),
            "CHANGELOG.md": _ledger_changelog("Added", "`fee()` helper for payment processing fees."),
        },
        "near_miss": {
            "ledgerlite/money.py": _t('''
                from decimal import ROUND_HALF_UP, Decimal

                CENT = Decimal("0.01")


                def cents(value):
                    """Return value as a Decimal rounded half-up to cents."""
                    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


                def to_minor(value):
                    return int(cents(value) * 100)


                def from_minor(minor):
                    return (Decimal(minor) / 100).quantize(CENT)


                def fee(amount, percent):
                    return round(float(amount) * float(percent) / 100, 2)
            '''),
        },
    },
    {
        "id": "ledger_statement",
        "split": "evolve",
        "prompt": (
            "Add `statement(conn, account)` to ledgerlite/ledger.py, returning the account's "
            "entries in posting order as (posted_on, memo, amount, running_balance) tuples."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def make():
                conn = fresh("cash")
                ledgerlite.post(conn, "cash", "10.00", memo="a", posted_on=date(2026, 1, 2))
                ledgerlite.post(conn, "cash", "-2.50", memo="b", posted_on=date(2026, 1, 3))
                return conn


            def test_statement_rows():
                from ledgerlite.ledger import statement

                rows = statement(make(), "cash")
                assert [(row[1], float(row[2]), float(row[3])) for row in rows] == [("a", 10.0, 10.0), ("b", -2.5, 7.5)]
                assert str(rows[0][0]).startswith("2026-01-02")


            def test_trap_edge_case_money_is_decimal():
                from ledgerlite.ledger import statement

                rows = statement(make(), "cash")
                assert all(isinstance(value, Decimal) for row in rows for value in row[2:])
                assert [str(row[3]) for row in rows] == ["10.00", "7.50"]


            def test_trap_interface_unknown_account():
                from ledgerlite.ledger import statement

                with pytest.raises(ledgerlite.UnknownAccount):
                    statement(fresh(), "nope")


            def test_trap_cross_file_public_api():
                assert exported("statement")


            def test_trap_deliverable_changelog():
                assert "statement" in unreleased("Added")
        '''),
        "golden": {
            "ledgerlite/ledger.py": LEDGER_LEDGER + _t('''


                def statement(conn, account):
                    _require(conn, account)
                    rows, running = [], 0
                    for posted_on, memo, minor in conn.execute(
                        "SELECT posted_on, memo, amount_minor FROM entries WHERE account = ? ORDER BY id", (account,)
                    ):
                        running += minor
                        rows.append((posted_on, memo, from_minor(minor), from_minor(running)))
                    return rows
            '''),
            "ledgerlite/__init__.py": _ledger_init("statement"),
            "CHANGELOG.md": _ledger_changelog("Added", "`statement()` with running balances."),
        },
        "near_miss": {
            "ledgerlite/ledger.py": LEDGER_LEDGER + _t('''


                def statement(conn, account):
                    rows, running = [], 0.0
                    for posted_on, memo, minor in conn.execute(
                        "SELECT posted_on, memo, amount_minor FROM entries WHERE account = ? ORDER BY id", (account,)
                    ):
                        running += minor / 100
                        rows.append((posted_on, memo, minor / 100, running))
                    return rows
            '''),
        },
    },
    {
        "id": "ledger_transfer",
        "split": "evolve",
        "prompt": (
            "Add `transfer(conn, source, target, amount, memo='')` to ledgerlite/ledger.py that "
            "moves money from one account to another."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def test_transfer_moves_money():
                from ledgerlite.ledger import transfer

                conn = fresh("cash", "bank")
                transfer(conn, "cash", "bank", "25.00")
                assert float(ledgerlite.balance(conn, "cash")) == -25.0
                assert float(ledgerlite.balance(conn, "bank")) == 25.0


            def test_trap_edge_case_failed_transfer_posts_nothing():
                from ledgerlite.ledger import transfer

                conn = fresh("cash")
                with pytest.raises(Exception):
                    transfer(conn, "cash", "nope", "5.00")
                assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0


            def test_trap_interface_domain_errors():
                from ledgerlite.ledger import transfer

                conn = fresh("cash", "bank")
                with pytest.raises(ledgerlite.LedgerError):
                    transfer(conn, "cash", "bank", "0")
                with pytest.raises(ledgerlite.LedgerError):
                    transfer(conn, "cash", "bank", "-3")


            def test_trap_cross_file_public_api():
                assert exported("transfer")


            def test_trap_deliverable_changelog():
                assert "transfer" in unreleased("Added")
        '''),
        "golden": {
            "ledgerlite/errors.py": _ledger_error("InvalidAmount", "Raised for zero, negative or malformed amounts."),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                "from .errors import UnknownAccount\n", "from .errors import InvalidAmount, UnknownAccount\n"
            ) + _t('''


                def transfer(conn, source, target, amount, memo=""):
                    """Move amount from source to target as one atomic pair of entries."""
                    _require(conn, source)
                    _require(conn, target)
                    try:
                        minor = to_minor(amount)
                    except Exception as exc:
                        raise InvalidAmount(amount) from exc
                    if minor <= 0:
                        raise InvalidAmount(amount)
                    today = date.today().isoformat()
                    with conn:
                        conn.execute(
                            "INSERT INTO entries (account, amount_minor, memo, posted_on) VALUES (?, ?, ?, ?)",
                            (source, -minor, memo, today),
                        )
                        conn.execute(
                            "INSERT INTO entries (account, amount_minor, memo, posted_on) VALUES (?, ?, ?, ?)",
                            (target, minor, memo, today),
                        )
            '''),
            "ledgerlite/__init__.py": _ledger_init("transfer", errors=("InvalidAmount",)),
            "CHANGELOG.md": _ledger_changelog("Added", "`transfer()` between accounts."),
        },
        "near_miss": {
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                "from .money import from_minor, to_minor\n", "from .money import cents, from_minor, to_minor\n"
            ) + _t('''


                def transfer(conn, source, target, amount, memo=""):
                    post(conn, source, -cents(amount), memo)
                    post(conn, target, amount, memo)
            '''),
        },
    },
    {
        "id": "ledger_account_currency",
        "split": "held_out",
        "prompt": (
            "Accounts in ledgerlite get a currency: `open_account(conn, name, currency='USD')`, and "
            "a new `currency(conn, account)` in ledgerlite/ledger.py returns it. Currency codes are "
            "three upper-case letters."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def test_currency_round_trip():
                from ledgerlite.ledger import currency, open_account

                conn = fresh()
                open_account(conn, "eur", currency="EUR")
                open_account(conn, "cash")
                assert currency(conn, "eur") == "EUR"
                assert currency(conn, "cash") == "USD"


            def test_trap_persistence_existing_databases_upgrade():
                from ledgerlite.ledger import currency

                assert shipped_migration_untouched()
                assert currency(legacy("cash"), "cash") == "USD"


            def test_trap_interface_invalid_code():
                from ledgerlite.ledger import open_account

                conn = fresh()
                for code in ("eu", "EURO", "eur"):
                    with pytest.raises(ledgerlite.LedgerError):
                        open_account(conn, f"acct-{code}", currency=code)
                assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0


            def test_trap_cross_file_public_api():
                assert exported("currency")


            def test_trap_deliverable_changelog():
                assert "currency" in unreleased("Added").lower()
        '''),
        "golden": {
            "ledgerlite/migrations/002_account_currency.sql": "ALTER TABLE accounts ADD COLUMN currency TEXT NOT NULL DEFAULT 'USD';\n",
            "ledgerlite/errors.py": _ledger_error("InvalidCurrency", "Raised for currency codes that are not three upper-case letters."),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                "from .errors import UnknownAccount\n", "import re\n\nfrom .errors import InvalidCurrency, UnknownAccount\n"
            ).replace(
                'def open_account(conn, name):\n    conn.execute("INSERT INTO accounts (name) VALUES (?)", (name,))\n',
                'def open_account(conn, name, currency="USD"):\n'
                '    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):\n'
                '        raise InvalidCurrency(currency)\n'
                '    conn.execute("INSERT INTO accounts (name, currency) VALUES (?, ?)", (name, currency))\n',
            ) + _t('''


                def currency(conn, account):
                    _require(conn, account)
                    return conn.execute("SELECT currency FROM accounts WHERE name = ?", (account,)).fetchone()[0]
            '''),
            "ledgerlite/__init__.py": _ledger_init("currency", errors=("InvalidCurrency",)),
            "CHANGELOG.md": _ledger_changelog("Added", "Account currencies (`open_account(..., currency=)`, `currency()`)."),
        },
        "near_miss": {
            "ledgerlite/migrations/001_init.sql": LEDGER_001.replace(
                "CREATE TABLE accounts (name TEXT PRIMARY KEY);",
                "CREATE TABLE accounts (name TEXT PRIMARY KEY, currency TEXT NOT NULL DEFAULT 'USD');",
            ),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                'def open_account(conn, name):\n    conn.execute("INSERT INTO accounts (name) VALUES (?)", (name,))\n',
                'def open_account(conn, name, currency="USD"):\n'
                '    conn.execute("INSERT INTO accounts (name, currency) VALUES (?, ?)", (name, currency))\n',
            ) + _t('''


                def currency(conn, account):
                    return conn.execute("SELECT currency FROM accounts WHERE name = ?", (account,)).fetchone()[0]
            '''),
        },
    },
    {
        "id": "ledger_trial_balance_fix",
        "split": "held_out",
        "prompt": "ledgerlite's trial balance shows totals like 0.30000000000000004 in exports. Fix it.",
        "files": {
            "ledgerlite/reports.py": _t('''
                def trial_balance(conn):
                    """Total per account, including accounts without entries."""
                    totals = {name: 0.0 for (name,) in conn.execute("SELECT name FROM accounts ORDER BY name")}
                    for account, minor in conn.execute("SELECT account, amount_minor FROM entries ORDER BY id"):
                        totals[account] += minor / 100
                    return totals
            '''),
            "tests/test_reports.py": _t("""
                import ledgerlite
                from ledgerlite.reports import trial_balance


                def test_trial_balance_lists_every_account():
                    conn = ledgerlite.connect()
                    ledgerlite.open_account(conn, "cash")
                    ledgerlite.open_account(conn, "empty")
                    ledgerlite.post(conn, "cash", "1.25")
                    totals = trial_balance(conn)
                    assert set(totals) == {"cash", "empty"}
                    assert totals["empty"] == 0
            """),
        },
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def totals():
                from ledgerlite.reports import trial_balance

                conn = fresh("cash", "empty")
                ledgerlite.post(conn, "cash", "0.10")
                ledgerlite.post(conn, "cash", "0.20")
                return trial_balance(conn)


            def test_no_float_noise():
                result = totals()
                assert float(result["cash"]) == 0.3
                assert str(result["cash"]) in ("0.3", "0.30")


            def test_trap_edge_case_money_is_decimal():
                result = totals()
                assert isinstance(result["cash"], Decimal) and str(result["cash"]) == "0.30"
                assert str(result["empty"]) == "0.00"


            def test_trap_deliverable_changelog_fixed():
                assert "trial" in unreleased("Fixed").lower()
        '''),
        "golden": {
            "ledgerlite/reports.py": _t('''
                from .money import from_minor


                def trial_balance(conn):
                    """Total per account, including accounts without entries."""
                    rows = conn.execute(
                        "SELECT a.name, COALESCE(SUM(e.amount_minor), 0) FROM accounts a "
                        "LEFT JOIN entries e ON e.account = a.name GROUP BY a.name ORDER BY a.name"
                    )
                    return {name: from_minor(total) for name, total in rows}
            '''),
            "CHANGELOG.md": _ledger_changelog("Fixed", "Trial balance totals no longer show float rounding noise."),
        },
        "near_miss": {
            "ledgerlite/reports.py": _t('''
                def trial_balance(conn):
                    """Total per account, including accounts without entries."""
                    totals = {name: 0.0 for (name,) in conn.execute("SELECT name FROM accounts ORDER BY name")}
                    for account, minor in conn.execute("SELECT account, amount_minor FROM entries ORDER BY id"):
                        totals[account] = round(totals[account] + minor / 100, 2)
                    return totals
            '''),
        },
    },
    {
        "id": "ledger_close_period",
        "split": "held_out",
        "prompt": (
            "Add period closing to ledgerlite: `close_period(conn, through)` in ledgerlite/ledger.py. "
            "Once a period is closed, entries dated on or before `through` can no longer be posted."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def test_closed_period_refuses_old_entries():
                from ledgerlite.ledger import close_period, post

                conn = fresh("cash")
                close_period(conn, date(2026, 6, 30))
                with pytest.raises(Exception):
                    post(conn, "cash", "1.00", posted_on=date(2026, 6, 1))
                post(conn, "cash", "1.00", posted_on=date(2026, 7, 1))


            def test_trap_persistence_survives_reconnect(tmp_path):
                from ledgerlite.ledger import close_period, post

                assert shipped_migration_untouched()
                path = str(tmp_path / "books.db")
                conn = ledgerlite.connect(path)
                ledgerlite.open_account(conn, "cash")
                close_period(conn, date(2026, 6, 30))
                conn.close()
                reopened = ledgerlite.connect(path)
                with pytest.raises(Exception):
                    post(reopened, "cash", "1.00", posted_on=date(2026, 6, 1))
                legacy("cash")


            def test_trap_interface_domain_error():
                from ledgerlite.ledger import close_period, post

                conn = fresh("cash")
                close_period(conn, date(2026, 6, 30))
                with pytest.raises(ledgerlite.LedgerError):
                    post(conn, "cash", "1.00", posted_on=date(2026, 6, 30))


            def test_trap_cross_file_public_api():
                assert exported("close_period")


            def test_trap_deliverable_changelog():
                assert "close" in unreleased("Added").lower()
        '''),
        "golden": {
            "ledgerlite/migrations/002_closed_periods.sql": "CREATE TABLE closed_periods (through TEXT NOT NULL);\n",
            "ledgerlite/errors.py": _ledger_error("PeriodClosed", "Raised when posting into a closed period."),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                "from .errors import UnknownAccount\n", "from .errors import PeriodClosed, UnknownAccount\n"
            ).replace(
                '    _require(conn, account)\n    cursor = conn.execute(\n',
                '    _require(conn, account)\n'
                '    day = (posted_on or date.today()).isoformat()\n'
                '    closed = conn.execute("SELECT MAX(through) FROM closed_periods").fetchone()[0]\n'
                '    if closed is not None and day <= closed:\n'
                '        raise PeriodClosed(day)\n'
                '    cursor = conn.execute(\n',
            ) + _t('''


                def close_period(conn, through):
                    conn.execute("INSERT INTO closed_periods (through) VALUES (?)", (through.isoformat(),))
                    conn.commit()
            '''),
            "ledgerlite/__init__.py": _ledger_init("close_period", errors=("PeriodClosed",)),
            "CHANGELOG.md": _ledger_changelog("Added", "`close_period()` to lock past periods."),
        },
        "near_miss": {
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                '    _require(conn, account)\n    cursor = conn.execute(\n',
                '    _require(conn, account)\n'
                '    closed = _CLOSED.get(id(conn))\n'
                '    if closed is not None and (posted_on or date.today()) <= closed:\n'
                '        raise ValueError("period is closed")\n'
                '    cursor = conn.execute(\n',
            ).replace("from .money import from_minor, to_minor\n", "from .money import from_minor, to_minor\n\n_CLOSED = {}\n") + _t('''


                def close_period(conn, through):
                    _CLOSED[id(conn)] = through
            '''),
        },
    },
    {
        "id": "ledger_entry_tags",
        "split": "held_out",
        "prompt": (
            "Entries in ledgerlite can carry tags: `post()` gets a `tags` argument (default no "
            "tags), and a new `entries_with_tag(conn, tag)` in ledgerlite/ledger.py returns the ids "
            "of matching entries in posting order."
        ),
        "files": {},
        "hidden_test": LEDGER_PRELUDE + _t(r'''


            def test_tags_round_trip():
                from ledgerlite.ledger import entries_with_tag, post

                conn = fresh("cash")
                first = post(conn, "cash", "5.00", tags=("travel", "q3"))
                post(conn, "cash", "1.00", tags=("office",))
                third = post(conn, "cash", "2.00", tags=("travel",))
                assert entries_with_tag(conn, "travel") == [first, third]


            def test_trap_verify_claim_positional_memo_still_works():
                conn = fresh("cash")
                ledgerlite.post(conn, "cash", "5.00", "lunch")
                assert conn.execute("SELECT memo FROM entries").fetchone()[0] == "lunch"


            def test_trap_persistence_existing_databases_upgrade():
                from ledgerlite.ledger import entries_with_tag, post

                assert shipped_migration_untouched()
                conn = legacy("cash")
                entry = post(conn, "cash", "5.00", tags=("travel",))
                assert entries_with_tag(conn, "travel") == [entry]


            def test_trap_cross_file_public_api():
                assert exported("entries_with_tag")


            def test_trap_deliverable_changelog():
                assert "tag" in unreleased("Added").lower()
        '''),
        "golden": {
            "ledgerlite/migrations/002_entry_tags.sql": _t("""
                CREATE TABLE entry_tags (
                    entry_id INTEGER NOT NULL REFERENCES entries(id),
                    tag TEXT NOT NULL
                );
            """),
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                'def post(conn, account, amount, memo="", posted_on=None):',
                'def post(conn, account, amount, memo="", posted_on=None, tags=()):',
            ).replace(
                "    conn.commit()\n    return cursor.lastrowid\n",
                "    conn.executemany(\n"
                '        "INSERT INTO entry_tags (entry_id, tag) VALUES (?, ?)", [(cursor.lastrowid, tag) for tag in tags]\n'
                "    )\n"
                "    conn.commit()\n    return cursor.lastrowid\n",
            ) + _t('''


                def entries_with_tag(conn, tag):
                    rows = conn.execute("SELECT entry_id FROM entry_tags WHERE tag = ? ORDER BY entry_id", (tag,))
                    return [row[0] for row in rows]
            '''),
            "ledgerlite/__init__.py": _ledger_init("entries_with_tag"),
            "CHANGELOG.md": _ledger_changelog("Added", "Entry tags (`post(..., tags=)`, `entries_with_tag()`)."),
        },
        "near_miss": {
            "ledgerlite/migrations/001_init.sql": LEDGER_001 + "CREATE TABLE entry_tags (entry_id INTEGER NOT NULL, tag TEXT NOT NULL);\n",
            "ledgerlite/ledger.py": LEDGER_LEDGER.replace(
                'def post(conn, account, amount, memo="", posted_on=None):',
                'def post(conn, account, amount, tags=(), memo="", posted_on=None):',
            ).replace(
                "    conn.commit()\n    return cursor.lastrowid\n",
                "    for tag in tags:\n"
                '        conn.execute("INSERT INTO entry_tags (entry_id, tag) VALUES (?, ?)", (cursor.lastrowid, tag))\n'
                "    conn.commit()\n    return cursor.lastrowid\n",
            ) + _t('''


                def entries_with_tag(conn, tag):
                    rows = conn.execute("SELECT entry_id FROM entry_tags WHERE tag = ? ORDER BY entry_id", (tag,))
                    return [row[0] for row in rows]
            '''),
        },
    },
]


# ======================================================================
# Project 3: notifykit, a notification command-line tool
# ======================================================================

NOTIFY_CHANGELOG = _t("""
    # Changelog

    ## [Unreleased]

    ## [0.9.0] - 2026-08-30
    ### Added
    - `send` with the email channel and text templates.
""")

NOTIFY_CHANNELS_DOC = _t("""
    # Channels

    ## email

    Delivers to an email address (it must contain `@`). Writes to `outbox/email.jsonl`.
""")

NOTIFY_CLI_DOC = _t("""
    # CLI

    ## send

    `python -m notifykit send --channel CHANNEL --to RECIPIENT --template NAME [--var KEY=VALUE ...]`
    renders the template and delivers it, then prints a JSON receipt.
""")

NOTIFY_MAIN = _t("""
    import argparse
    import json
    import sys
    from pathlib import Path

    from . import settings
    from .channels import CHANNELS
    from .errors import NotifyError
    from .templates import render


    def build_parser():
        parser = argparse.ArgumentParser(prog="notifykit")
        sub = parser.add_subparsers(dest="command", required=True)
        send = sub.add_parser("send")
        send.add_argument("--channel", required=True)
        send.add_argument("--to", required=True)
        send.add_argument("--template", required=True)
        send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
        return parser


    def parse_vars(items):
        variables = {}
        for item in items:
            key, sep, value = item.partition("=")
            if not sep:
                raise NotifyError(f"bad --var {item!r}, expected KEY=VALUE")
            variables[key] = value
        return variables


    def send(args):
        if args.channel not in CHANNELS:
            raise NotifyError(f"unknown channel {args.channel!r}")
        channel = CHANNELS[args.channel]()
        message = render(args.template, parse_vars(args.var))
        channel.validate_recipient(args.to)
        return channel.deliver(args.to, message, Path(settings.get("outbox_dir")))


    def main(argv=None):
        args = build_parser().parse_args(argv)
        try:
            receipt = send(args)
        except NotifyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(receipt))
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
""")

NOTIFY_EMAIL = _t("""
    import json

    from .. import settings
    from ..errors import DeliveryError


    class EmailChannel:
        name = "email"

        def validate_recipient(self, to):
            if "@" not in to:
                raise DeliveryError(f"not an email address: {to!r}")

        def deliver(self, to, message, outbox):
            outbox.mkdir(parents=True, exist_ok=True)
            record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}
            with (outbox / "email.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\\n")
            return {"channel": self.name, "to": to, "status": "queued"}
""")

NOTIFY_TEMPLATES = _t("""
        import re
        from pathlib import Path

        from .errors import TemplateError

        TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
        _FIELD = re.compile(r"\\{(\\w+)\\}")


        def render(name, variables):
            path = TEMPLATES / f"{name}.txt"
            if not path.is_file():
                raise TemplateError(f"unknown template {name!r}")
            text = path.read_text(encoding="utf-8")
            missing = sorted(set(_FIELD.findall(text)) - set(variables))
            if missing:
                raise TemplateError(f"missing variables: {', '.join(missing)}")
            return _FIELD.sub(lambda match: str(variables[match.group(1)]), text)
    """)

NOTIFY_DEFAULTS = 'sender = "noreply@example.com"\noutbox_dir = "outbox"\n'

NOTIFYKIT = {
    "README.md": _t("""
        # notifykit

        Send templated notifications: `python -m notifykit send --channel email --to ada@example.com --template welcome --var name=Ada --var product=Kit`.

        Contributors: read CONTRIBUTING.md before changing anything.
    """),
    "CONTRIBUTING.md": _t("""
        # Contributing to notifykit

        1. **Changelog.** Every user-visible change adds a bullet to CHANGELOG.md under
           `## [Unreleased]`, in the matching subsection: `### Added`, `### Changed` or
           `### Fixed`. Create the subsection if it does not exist yet.
        2. **Channels.** A channel is a class in `notifykit/channels/<name>.py` with a `name`
           attribute, `validate_recipient(to)` and `deliver(to, message, outbox)`. Register it in
           `notifykit/channels/__init__.py` and document it in `docs/channels.md` under a
           `## <name>` heading. Invalid recipients and undeliverable messages raise
           `notifykit.errors.DeliveryError`.
        3. **CLI.** Subcommands and their options are documented in `docs/cli.md`. stdout carries
           only machine-readable output (a single JSON document); human messages, warnings and
           errors go to stderr. Exit status is 0 on success, 1 when the notification cannot be
           sent, 2 for usage errors.
        4. **Settings.** Defaults live in `config/defaults.toml` and are read with
           `notifykit.settings.get(name)`, which also applies `NOTIFYKIT_<NAME>` environment
           overrides. Do not hard-code defaults in code.
        5. **All or nothing.** Validate every recipient and render the message before delivering
           anything; a rejected request delivers nothing.
        6. **Tests.** Keep `python -m pytest tests` green.
    """),
    "CHANGELOG.md": NOTIFY_CHANGELOG,
    "config/defaults.toml": NOTIFY_DEFAULTS,
    "docs/channels.md": NOTIFY_CHANNELS_DOC,
    "docs/cli.md": NOTIFY_CLI_DOC,
    "templates/welcome.txt": "Hello {name}, welcome to {product}!\n",
    "templates/reset.txt": "Hi {name}, use code {code} to reset your password.\n",
    "notifykit/__init__.py": "",
    "notifykit/__main__.py": NOTIFY_MAIN,
    "notifykit/errors.py": _t('''
        class NotifyError(Exception):
            """Base class for notifykit errors reported to the user."""


        class DeliveryError(NotifyError):
            pass


        class TemplateError(NotifyError):
            pass
    '''),
    "notifykit/clock.py": _t('''
        import os
        from datetime import datetime


        def now():
            """Current local time. Tests set NOTIFYKIT_FAKE_NOW to an ISO timestamp."""
            fake = os.environ.get("NOTIFYKIT_FAKE_NOW")
            return datetime.fromisoformat(fake) if fake else datetime.now()
    '''),
    "notifykit/settings.py": _t("""
        import os
        import tomllib
        from pathlib import Path

        DEFAULTS_FILE = Path(__file__).resolve().parent.parent / "config" / "defaults.toml"


        def get(name):
            with DEFAULTS_FILE.open("rb") as fh:
                defaults = tomllib.load(fh)
            if name not in defaults:
                raise KeyError(f"unknown setting: {name}")
            default = defaults[name]
            raw = os.environ.get(f"NOTIFYKIT_{name.upper()}")
            if raw is None:
                return default
            if isinstance(default, bool):
                return raw.lower() in ("1", "true", "yes")
            if isinstance(default, int):
                return int(raw)
            return raw
    """),
    "notifykit/templates.py": NOTIFY_TEMPLATES,
    "notifykit/channels/__init__.py": _t("""
        from .email import EmailChannel

        CHANNELS = {
            "email": EmailChannel,
        }
    """),
    "notifykit/channels/email.py": NOTIFY_EMAIL,
    "tests/test_send.py": _t("""
        import json
        import os
        import shutil
        import subprocess
        import sys


        def cli(*args):
            env = {key: value for key, value in os.environ.items() if not key.startswith("NOTIFYKIT_")}
            env["NOTIFYKIT_FAKE_NOW"] = "2026-01-01T12:00:00"
            return subprocess.run([sys.executable, "-m", "notifykit", *args], capture_output=True, text=True, env=env)


        def setup_function():
            shutil.rmtree("outbox", ignore_errors=True)


        def test_send_email_prints_receipt():
            result = cli("send", "--channel", "email", "--to", "ada@example.com", "--template", "welcome",
                         "--var", "name=Ada", "--var", "product=Kit")
            assert result.returncode == 0
            assert json.loads(result.stdout)["status"] == "queued"


        def test_bad_recipient():
            result = cli("send", "--channel", "email", "--to", "nobody", "--template", "welcome",
                         "--var", "name=A", "--var", "product=B")
            assert result.returncode == 1 and result.stdout == ""


        def test_malformed_var_is_reported():
            result = cli("send", "--channel", "email", "--to", "ada@example.com", "--template", "welcome", "--var", "name")
            assert result.returncode == 1
            assert result.stderr.startswith("error:")
    """),
}

NOTIFY_PRELUDE = _t(r'''
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tomllib
    from datetime import datetime
    from pathlib import Path

    import pytest

    WELCOME = ("--template", "welcome", "--var", "name=Ada", "--var", "product=Kit")


    @pytest.fixture(autouse=True)
    def clean(monkeypatch):
        shutil.rmtree("outbox", ignore_errors=True)
        for key in list(os.environ):
            if key.startswith("NOTIFYKIT_"):
                monkeypatch.delenv(key)
        yield
        shutil.rmtree("outbox", ignore_errors=True)


    def cli(*args, env=None):
        full = {key: value for key, value in os.environ.items() if not key.startswith("NOTIFYKIT_")}
        full["NOTIFYKIT_FAKE_NOW"] = "2026-01-01T12:00:00"
        full.update(env or {})
        return subprocess.run([sys.executable, "-m", "notifykit", *args], capture_output=True, text=True, env=full)


    def outbox(channel):
        path = Path("outbox") / f"{channel}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def documented(doc, text):
        return text in Path(f"docs/{doc}.md").read_text(encoding="utf-8")


    def unreleased(section):
        text = Path("CHANGELOG.md").read_text(encoding="utf-8")
        block = text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        if f"### {section}" not in block:
            return ""
        return block.split(f"### {section}", 1)[1].split("\n### ", 1)[0]
''')


def _notify_changelog(section: str, bullet: str) -> str:
    return NOTIFY_CHANGELOG.replace("## [Unreleased]\n", f"## [Unreleased]\n### {section}\n- {bullet}\n", 1)


def _notify_registry(*pairs: tuple[str, str]) -> str:
    imports = "".join(f"from .{module} import {cls}\n" for module, cls in pairs)
    entries = "".join(f'    "{module}": {cls},\n' for module, cls in pairs)
    return (
        "from .email import EmailChannel\n" + imports
        + '\nCHANNELS = {\n    "email": EmailChannel,\n' + entries + "}\n"
    )


NOTIFY_TASKS: list[dict] = [
    {
        "id": "notify_webhook_channel",
        "split": "evolve",
        "prompt": (
            "Add a `webhook` channel to notifykit. Recipients are http or https URLs; delivering "
            "appends a record to outbox/webhook.jsonl."
        ),
        "files": {},
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_webhook_delivery():
                from notifykit.channels.webhook import WebhookChannel

                channel = WebhookChannel()
                channel.validate_recipient("https://hooks.example.com/x")
                channel.deliver("https://hooks.example.com/x", "hi", Path("outbox"))
                assert outbox("webhook")[0]["message"] == "hi"


            def test_trap_interface_bad_url_is_a_delivery_error():
                from notifykit.channels.webhook import WebhookChannel
                from notifykit.errors import DeliveryError

                for bad in ("not-a-url", "ftp://files.example.com/x"):
                    with pytest.raises(DeliveryError):
                        WebhookChannel().validate_recipient(bad)


            def test_trap_cross_file_registered_and_documented():
                result = cli("send", "--channel", "webhook", "--to", "https://hooks.example.com/x", *WELCOME)
                assert result.returncode == 0 and json.loads(result.stdout)
                assert documented("channels", "## webhook")


            def test_trap_deliverable_changelog():
                assert "webhook" in unreleased("Added")
        '''),
        "golden": {
            "notifykit/channels/webhook.py": _t("""
                import json
                from urllib.parse import urlparse

                from ..errors import DeliveryError


                class WebhookChannel:
                    name = "webhook"

                    def validate_recipient(self, to):
                        parsed = urlparse(to)
                        if parsed.scheme not in ("http", "https") or not parsed.netloc:
                            raise DeliveryError(f"not an http(s) URL: {to!r}")

                    def deliver(self, to, message, outbox):
                        outbox.mkdir(parents=True, exist_ok=True)
                        with (outbox / "webhook.jsonl").open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"channel": self.name, "to": to, "message": message}) + "\\n")
                        return {"channel": self.name, "to": to, "status": "queued"}
            """),
            "notifykit/channels/__init__.py": _notify_registry(("webhook", "WebhookChannel")),
            "docs/channels.md": NOTIFY_CHANNELS_DOC + "\n## webhook\n\nPosts to an http(s) URL. Writes to `outbox/webhook.jsonl`.\n",
            "CHANGELOG.md": _notify_changelog("Added", "`webhook` channel."),
        },
        "near_miss": {
            "notifykit/channels/webhook.py": _t("""
                import json


                class WebhookChannel:
                    name = "webhook"

                    def validate_recipient(self, to):
                        pass

                    def deliver(self, to, message, outbox):
                        outbox.mkdir(parents=True, exist_ok=True)
                        with (outbox / "webhook.jsonl").open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"channel": self.name, "to": to, "message": message}) + "\\n")
                        return {"channel": self.name, "to": to, "status": "queued"}
            """),
        },
    },
    {
        "id": "notify_templates_command",
        "split": "evolve",
        "prompt": "Add a `templates` subcommand to notifykit's CLI that lists the available templates.",
        "files": {},
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_templates_listed():
                result = cli("templates")
                assert result.returncode == 0
                assert "welcome" in result.stdout and "reset" in result.stdout


            def test_trap_interface_stdout_is_json():
                assert json.loads(cli("templates").stdout) == ["reset", "welcome"]


            def test_trap_cross_file_documented():
                assert documented("cli", "## templates")


            def test_trap_deliverable_changelog():
                assert "templates" in unreleased("Added")
        '''),
        "golden": {
            "notifykit/templates.py": NOTIFY_TEMPLATES + _t("""


                def available():
                    return sorted(path.stem for path in TEMPLATES.glob("*.txt"))
            """),
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n    return parser\n',
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n'
                '    sub.add_parser("templates")\n    return parser\n',
            ).replace(
                "from .templates import render\n", "from .templates import available, render\n"
            ).replace(
                "    args = build_parser().parse_args(argv)\n",
                "    args = build_parser().parse_args(argv)\n"
                '    if args.command == "templates":\n'
                "        print(json.dumps(available()))\n"
                "        return 0\n",
            ),
            "docs/cli.md": NOTIFY_CLI_DOC + "\n## templates\n\n`python -m notifykit templates` prints the template names as a JSON array.\n",
            "CHANGELOG.md": _notify_changelog("Added", "`templates` subcommand."),
        },
        "near_miss": {
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n    return parser\n',
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n'
                '    sub.add_parser("templates")\n    return parser\n',
            ).replace(
                "    args = build_parser().parse_args(argv)\n",
                "    args = build_parser().parse_args(argv)\n"
                '    if args.command == "templates":\n'
                '        for path in sorted(Path("templates").glob("*.txt")):\n'
                "            print(path.stem)\n"
                "        return 0\n",
            ),
        },
    },
    {
        "id": "notify_quiet_hours",
        "split": "evolve",
        "prompt": (
            "notifykit must not send during quiet hours. Add `in_quiet_hours(moment)` in a new "
            "notifykit/quiet.py and make `send` refuse while it is true, using notifykit.clock.now() "
            "for the current time. Quiet hours come from a `quiet_hours` setting written as "
            "\"START-END\" in whole hours, \"22-7\" by default."
        ),
        "files": {},
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_default_quiet_hours():
                from notifykit.quiet import in_quiet_hours

                assert in_quiet_hours(datetime(2026, 1, 1, 23, 0))
                assert in_quiet_hours(datetime(2026, 1, 1, 6, 59))
                assert not in_quiet_hours(datetime(2026, 1, 1, 7, 0))
                assert not in_quiet_hours(datetime(2026, 1, 1, 12, 0))


            def test_trap_persistence_setting_in_config(monkeypatch):
                from notifykit.quiet import in_quiet_hours

                with open("config/defaults.toml", "rb") as fh:
                    assert tomllib.load(fh)["quiet_hours"] == "22-7"
                monkeypatch.setenv("NOTIFYKIT_QUIET_HOURS", "9-17")
                assert in_quiet_hours(datetime(2026, 1, 1, 12, 0))
                assert not in_quiet_hours(datetime(2026, 1, 1, 23, 0))


            def test_trap_interface_refusal_contract():
                result = cli("send", "--channel", "email", "--to", "ada@example.com", *WELCOME,
                             env={"NOTIFYKIT_FAKE_NOW": "2026-01-01T23:30:00"})
                assert result.returncode == 1
                assert result.stderr.strip() and result.stdout == ""
                assert outbox("email") == []
                daytime = cli("send", "--channel", "email", "--to", "ada@example.com", *WELCOME,
                              env={"NOTIFYKIT_FAKE_NOW": "2026-01-01T12:00:00"})
                assert daytime.returncode == 0


            def test_trap_deliverable_changelog():
                assert "quiet" in unreleased("Added").lower()
        '''),
        "golden": {
            "notifykit/quiet.py": _t("""
                from . import settings


                def in_quiet_hours(moment):
                    start, end = (int(part) for part in settings.get("quiet_hours").split("-"))
                    hour = moment.hour
                    if start <= end:
                        return start <= hour < end
                    return hour >= start or hour < end
            """),
            "config/defaults.toml": NOTIFY_DEFAULTS + 'quiet_hours = "22-7"\n',
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                "from . import settings\n", "from . import clock, settings\n"
            ).replace(
                "from .templates import render\n", "from .quiet import in_quiet_hours\nfrom .templates import render\n"
            ).replace(
                "    channel = CHANNELS[args.channel]()\n",
                "    if in_quiet_hours(clock.now()):\n"
                '        raise NotifyError("quiet hours: not sending now")\n'
                "    channel = CHANNELS[args.channel]()\n",
            ),
            "CHANGELOG.md": _notify_changelog("Added", "Quiet hours (`quiet_hours` setting, default 22-7)."),
        },
        "near_miss": {
            "notifykit/quiet.py": _t("""
                QUIET_START, QUIET_END = 22, 7


                def in_quiet_hours(moment):
                    return moment.hour >= QUIET_START or moment.hour < QUIET_END
            """),
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                "from . import settings\n", "from . import clock, settings\n"
            ).replace(
                "from .templates import render\n", "from .quiet import in_quiet_hours\nfrom .templates import render\n"
            ).replace(
                "    args = build_parser().parse_args(argv)\n",
                "    args = build_parser().parse_args(argv)\n"
                "    if in_quiet_hours(clock.now()):\n"
                '        print("Quiet hours, message not sent.")\n'
                "        return 0\n",
            ),
        },
    },
    {
        "id": "notify_dry_run",
        "split": "evolve",
        "prompt": "Add a `--dry-run` flag to `notifykit send` that shows what would be sent without delivering it.",
        "files": {},
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_dry_run_does_not_deliver():
                result = cli("send", "--dry-run", "--channel", "email", "--to", "ada@example.com", *WELCOME)
                assert result.returncode == 0
                assert outbox("email") == []
                assert "Hello Ada, welcome to Kit!" in result.stdout + result.stderr


            def test_trap_interface_stdout_is_json():
                result = cli("send", "--dry-run", "--channel", "email", "--to", "ada@example.com", *WELCOME)
                assert "Hello Ada, welcome to Kit!" in json.dumps(json.loads(result.stdout))


            def test_trap_edge_case_dry_run_still_validates():
                result = cli("send", "--dry-run", "--channel", "email", "--to", "nobody", *WELCOME)
                assert result.returncode == 1


            def test_trap_cross_file_documented():
                assert documented("cli", "--dry-run")


            def test_trap_deliverable_changelog():
                assert "--dry-run" in unreleased("Added")
        '''),
        "golden": {
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n',
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n'
                '    send.add_argument("--dry-run", action="store_true")\n',
            ).replace(
                "    channel.validate_recipient(args.to)\n    return channel.deliver(",
                "    channel.validate_recipient(args.to)\n"
                "    if args.dry_run:\n"
                '        return {"channel": args.channel, "to": args.to, "message": message, "dry_run": True}\n'
                "    return channel.deliver(",
            ),
            "docs/cli.md": NOTIFY_CLI_DOC.rstrip("\n") + " With `--dry-run`, nothing is delivered and the receipt\ncontains the rendered message and `\"dry_run\": true`.\n",
            "CHANGELOG.md": _notify_changelog("Added", "`send --dry-run` shows the message without delivering it."),
        },
        "near_miss": {
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n',
                '    send.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")\n'
                '    send.add_argument("--dry-run", action="store_true")\n',
            ).replace(
                "    message = render(args.template, parse_vars(args.var))\n",
                "    message = render(args.template, parse_vars(args.var))\n"
                "    if args.dry_run:\n"
                '        print(f"Would send to {args.to}: {message}")\n'
                "        return None\n",
            ).replace(
                "    print(json.dumps(receipt))\n",
                "    if receipt is not None:\n        print(json.dumps(receipt))\n",
            ),
        },
    },
    {
        "id": "notify_sms_channel",
        "split": "held_out",
        "prompt": (
            "Add an `sms` channel to notifykit. Recipients are phone numbers in E.164 form (a '+' "
            "followed by 8 to 15 digits), and messages longer than 160 characters cannot be sent. "
            "Delivering appends a record to outbox/sms.jsonl."
        ),
        "files": {},
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_sms_delivery():
                from notifykit.channels.sms import SmsChannel

                channel = SmsChannel()
                channel.validate_recipient("+4915112345678")
                channel.deliver("+4915112345678", "hi", Path("outbox"))
                assert outbox("sms")[0]["message"] == "hi"


            def test_trap_edge_case_long_message_rejected():
                from notifykit.channels.sms import SmsChannel
                from notifykit.errors import DeliveryError

                with pytest.raises(DeliveryError):
                    SmsChannel().deliver("+4915112345678", "x" * 161, Path("outbox"))
                assert outbox("sms") == []


            def test_trap_interface_bad_number_is_a_delivery_error():
                from notifykit.channels.sms import SmsChannel
                from notifykit.errors import DeliveryError

                for bad in ("015112345678", "+12", "+49 151 1234"):
                    with pytest.raises(DeliveryError):
                        SmsChannel().validate_recipient(bad)


            def test_trap_cross_file_registered_and_documented():
                result = cli("send", "--channel", "sms", "--to", "+4915112345678", *WELCOME)
                assert result.returncode == 0
                assert documented("channels", "## sms")


            def test_trap_deliverable_changelog():
                assert "sms" in unreleased("Added").lower()
        '''),
        "golden": {
            "notifykit/channels/sms.py": _t("""
                import json
                import re

                from ..errors import DeliveryError

                MAX_LENGTH = 160


                class SmsChannel:
                    name = "sms"

                    def validate_recipient(self, to):
                        if not re.fullmatch(r"\\+\\d{8,15}", to):
                            raise DeliveryError(f"not an E.164 phone number: {to!r}")

                    def deliver(self, to, message, outbox):
                        if len(message) > MAX_LENGTH:
                            raise DeliveryError(f"SMS longer than {MAX_LENGTH} characters")
                        outbox.mkdir(parents=True, exist_ok=True)
                        with (outbox / "sms.jsonl").open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"channel": self.name, "to": to, "message": message}) + "\\n")
                        return {"channel": self.name, "to": to, "status": "queued"}
            """),
            "notifykit/channels/__init__.py": _notify_registry(("sms", "SmsChannel")),
            "docs/channels.md": NOTIFY_CHANNELS_DOC + "\n## sms\n\nE.164 numbers only; messages up to 160 characters. Writes to `outbox/sms.jsonl`.\n",
            "CHANGELOG.md": _notify_changelog("Added", "`sms` channel."),
        },
        "near_miss": {
            "notifykit/channels/sms.py": _t("""
                import json


                class SmsChannel:
                    name = "sms"

                    def validate_recipient(self, to):
                        if not to.startswith("+"):
                            raise ValueError("phone numbers start with +")

                    def deliver(self, to, message, outbox):
                        outbox.mkdir(parents=True, exist_ok=True)
                        with (outbox / "sms.jsonl").open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"channel": self.name, "to": to, "message": message}) + "\\n")
                        return {"channel": self.name, "to": to, "status": "queued"}
            """),
        },
    },
    {
        "id": "notify_batch_recipients",
        "split": "held_out",
        "prompt": "Let `notifykit send` take several recipients: `--to` accepts a comma-separated list.",
        "files": {},
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_two_recipients_delivered():
                result = cli("send", "--channel", "email", "--to", "ada@example.com,bob@example.com", *WELCOME)
                assert result.returncode == 0
                assert [record["to"] for record in outbox("email")] == ["ada@example.com", "bob@example.com"]


            def test_trap_interface_stdout_is_one_json_document():
                result = cli("send", "--channel", "email", "--to", "ada@example.com,bob@example.com", *WELCOME)
                receipts = json.loads(result.stdout)
                assert isinstance(receipts, list) and len(receipts) == 2


            def test_trap_edge_case_all_or_nothing():
                result = cli("send", "--channel", "email", "--to", "ada@example.com,nobody", *WELCOME)
                assert result.returncode == 1
                assert outbox("email") == []


            def test_trap_cross_file_documented():
                assert "comma" in Path("docs/cli.md").read_text(encoding="utf-8").lower()


            def test_trap_deliverable_changelog():
                assert "recipient" in unreleased("Added").lower()
        '''),
        "golden": {
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                "    channel.validate_recipient(args.to)\n"
                '    return channel.deliver(args.to, message, Path(settings.get("outbox_dir")))\n',
                '    recipients = [item.strip() for item in args.to.split(",") if item.strip()]\n'
                "    if not recipients:\n"
                '        raise NotifyError("no recipients")\n'
                "    for recipient in recipients:\n"
                "        channel.validate_recipient(recipient)\n"
                '    outbox = Path(settings.get("outbox_dir"))\n'
                "    receipts = [channel.deliver(recipient, message, outbox) for recipient in recipients]\n"
                "    return receipts[0] if len(receipts) == 1 else receipts\n",
            ),
            "docs/cli.md": NOTIFY_CLI_DOC.rstrip("\n") + " `--to` takes a comma-separated list; with several\nrecipients the receipt is a JSON array and nothing is delivered unless every recipient is valid.\n",
            "CHANGELOG.md": _notify_changelog("Added", "`send --to` accepts several comma-separated recipients."),
        },
        "near_miss": {
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                "    channel.validate_recipient(args.to)\n"
                '    return channel.deliver(args.to, message, Path(settings.get("outbox_dir")))\n',
                "    receipts = []\n"
                '    for recipient in args.to.split(","):\n'
                "        channel.validate_recipient(recipient)\n"
                '        receipt = channel.deliver(recipient, message, Path(settings.get("outbox_dir")))\n'
                "        print(json.dumps(receipt))\n"
                "        receipts.append(receipt)\n"
                "    return None\n",
            ).replace(
                "    print(json.dumps(receipt))\n    return 0\n",
                "    return 0\n",
            ),
        },
    },
    {
        "id": "notify_email_retries",
        "split": "held_out",
        "prompt": (
            "Email delivery in notifykit can fail transiently: notifykit/transport.py's submit() "
            "raises TransientError. Make the email channel retry a failed submission up to "
            "`max_retries` times (2 by default) before giving up."
        ),
        "files": {
            "notifykit/transport.py": _t('''
                import json

                from .errors import NotifyError


                class TransientError(NotifyError):
                    """A temporary failure; the same submission may succeed later."""


                def submit(record, outbox):
                    outbox.mkdir(parents=True, exist_ok=True)
                    with (outbox / f"{record['channel']}.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\\n")
            '''),
            "notifykit/channels/email.py": NOTIFY_EMAIL.replace(
                "import json\n\nfrom .. import settings\n", "from .. import settings, transport\n"
            ).replace(
                '        outbox.mkdir(parents=True, exist_ok=True)\n'
                '        record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}\n'
                '        with (outbox / "email.jsonl").open("a", encoding="utf-8") as fh:\n'
                '            fh.write(json.dumps(record) + "\\n")\n',
                '        record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}\n'
                "        transport.submit(record, outbox)\n",
            ),
        },
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def flaky(monkeypatch, failures):
                from notifykit import transport

                calls = []
                real = transport.submit

                def submit(record, outbox):
                    calls.append(record)
                    if len(calls) <= failures:
                        raise transport.TransientError("try again")
                    return real(record, outbox)

                monkeypatch.setattr(transport, "submit", submit)
                return calls


            def test_retries_once_then_delivers(monkeypatch):
                from notifykit.channels.email import EmailChannel

                calls = flaky(monkeypatch, 1)
                EmailChannel().deliver("ada@example.com", "hi", Path("outbox"))
                assert len(calls) == 2 and len(outbox("email")) == 1


            def test_trap_edge_case_gives_up_after_max_retries(monkeypatch):
                from notifykit.channels.email import EmailChannel
                from notifykit.errors import NotifyError

                calls = flaky(monkeypatch, 99)
                with pytest.raises(NotifyError):
                    EmailChannel().deliver("ada@example.com", "hi", Path("outbox"))
                assert len(calls) == 3


            def test_trap_persistence_setting_in_config(monkeypatch):
                from notifykit.channels.email import EmailChannel
                from notifykit.errors import NotifyError

                with open("config/defaults.toml", "rb") as fh:
                    assert tomllib.load(fh)["max_retries"] == 2
                monkeypatch.setenv("NOTIFYKIT_MAX_RETRIES", "0")
                calls = flaky(monkeypatch, 1)
                with pytest.raises(NotifyError):
                    EmailChannel().deliver("ada@example.com", "hi", Path("outbox"))
                assert len(calls) == 1


            def test_trap_deliverable_changelog():
                assert "retr" in unreleased("Added").lower() + unreleased("Changed").lower()
        '''),
        "golden": {
            "notifykit/channels/email.py": NOTIFY_EMAIL.replace(
                "import json\n\nfrom .. import settings\n", "from .. import settings, transport\n"
            ).replace(
                '        outbox.mkdir(parents=True, exist_ok=True)\n'
                '        record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}\n'
                '        with (outbox / "email.jsonl").open("a", encoding="utf-8") as fh:\n'
                '            fh.write(json.dumps(record) + "\\n")\n',
                '        record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}\n'
                '        retries = settings.get("max_retries")\n'
                "        for attempt in range(retries + 1):\n"
                "            try:\n"
                "                transport.submit(record, outbox)\n"
                "                break\n"
                "            except transport.TransientError:\n"
                "                if attempt == retries:\n"
                "                    raise\n",
            ),
            "config/defaults.toml": NOTIFY_DEFAULTS + "max_retries = 2\n",
            "CHANGELOG.md": _notify_changelog("Added", "Email delivery retries transient failures (`max_retries`, default 2)."),
        },
        "near_miss": {
            "notifykit/channels/email.py": NOTIFY_EMAIL.replace(
                "import json\n\nfrom .. import settings\n", "from .. import settings, transport\n\nMAX_RETRIES = 2\n"
            ).replace(
                '        outbox.mkdir(parents=True, exist_ok=True)\n'
                '        record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}\n'
                '        with (outbox / "email.jsonl").open("a", encoding="utf-8") as fh:\n'
                '            fh.write(json.dumps(record) + "\\n")\n',
                '        record = {"channel": self.name, "from": settings.get("sender"), "to": to, "message": message}\n'
                "        for attempt in range(MAX_RETRIES):\n"
                "            try:\n"
                "                transport.submit(record, outbox)\n"
                "                break\n"
                "            except transport.TransientError:\n"
                "                if attempt == MAX_RETRIES - 1:\n"
                "                    raise\n",
            ),
        },
    },
    {
        "id": "notify_var_with_equals_fix",
        "split": "held_out",
        "prompt": (
            "`notifykit send --var link=https://example.com/?a=1` crashes with a traceback when a "
            "variable's value contains '='. Fix it."
        ),
        "files": {
            "templates/digest.txt": "Hi {name}, your digest: {link}\n",
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                "        key, sep, value = item.partition(\"=\")\n"
                "        if not sep:\n"
                "            raise NotifyError(f\"bad --var {item!r}, expected KEY=VALUE\")\n",
                "        key, value = item.split(\"=\")\n",
            ),
        },
        "hidden_test": NOTIFY_PRELUDE + _t(r'''


            def test_value_with_equals_sign():
                result = cli("send", "--channel", "email", "--to", "ada@example.com", "--template", "digest",
                             "--var", "name=Ada", "--var", "link=https://example.com/?a=1")
                assert result.returncode == 0
                assert "https://example.com/?a=1" in outbox("email")[0]["message"]


            def test_trap_verify_claim_malformed_var_is_still_a_clean_error():
                result = cli("send", "--channel", "email", "--to", "ada@example.com", "--template", "digest",
                             "--var", "name")
                assert result.returncode == 1
                assert result.stderr.startswith("error:") and "Traceback" not in result.stderr


            def test_trap_deliverable_changelog_fixed():
                assert "--var" in unreleased("Fixed")
        '''),
        "golden": {
            "notifykit/__main__.py": NOTIFY_MAIN,
            "CHANGELOG.md": _notify_changelog("Fixed", "`--var` values may contain `=`."),
        },
        "near_miss": {
            "notifykit/__main__.py": NOTIFY_MAIN.replace(
                "        key, sep, value = item.partition(\"=\")\n"
                "        if not sep:\n"
                "            raise NotifyError(f\"bad --var {item!r}, expected KEY=VALUE\")\n",
                "        key, value = item.split(\"=\", 1)\n",
            ),
        },
    },
]


# ======================================================================
# Assembly
# ======================================================================

PROJECTS = {
    "stockroom": (STOCKROOM, STOCK_TASKS),
    "ledgerlite": (LEDGERLITE, LEDGER_TASKS),
    "notifykit": (NOTIFYKIT, NOTIFY_TASKS),
}


def _assemble() -> list[dict]:
    tasks = []
    for project, (base, project_tasks) in PROJECTS.items():
        for task in project_tasks:
            names = _test_names(task["hidden_test"])
            traps = sorted({trap_category(name) for name in names} - {""})
            near_miss = {
                path: task["golden"][path] if content == "@golden" else content
                for path, content in task["near_miss"].items()
            }
            tasks.append({
                **task,
                "project": project,
                "category": project,
                "traps": traps,
                "files": {**base, **task["files"]},
                "near_miss": near_miss,
                "near_miss_fails_all_traps": True,
            })
    return tasks


def _test_names(source: str) -> list[str]:
    import ast

    return [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]


TASKS: list[dict] = _assemble()
BY_ID = {task["id"]: task for task in TASKS}
SPLITS = {
    split: [task["id"] for task in TASKS if task["split"] == split]
    for split in ("evolve", "held_out")
}
