#!/usr/bin/env python3
"""
EigenDialectos Project Validator
=================================
Reads MANIFEST.yaml and validates every component:
  - File existence
  - Python importability
  - Test execution (optional)
  - Dependency ordering
  - Status consistency

Usage:
    python scripts/validate_project.py                  # full validation
    python scripts/validate_project.py --quick          # file existence only
    python scripts/validate_project.py --phase P0_FOUNDATION
    python scripts/validate_project.py --run-tests      # also execute pytest
    python scripts/validate_project.py --fix-status      # update MANIFEST statuses
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "MANIFEST.yaml"
PROGRESS_PATH = PROJECT_ROOT / "PROGRESS.md"
SRC_ROOT = PROJECT_ROOT / "src"


class Status(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


# ---------------------------------------------------------------------------
# ANSI colours (respects NO_COLOR env)
# ---------------------------------------------------------------------------
_NO_COLOR = os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def red(t: str) -> str:
    return _c("31", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


STATUS_COLORS = {
    Status.DONE: green,
    Status.PENDING: yellow,
    Status.IN_PROGRESS: cyan,
    Status.BLOCKED: red,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    component_id: str
    severity: Severity
    message: str


@dataclass
class ComponentResult:
    component_id: str
    name: str
    phase: str
    declared_status: Status
    computed_status: Status
    files_exist: dict[str, bool] = field(default_factory=dict)
    imports_ok: dict[str, bool | None] = field(default_factory=dict)
    tests_passed: dict[str, bool | None] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def all_files_exist(self) -> bool:
        return all(self.files_exist.values()) if self.files_exist else False

    @property
    def all_imports_ok(self) -> bool:
        checked = {k: v for k, v in self.imports_ok.items() if v is not None}
        return all(checked.values()) if checked else True

    @property
    def all_tests_passed(self) -> bool | None:
        checked = {k: v for k, v in self.tests_passed.items() if v is not None}
        if not checked:
            return None
        return all(checked.values())


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def load_manifest() -> dict[str, Any]:
    """Load and return parsed MANIFEST.yaml."""
    if not MANIFEST_PATH.exists():
        print(red(f"FATAL: {MANIFEST_PATH} not found."), file=sys.stderr)
        sys.exit(2)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_manifest(data: dict[str, Any]) -> None:
    """Write updated manifest back to disk preserving order."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


def iter_components(
    manifest: dict[str, Any],
    phase_filter: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Yield (phase_id, component_dict) pairs."""
    results = []
    phases = manifest.get("phases", {})
    for phase_id, phase_data in phases.items():
        if phase_filter and phase_id != phase_filter:
            continue
        for comp in phase_data.get("components", []):
            results.append((phase_id, comp))
    return results


def build_component_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map component id -> component dict for dependency lookup."""
    index: dict[str, dict[str, Any]] = {}
    for _, comp in iter_components(manifest):
        index[comp["id"]] = comp
    return index


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_file_existence(files: list[str]) -> dict[str, bool]:
    """Check whether each relative path exists under PROJECT_ROOT."""
    return {f: (PROJECT_ROOT / f).exists() for f in files}


def check_imports(files: list[str]) -> dict[str, bool | None]:
    """Attempt to import each .py file. Returns None for non-Python files."""
    results: dict[str, bool | None] = {}
    for f in files:
        if not f.endswith(".py"):
            results[f] = None
            continue
        if not (PROJECT_ROOT / f).exists():
            results[f] = None
            continue
        # Build module path from src-relative path
        rel = f
        if rel.startswith("src/"):
            rel = rel[4:]
        module = rel.replace("/", ".").removesuffix(".py")
        # Handle __init__.py -> package import
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        try:
            # Ensure src is on sys.path
            src_str = str(SRC_ROOT)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            importlib.import_module(module)
            results[f] = True
        except Exception as exc:
            results[f] = False
            # Store the error message for later reporting
            results[f"_err_{f}"] = str(exc)  # type: ignore[assignment]
    return results


def check_tests(test_files: list[str]) -> dict[str, bool | None]:
    """Run pytest on each test file. Returns None if file missing."""
    results: dict[str, bool | None] = {}
    for tf in test_files:
        full = PROJECT_ROOT / tf
        if not full.exists():
            results[tf] = None
            continue
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(full), "-x", "-q", "--tb=short", "--no-header"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(PROJECT_ROOT),
            )
            results[tf] = proc.returncode == 0
        except subprocess.TimeoutExpired:
            results[tf] = False
        except Exception:
            results[tf] = False
    return results


def check_dependencies(
    comp: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> list[Issue]:
    """Verify dependency ordering: DONE requires all deps DONE."""
    issues: list[Issue] = []
    comp_status = Status(comp.get("status", "PENDING"))
    deps = comp.get("depends_on", [])

    if comp_status == Status.DONE:
        for dep_id in deps:
            dep = index.get(dep_id)
            if dep is None:
                issues.append(Issue(
                    component_id=comp["id"],
                    severity=Severity.ERROR,
                    message=f"Depends on unknown component '{dep_id}'",
                ))
                continue
            dep_status = Status(dep.get("status", "PENDING"))
            if dep_status != Status.DONE:
                issues.append(Issue(
                    component_id=comp["id"],
                    severity=Severity.ERROR,
                    message=f"Marked DONE but dependency '{dep_id}' is {dep_status.value}",
                ))
    return issues


# ---------------------------------------------------------------------------
# Compute effective status
# ---------------------------------------------------------------------------
def compute_status(
    comp: dict[str, Any],
    files_exist: dict[str, bool],
    imports_ok: dict[str, bool | None],
    tests_passed: dict[str, bool | None] | None,
    dep_issues: list[Issue],
) -> Status:
    """Determine what the status *should* be based on evidence."""
    # If any dependency is violated, cannot be DONE
    has_dep_errors = any(i.severity == Severity.ERROR for i in dep_issues)

    all_exist = all(files_exist.values()) if files_exist else False
    checked_imports = {k: v for k, v in imports_ok.items() if v is not None}
    all_importable = all(checked_imports.values()) if checked_imports else True

    if not all_exist:
        # Some or all files missing
        existing_count = sum(1 for v in files_exist.values() if v)
        if existing_count == 0:
            return Status.PENDING
        return Status.IN_PROGRESS

    if not all_importable:
        return Status.IN_PROGRESS

    if has_dep_errors:
        return Status.BLOCKED

    # If tests were checked and any failed, IN_PROGRESS
    if tests_passed is not None:
        checked_tests = {k: v for k, v in tests_passed.items() if v is not None}
        if checked_tests and not all(checked_tests.values()):
            return Status.IN_PROGRESS

    return Status.DONE


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def validate(
    manifest: dict[str, Any],
    *,
    quick: bool = False,
    run_tests: bool = False,
    phase_filter: str | None = None,
) -> list[ComponentResult]:
    """Run validation across all (or filtered) components."""
    components = iter_components(manifest, phase_filter)
    index = build_component_index(manifest)
    results: list[ComponentResult] = []

    total = len(components)
    for i, (phase_id, comp) in enumerate(components, 1):
        cid = comp["id"]
        cname = comp.get("name", cid)
        declared = Status(comp.get("status", "PENDING"))
        files = comp.get("files", [])
        test_files = comp.get("tests", [])

        # Progress indicator
        progress_pct = int(i / total * 100) if total else 0
        print(
            f"\r  [{progress_pct:3d}%] Validating {dim(cid):<45s}",
            end="",
            flush=True,
        )

        # 1. File existence (always)
        fe = check_file_existence(files)

        # 2. Imports (unless --quick)
        io: dict[str, bool | None] = {}
        if not quick:
            io = check_imports(files)

        # 3. Tests (only with --run-tests)
        tp: dict[str, bool | None] | None = None
        if run_tests and test_files:
            tp = check_tests(test_files)

        # 4. Dependency check
        dep_issues = check_dependencies(comp, index)

        # 5. Compute effective status
        computed = compute_status(comp, fe, io, tp, dep_issues)

        # Build issues list
        issues = list(dep_issues)
        for fname, exists in fe.items():
            if not exists:
                issues.append(Issue(cid, Severity.ERROR, f"Missing file: {fname}"))
        for fname, ok in io.items():
            if fname.startswith("_err_"):
                continue
            if ok is False:
                err_key = f"_err_{fname}"
                err_msg = io.get(err_key, "unknown error")
                issues.append(Issue(cid, Severity.ERROR, f"Import failed: {fname} ({err_msg})"))
        if tp:
            for tname, passed in tp.items():
                if passed is False:
                    issues.append(Issue(cid, Severity.WARNING, f"Test failed: {tname}"))

        if declared == Status.DONE and computed != Status.DONE:
            issues.append(Issue(
                cid, Severity.ERROR,
                f"Declared DONE but computed {computed.value}",
            ))

        results.append(ComponentResult(
            component_id=cid,
            name=cname,
            phase=phase_id,
            declared_status=declared,
            computed_status=computed,
            files_exist=fe,
            imports_ok={k: v for k, v in io.items() if not k.startswith("_err_")},
            tests_passed=tp or {},
            issues=issues,
        ))

    print("\r" + " " * 70 + "\r", end="", flush=True)  # clear progress line
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(results: list[ComponentResult]) -> None:
    """Print a coloured summary table to stdout."""
    print()
    print(bold("=" * 90))
    print(bold("  EigenDialectos -- Project Validation Report"))
    print(bold("=" * 90))
    print()

    current_phase = ""
    for r in results:
        if r.phase != current_phase:
            current_phase = r.phase
            print(bold(f"\n  {current_phase}"))
            print(f"  {'Component':<40s} {'Files':>6s} {'Import':>7s} {'Tests':>6s}  {'Status':<15s}")
            print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*6}  {'-'*15}")

        # File count
        file_total = len(r.files_exist)
        file_ok = sum(1 for v in r.files_exist.values() if v)
        if file_ok == file_total and file_total > 0:
            files_str = green(f"{file_ok}/{file_total}")
        elif file_ok == 0:
            files_str = red(f"{file_ok}/{file_total}")
        else:
            files_str = yellow(f"{file_ok}/{file_total}")

        # Import count
        checked_imports = {k: v for k, v in r.imports_ok.items() if v is not None}
        if not checked_imports:
            import_str = dim("  --  ")
        else:
            imp_ok = sum(1 for v in checked_imports.values() if v)
            imp_total = len(checked_imports)
            if imp_ok == imp_total:
                import_str = green(f" {imp_ok}/{imp_total}  ")
            else:
                import_str = red(f" {imp_ok}/{imp_total}  ")

        # Tests
        checked_tests = {k: v for k, v in r.tests_passed.items() if v is not None}
        if not checked_tests:
            test_str = dim("  -- ")
        else:
            t_ok = sum(1 for v in checked_tests.values() if v)
            t_total = len(checked_tests)
            if t_ok == t_total:
                test_str = green(f" {t_ok}/{t_total}")
            else:
                test_str = red(f" {t_ok}/{t_total}")

        # Status
        color_fn = STATUS_COLORS.get(r.computed_status, str)
        status_str = color_fn(r.computed_status.value)

        print(f"  {r.component_id:<40s} {files_str:>16s} {import_str:>17s} {test_str:>16s}  {status_str}")

    # Summary
    print()
    print(bold("-" * 90))
    total = len(results)
    done = sum(1 for r in results if r.computed_status == Status.DONE)
    in_progress = sum(1 for r in results if r.computed_status == Status.IN_PROGRESS)
    pending = sum(1 for r in results if r.computed_status == Status.PENDING)
    blocked = sum(1 for r in results if r.computed_status == Status.BLOCKED)
    pct = (done / total * 100) if total else 0

    print(f"  Total: {bold(str(total))}  |  "
          f"{green('DONE')}: {done}  |  "
          f"{cyan('IN_PROGRESS')}: {in_progress}  |  "
          f"{yellow('PENDING')}: {pending}  |  "
          f"{red('BLOCKED')}: {blocked}  |  "
          f"Progress: {bold(f'{pct:.1f}%')}")
    print(bold("=" * 90))

    # Print issues
    all_issues = []
    for r in results:
        all_issues.extend(r.issues)

    errors = [i for i in all_issues if i.severity == Severity.ERROR]
    warnings = [i for i in all_issues if i.severity == Severity.WARNING]

    if errors:
        print()
        print(red(bold(f"  ERRORS ({len(errors)}):")))
        for issue in errors[:30]:  # cap output
            print(f"    {red('E')} [{issue.component_id}] {issue.message}")
        if len(errors) > 30:
            print(f"    ... and {len(errors) - 30} more errors")

    if warnings:
        print()
        print(yellow(bold(f"  WARNINGS ({len(warnings)}):")))
        for issue in warnings[:20]:
            print(f"    {yellow('W')} [{issue.component_id}] {issue.message}")
        if len(warnings) > 20:
            print(f"    ... and {len(warnings) - 20} more warnings")

    print()


def print_phase_summary(results: list[ComponentResult]) -> None:
    """Print per-phase rollup."""
    phases: dict[str, list[ComponentResult]] = {}
    for r in results:
        phases.setdefault(r.phase, []).append(r)

    print(bold("\n  Phase Summary:"))
    print(f"  {'Phase':<25s} {'Done':>5s} {'Total':>6s} {'Progress':>10s}")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*10}")
    for phase_id, comps in phases.items():
        done = sum(1 for c in comps if c.computed_status == Status.DONE)
        total = len(comps)
        pct = (done / total * 100) if total else 0
        bar_len = 20
        filled = int(bar_len * done / total) if total else 0
        bar = green("█" * filled) + dim("░" * (bar_len - filled))
        print(f"  {phase_id:<25s} {done:>5d} {total:>6d}  {bar} {pct:5.1f}%")
    print()


# ---------------------------------------------------------------------------
# Manifest status updater
# ---------------------------------------------------------------------------
def update_manifest_statuses(
    manifest: dict[str, Any],
    results: list[ComponentResult],
) -> int:
    """Update MANIFEST.yaml component statuses. Returns count of changes."""
    result_map = {r.component_id: r for r in results}
    changes = 0
    for phase_data in manifest.get("phases", {}).values():
        for comp in phase_data.get("components", []):
            cid = comp["id"]
            if cid in result_map:
                new_status = result_map[cid].computed_status.value
                if comp.get("status") != new_status:
                    comp["status"] = new_status
                    changes += 1
    return changes


# ---------------------------------------------------------------------------
# PROGRESS.md writer
# ---------------------------------------------------------------------------
def write_progress(results: list[ComponentResult]) -> None:
    """Write a human-readable PROGRESS.md file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    done = sum(1 for r in results if r.computed_status == Status.DONE)
    pct = (done / total * 100) if total else 0

    lines = [
        "# EigenDialectos -- Project Progress",
        "",
        f"*Auto-generated by `scripts/validate_project.py` on {now}*",
        "",
        f"**Overall: {done}/{total} components DONE ({pct:.1f}%)**",
        "",
    ]

    # Phase table
    lines.append("## Phase Summary")
    lines.append("")
    lines.append("| Phase | Done | Total | Progress |")
    lines.append("|-------|-----:|------:|---------:|")

    phases: dict[str, list[ComponentResult]] = {}
    for r in results:
        phases.setdefault(r.phase, []).append(r)

    for phase_id, comps in phases.items():
        d = sum(1 for c in comps if c.computed_status == Status.DONE)
        t = len(comps)
        p = (d / t * 100) if t else 0
        lines.append(f"| {phase_id} | {d} | {t} | {p:.0f}% |")

    lines.append("")

    # Detail per phase
    lines.append("## Component Details")
    lines.append("")
    for phase_id, comps in phases.items():
        lines.append(f"### {phase_id}")
        lines.append("")
        lines.append("| Component | Status | Files |")
        lines.append("|-----------|--------|------:|")
        for c in comps:
            file_ok = sum(1 for v in c.files_exist.values() if v)
            file_total = len(c.files_exist)
            status_emoji = {
                Status.DONE: "DONE",
                Status.IN_PROGRESS: "IN_PROGRESS",
                Status.PENDING: "PENDING",
                Status.BLOCKED: "BLOCKED",
            }.get(c.computed_status, "?")
            lines.append(
                f"| {c.component_id} | {status_emoji} | {file_ok}/{file_total} |"
            )
        lines.append("")

    # Issues
    all_issues = []
    for r in results:
        all_issues.extend(r.issues)
    if all_issues:
        lines.append("## Open Issues")
        lines.append("")
        for issue in all_issues[:50]:
            prefix = "ERROR" if issue.severity == Severity.ERROR else "WARN"
            lines.append(f"- **{prefix}** [{issue.component_id}]: {issue.message}")
        if len(all_issues) > 50:
            lines.append(f"- ... and {len(all_issues) - 50} more")
        lines.append("")

    content = "\n".join(lines) + "\n"
    PROGRESS_PATH.write_text(content, encoding="utf-8")
    print(f"  Wrote {PROGRESS_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate EigenDialectos project against MANIFEST.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/validate_project.py                    # full validation
              python scripts/validate_project.py --quick            # file checks only
              python scripts/validate_project.py --phase P0_FOUNDATION
              python scripts/validate_project.py --run-tests        # include pytest
              python scripts/validate_project.py --fix-status       # update MANIFEST statuses
        """),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only check file existence (skip imports and tests)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        metavar="PHASE_ID",
        help="Validate only a specific phase (e.g. P0_FOUNDATION)",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute pytest on each component's test files",
    )
    parser.add_argument(
        "--fix-status",
        action="store_true",
        help="Update MANIFEST.yaml statuses to match computed values",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Skip writing PROGRESS.md",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout (suppresses table)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print(bold("\n  EigenDialectos Project Validator"))
    print(f"  Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Mode: {'quick' if args.quick else 'full'}"
          f"{' +tests' if args.run_tests else ''}"
          f"{' +fix' if args.fix_status else ''}")
    if args.phase:
        print(f"  Phase filter: {args.phase}")
    print()

    manifest = load_manifest()

    # Validate phase filter
    if args.phase and args.phase not in manifest.get("phases", {}):
        available = ", ".join(manifest.get("phases", {}).keys())
        print(red(f"  ERROR: Unknown phase '{args.phase}'. Available: {available}"))
        return 2

    results = validate(
        manifest,
        quick=args.quick,
        run_tests=args.run_tests,
        phase_filter=args.phase,
    )

    if args.json:
        import json

        out = []
        for r in results:
            out.append({
                "id": r.component_id,
                "name": r.name,
                "phase": r.phase,
                "declared_status": r.declared_status.value,
                "computed_status": r.computed_status.value,
                "files_exist": r.files_exist,
                "issues": [
                    {"severity": i.severity.value, "message": i.message}
                    for i in r.issues
                ],
            })
        print(json.dumps(out, indent=2))
    else:
        print_summary_table(results)
        print_phase_summary(results)

    # Fix statuses
    if args.fix_status:
        changes = update_manifest_statuses(manifest, results)
        if changes:
            save_manifest(manifest)
            print(f"  Updated {changes} component status(es) in MANIFEST.yaml")
        else:
            print("  No status changes needed in MANIFEST.yaml")

    # Write progress
    if not args.no_progress:
        write_progress(results)

    # Exit code
    all_done = all(r.computed_status == Status.DONE for r in results)
    has_errors = any(
        i.severity == Severity.ERROR
        for r in results
        for i in r.issues
    )

    if all_done:
        print(green(bold("\n  ALL COMPONENTS DONE\n")))
        return 0
    elif has_errors:
        print(red(bold(f"\n  VALIDATION FAILED ({sum(1 for r in results if r.computed_status != Status.DONE)} components incomplete)\n")))
        return 1
    else:
        print(yellow(bold("\n  VALIDATION INCOMPLETE (no errors, but not all DONE)\n")))
        return 1


if __name__ == "__main__":
    sys.exit(main())
