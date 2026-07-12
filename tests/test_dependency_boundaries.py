"""Enforce invariant #2 from the design notes: the base SDK (everything
under src/proxyml except proxyml/local/**) never imports proxyml_core.modeling
or sklearn/scipy directly, so a REST-only `pip install proxyml` never needs
scikit-learn.
"""

import ast
import subprocess
import sys
from pathlib import Path

import proxyml

_DISALLOWED = {"sklearn", "scipy"}
_DISALLOWED_SUBMODULES = {"proxyml_core.modeling"}

_PACKAGE_DIR = Path(proxyml.__file__).parent


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
    return modules


def _base_sdk_files() -> list[Path]:
    return [
        p for p in _PACKAGE_DIR.rglob("*.py")
        if "local" not in p.relative_to(_PACKAGE_DIR).parts
    ]


def test_base_sdk_never_imports_sklearn_or_modeling():
    for path in _base_sdk_files():
        imported = _top_level_imports(path)
        bad_direct = imported & _DISALLOWED
        bad_submodules = {m for m in imported if any(m.startswith(sub) for sub in _DISALLOWED_SUBMODULES)}
        assert not bad_direct, f"{path} imports disallowed module(s): {bad_direct}"
        assert not bad_submodules, f"{path} imports disallowed submodule(s): {bad_submodules}"


def test_import_proxyml_without_sklearn_or_scipy():
    """`import proxyml` alone (not proxyml.local) must never require sklearn/scipy."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for m in ('sklearn', 'scipy'):\n"
            "    sys.modules[m] = None\n"
            "import proxyml\n"
            "print('ok')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_local_import_guard_gives_clear_error_without_modeling_extra():
    """proxyml.local must fail fast with a clear message if sklearn isn't installed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for m in ('sklearn', 'scipy'):\n"
            "    sys.modules[m] = None\n"
            "try:\n"
            "    import proxyml.local\n"
            "except ImportError as e:\n"
            "    assert 'proxyml[local]' in str(e), str(e)\n"
            "    print('ok')\n"
            "else:\n"
            "    raise SystemExit('expected ImportError')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
