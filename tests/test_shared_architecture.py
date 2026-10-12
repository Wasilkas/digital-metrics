"""Consumer and workflow dependencies cannot flow into extracted library modules."""

import ast
from pathlib import Path

import pytest

SCOPED = [
    "__init__.py",
    "fixed.py",
    "comparison.py",
    "scoring/ap.py",
    "scoring/__init__.py",
    "reporting/dashboard.py",
    "reporting/__init__.py",
]


def assert_library_boundary(source: str) -> None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [node.module or ""]
        else:
            continue
        for name in imports:
            assert name.split(".")[0] not in {"clearml_yolo", "hydra", "clearml"}, name


def test_extracted_modules_remain_library_owned() -> None:
    root = Path(__file__).parents[1] / "src/digital_metrics"
    for path in SCOPED:
        assert_library_boundary((root / path).read_text())


@pytest.mark.parametrize(
    "source",
    [
        "import clearml_yolo.adapters.reporting",
        "from hydra import main",
        "from clearml import Task",
        "if False:\n    import clearml_yolo.core",
    ],
)
def test_prohibited_dependency_negative_case(source: str) -> None:
    with pytest.raises(AssertionError):
        assert_library_boundary(source)


def test_shared_documentation_links_and_examples() -> None:
    import re

    root = Path(__file__).parents[1]
    for relative in (
        "docs/shared-evaluation.md",
        "docs/plans/shared-extraction/plan.md",
        "docs/plans/shared-extraction/evidence-2026-10-12.md",
    ):
        path = root / relative
        text = path.read_text()
        assert text.count("```") % 2 == 0
        for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if not destination.startswith(("http:", "https:", "#")):
                assert (path.parent / destination.split("#")[0]).exists(), destination
        for source in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
            ast.parse(source)
