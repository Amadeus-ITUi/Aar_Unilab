from pathlib import Path

from tools.audit_repository import audit


def test_runtime_tree_has_no_predecessor_path_dependencies():
    assert audit(Path(__file__).resolve().parents[2]) == []
