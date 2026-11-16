"""The store the notebooks clone gives first_added the same history as its source, once its remote can no longer be
reached.

The store is a partial clone, so git fetches a missing blob from the remote when it needs one, and the rename
detection of first_added's log needs the blobs of deleted files. The remote holds no token, so a fetch after the
repositories cell fails. This runs the cell's checkout with real git on a local repository, offline.
"""
import ast
import importlib.util
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "make_colab_notebook.py"
HELPERS = ROOT / "scripts" / "nested_vs_one.py"


def cell_source(marker):
    """The source of the one code cell the generator builds from a string literal holding marker."""
    found = [node.args[0].value for node in ast.walk(ast.parse(GENERATOR.read_text(encoding="utf-8")))
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "code" and node.args
             and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
             and marker in node.args[0].value]
    assert len(found) == 1, marker
    return found[0]


def first_added():
    spec = importlib.util.spec_from_file_location("nested_vs_one_under_test", HELPERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.first_added


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd, check=True,
                   capture_output=True)


def store_with_a_rename(path):
    """a.txt and c.txt added, then a.txt renamed to b.txt with a line changed."""
    path.mkdir()
    git("init", "-q", "-b", "main", cwd=path)
    body = "".join(f"line {i} of the store's oldest entry\n" for i in range(40))
    (path / "a.txt").write_text(body, encoding="utf-8")
    (path / "c.txt").write_text("another entry\n", encoding="utf-8")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "first", cwd=path)
    (path / "a.txt").unlink()
    (path / "b.txt").write_text(body + "one line more\n", encoding="utf-8")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "rename with a change", cwd=path)
    git("config", "uploadpack.allowFilter", "true", cwd=path)


def checkout_functions(source):
    ns = {"IN_COLAB": False, "os": os, "subprocess": subprocess, "pathlib": pathlib, "shutil": shutil}
    tree = ast.parse(cell_source("def checkout(repo, ref, dest, pin"))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    exec(compile(tree, "repositories cell", "exec"), ns)
    ns["url"] = lambda repo: source.as_uri()
    return ns


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def test_the_cell_clones_the_store_with_its_history():
    cell = cell_source("def checkout(repo, ref, dest, pin")
    assert 'checkout(STORE_REPO, STORE_BRANCH, STORE, pins.get("store"), history=True)' in cell


def test_first_added_on_the_cloned_store_equals_its_source(tmp_path):
    source, clone = tmp_path / "source", tmp_path / "clone"
    store_with_a_rename(source)
    checkout_functions(source)["checkout"]("ezexe/store", "main", clone, None, history=True)
    git("remote", "set-url", "origin", (tmp_path / "gone").as_uri(), cwd=clone)
    added = first_added()
    assert set(added(str(source))) == {"a.txt", "c.txt"}
    assert added(str(clone)) == added(str(source))
