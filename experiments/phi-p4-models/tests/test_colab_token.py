"""The Colab notebooks' git cells keep the GitHub token out of every command line, address and error message.

The cells are read from the generator's source and run against a stand-in for subprocess that records each git
command and fails the ones a test picks, so no test reaches the network or runs git.
"""
import ast
import base64
import os
import pathlib
import shutil
import subprocess
import types

import pytest

GENERATOR = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_colab_notebook.py"
TOKEN = "ghp_0123456789abcdefSECRET"
HEADER = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()


def cell_source(marker):
    """The source of the one code cell the generator builds from a string literal holding marker."""
    found = [node.args[0].value for node in ast.walk(ast.parse(GENERATOR.read_text(encoding="utf-8")))
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "code" and node.args
             and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
             and marker in node.args[0].value]
    assert len(found) == 1, marker
    return found[0]


class Git:
    """Records each command; a command for which fails(args) is true exits 128 and echoes the address it was given,
    as git does when it cannot reach a repository."""

    CalledProcessError = subprocess.CalledProcessError

    def __init__(self, fails=lambda args: False):
        self.fails = fails
        self.calls = []

    def run(self, args, env=None, check=False, capture_output=False, text=False, **_):
        args = [str(a) for a in args]
        self.calls.append((args, dict(env or {})))
        if not self.fails(args):
            return subprocess.CompletedProcess(args, 0, "", "")
        address = next((a for a in args if "github.com" in a), "")
        err = f"fatal: unable to access '{address}': The requested URL returned error: 403\n"
        if check:
            raise subprocess.CalledProcessError(128, args, "", err)
        return subprocess.CompletedProcess(args, 128, "", err)


def namespace(git, tmp_path):
    """The repositories cell's functions, with the names the notebook's first cell gives them."""
    ns = {"IN_COLAB": False, "os": os, "subprocess": git, "pathlib": pathlib, "shutil": shutil}
    tree = ast.parse(cell_source("def checkout(repo, ref, dest, pin"))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    exec(compile(tree, "repositories cell", "exec"), ns)
    return ns


def secret_in(text):
    return TOKEN in text or HEADER in text


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)


def test_a_failed_clone_names_no_token(tmp_path):
    git = Git(fails=lambda args: "clone" in args)
    ns = namespace(git, tmp_path)
    with pytest.raises(RuntimeError) as e:
        ns["checkout"]("ezexe/phi", "main", tmp_path / "phi", None)
    assert not secret_in(str(e.value)), str(e.value)


def test_no_git_command_line_holds_the_token(tmp_path):
    git = Git()
    namespace(git, tmp_path)["checkout"]("ezexe/phi", "main", tmp_path / "phi", None)
    assert git.calls, "checkout ran no git command"
    for args, _env in git.calls:
        assert not any(secret_in(a) for a in args), args


def test_the_clone_carries_the_token_as_a_header(tmp_path):
    git = Git()
    namespace(git, tmp_path)["checkout"]("ezexe/phi", "main", tmp_path / "phi", None)
    clone_env = next(env for args, env in git.calls if "clone" in args)
    assert clone_env.get("GIT_CONFIG_COUNT") == "1"
    assert clone_env.get("GIT_CONFIG_KEY_0") == "http.https://github.com/.extraheader"
    assert clone_env.get("GIT_CONFIG_VALUE_0") == f"AUTHORIZATION: basic {HEADER}"


def push_cell(git, tmp_path):
    ns = namespace(git, tmp_path)
    ns.update({"PUSH_RESULTS": True, "PUSH_BRANCH": "colab/test", "RUN": "test-1", "WORK": tmp_path,
               "ROOT": tmp_path / "run", "done": lambda step: False, "mark": lambda step, **info: None})
    (tmp_path / "run" / "results").mkdir(parents=True)
    exec(compile(cell_source('if PUSH_RESULTS and not done("push"):'), "push cell", "exec"), ns)


@pytest.mark.parametrize("step", ["clone", "push"])
def test_a_failed_push_cell_names_no_token(tmp_path, step):
    git = Git(fails=lambda args: step in args)
    with pytest.raises(Exception) as e:
        push_cell(git, tmp_path)
    assert not secret_in(str(e.value)), str(e.value)
    for args, _env in git.calls:
        assert not any(secret_in(a) for a in args), args
