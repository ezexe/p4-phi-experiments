#!/usr/bin/env python3
"""Offline test of by_command.py's kinds.

    python experiments/context_growth/test_by_command.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import by_command as bc  # noqa: E402


class ByCommand(unittest.TestCase):
    def test_kind_is_the_tool_and_a_shell_commands_program(self):
        self.assertEqual(bc.kind("Read", {"file_path": "a.py"}), "Read")
        self.assertEqual(bc.kind("Bash", {"command": "cd /x && git -C y status --short"}), "Bash: git status")
        self.assertEqual(bc.kind("Bash", {"command": "FOO=1 /usr/bin/python3.exe -I run.py"}), "Bash: python3")
        self.assertEqual(bc.kind("PowerShell", {"command": "Get-ChildItem -Recurse"}), "PowerShell: get-childitem")
        self.assertEqual(bc.kind("Bash", {"command": ""}), "Bash: ?")


if __name__ == "__main__":
    unittest.main()
