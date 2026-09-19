import ast
import os
import pytest

BANNED_HTTP_MODULES = {"requests", "urllib.request", "httpx", "aiohttp", "urllib3", "http.client", "socket"}
BANNED_OFFLINE_IMPORTS = {"src.offline", "offline"}

def test_src_online_has_zero_http_imports_and_no_training_leakage():
    online_dir = "src/online"
    violations = []

    for root, _, files in os.walk(online_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=filepath)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in BANNED_HTTP_MODULES:
                                violations.append(f"HTTP import in {filepath}: {alias.name}")
                            if any(alias.name.startswith(b) for b in BANNED_OFFLINE_IMPORTS):
                                violations.append(f"Training leak import in {filepath}: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        if mod in BANNED_HTTP_MODULES or any(mod.startswith(b) for b in BANNED_HTTP_MODULES):
                            violations.append(f"HTTP from-import in {filepath}: {mod}")
                        if any(mod.startswith(b) for b in BANNED_OFFLINE_IMPORTS):
                            violations.append(f"Training leak from-import in {filepath}: {mod}")

    assert len(violations) == 0, f"Found isolation violations: {violations}"
