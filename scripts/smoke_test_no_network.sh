#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Running Offline Edge Zero-Network Smoke Test"
echo "============================================================"

# 1. Static AST Inspection: Verify NO HTTP client imports in src/online/
echo "Step 1: Inspecting src/online/ import tree for HTTP clients..."
python -c "
import ast
import os
import sys

banned_modules = {'requests', 'urllib.request', 'httpx', 'aiohttp', 'urllib3', 'http.client'}
online_dir = 'src/online'
violations = []

for root, _, files in os.walk(online_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in banned_modules:
                            violations.append((filepath, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ''
                    if mod in banned_modules or any(mod.startswith(b) for b in banned_modules):
                        violations.append((filepath, mod))

if violations:
    print(f'ERROR: Banned HTTP client imports detected in src/online/: {violations}')
    sys.exit(1)
else:
    print('Static Import Inspection Passed: 0 HTTP client imports found in src/online/.')
"

# 2. Runtime Network Block Test
echo "Step 2: Testing live inference execution with network sockets disabled..."
python -c "
import socket
import sys

# Monkey patch socket to forbid any external outbound connections
def guarded_socket_connect(*args, **kwargs):
    raise ConnectionRefusedError('EMERGENCY: Outbound network call detected in offline edge inference!')

socket.socket.connect = guarded_socket_connect

from src.online.infer import ComplianceInferenceEngine

engine = ComplianceInferenceEngine()
query = 'What is the human oversight intervention threshold under Clause AI-102?'
result = engine.process_query(query)

assert result['status'] == 'success', f'Expected success, got {result}'
assert result['confidence_passed'] is True
assert '15 seconds' in result['answer'] or 'Clause AI-102' in result['answer']
print('Runtime Zero-Network Verification Passed: Query answered completely offline.')
"

echo "============================================================"
echo " ALL ZERO-NETWORK COMPLIANCE GATES PASSED."
echo "============================================================"
