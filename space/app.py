"""Hugging Face Space entry point for Confidant.

The Space holds no application code of its own: it runs the demo from the public
repository https://github.com/yannabadie/confidant at the reference below, so what
judges run is exactly what they can read. API keys come from the Space secrets
(NEBIUS_API_KEY, TAVILY_API_KEY); without them the demo runs in replay mode.
"""
import importlib.util
import os
import subprocess
import sys

REPO = "https://github.com/yannabadie/confidant"
REF = os.environ.get("CONFIDANT_REF", "main")
DEST = "/tmp/confidant-src"

if not os.path.isdir(DEST):
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", REF, REPO, DEST], check=True)
sys.path.insert(0, DEST)
os.chdir(DEST)

_spec = importlib.util.spec_from_file_location("confidant_demo_app", os.path.join(DEST, "app.py"))
_module = importlib.util.module_from_spec(_spec)
sys.modules["confidant_demo_app"] = _module
_spec.loader.exec_module(_module)
demo = _module.demo

if __name__ == "__main__":
    demo.launch()
