"""Print the installed distributions as a lock file (the venv has no pip)."""
import importlib.metadata as m
import platform

print(f"# python {platform.python_version()} on {platform.platform()}")
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower()):
    print(f"{d.metadata['Name']}=={d.version}")
