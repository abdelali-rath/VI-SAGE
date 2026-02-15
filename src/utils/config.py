"""
Resolve ${a.b.c} placeholders in YAML-like structures.

Allows config files to reference other keys, e.g.:
  data_dir: "${paths.data.utkface}"
"""

import re

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def lookup_path(cfg: dict, path: str):
    """
    Resolve a dotted path like 'paths.checkpoints.age' inside cfg.

    Returns the value at that path, or None if any key is missing.
    """
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def resolve_placeholders(obj, root_cfg: dict):
    """
    Recursively resolve ${a.b.c} placeholders in a loaded YAML structure.

    root_cfg is the full config dict used to resolve placeholder keys.
    """
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, root_cfg) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, root_cfg) for v in obj]
    if isinstance(obj, str):

        def repl(match):
            key = match.group(1)
            val = lookup_path(root_cfg, key)
            return str(val) if val is not None else match.group(0)

        return _PLACEHOLDER_RE.sub(repl, obj)
    return obj
