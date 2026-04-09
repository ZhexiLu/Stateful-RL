# ESA predicate evaluation, progress reward, tool execution, control block parsing.

import asyncio
import re
import sqlite3

from esa_config import ESA_CONFIGS, logger

from awm.core.agent import parse_tool_calls, parse_call_tool_arguments


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Execution
# ═══════════════════════════════════════════════════════════════════════════════
async def _execute_tool_call(mcp, response_text, tools_text):
    """Execute tool call, return (result_text, done, is_write_op)."""
    tool_calls = parse_tool_calls(response_text)
    if not tool_calls:
        return "", True, False

    tc = tool_calls[0]
    name, arguments = tc["name"], tc["arguments"]

    if name == "list_tools":
        return tools_text, False, False

    if name == "call_tool":
        tool_name, tool_args = parse_call_tool_arguments(arguments)
        is_write = _is_write_operation(tool_name, tool_args)
        try:
            result = await mcp.call_tool(tool_name, tool_args)
            return result, False, is_write
        except asyncio.TimeoutError:
            return f"Error: Tool call timed out after {ESA_CONFIGS['tool_timeout']}s", False, is_write
        except Exception as e:
            return f"Error: {e}", False, is_write

    return f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'.", False, False


_WRITE_KEYWORDS = {
    "create", "add", "insert", "update", "delete", "remove",
    "post", "put", "patch", "set", "modify", "change",
    "book", "cancel", "submit", "send", "publish", "edit",
    "purchase", "buy", "order", "checkout", "pay", "register",
    "subscribe", "unsubscribe", "assign", "unassign",
    "approve", "reject", "close", "resolve", "archive",
    "restore", "enable", "disable", "activate", "deactivate",
    "reset", "revoke", "grant", "transfer", "move",
    "upload", "import", "export", "schedule", "complete",
    "start", "stop", "pause", "resume", "apply",
    "link", "unlink", "connect", "disconnect",
    "follow", "unfollow", "block", "unblock", "mute",
    "pin", "unpin", "save", "unsave", "mark", "flag",
    "rate", "review", "comment", "reply", "react",
    "invite", "accept", "decline", "join", "leave",
    "rename", "configure", "upsert", "merge", "split",
}


def _is_write_operation(tool_name: str, tool_args: dict) -> bool:
    """Heuristic to detect state-mutating operations."""
    name_lower = tool_name.lower()
    for kw in _WRITE_KEYWORDS:
        if kw in name_lower:
            return True
    method = str(tool_args.get("method", "")).upper()
    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Control Block Parsing
# ═══════════════════════════════════════════════════════════════════════════════
VALID_STATUSES = {"CONTINUE", "VERIFY", "REPLAN", "STOP"}

_STATUS_PATTERN = re.compile(
    r"(?:<status>)?\s*(CONTINUE|VERIFY|REPLAN|STOP)\s*</status>", re.IGNORECASE
)


def _parse_control_block(response_text: str) -> tuple[str, bool]:
    """Extract execution-status from model output.

    Returns (status_str, is_valid_format).
    If no valid control block found, returns ("CONTINUE", False).
    """
    match = _STATUS_PATTERN.search(response_text)
    if match:
        status = match.group(1).upper()
        if status in VALID_STATUSES:
            return status, True
    return "CONTINUE", False


# ═══════════════════════════════════════════════════════════════════════════════
# Predicate Evaluation & Progress Reward
# ═══════════════════════════════════════════════════════════════════════════════
def _is_tautological(p: dict) -> bool:
    """Check if a predicate is already satisfied on the initial DB."""
    iv = p.get("initial_value")
    ev = p.get("expected")
    ct = p.get("check_type", "existence")
    if ct == "count_gte":
        if iv is not None and ev is not None:
            try:
                return float(iv) >= float(ev)
            except (TypeError, ValueError):
                return iv == ev
    return iv == ev


def _check_predicate(cursor, pred: dict) -> bool:
    """Evaluate a single predicate against a DB cursor. Returns True if satisfied."""
    cursor.execute(pred["sql"])
    result = cursor.fetchone()
    actual = result[0] if result else None
    check_type = pred.get("check_type", "existence")
    expected = pred.get("expected")

    if check_type == "existence":
        return (actual == expected) if expected is not None else bool(actual)
    elif check_type == "count_gte":
        return actual is not None and expected is not None and actual >= expected
    elif check_type == "exact_match":
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(actual - expected) < 0.01
        return str(actual) == str(expected) if actual is not None else False
    elif check_type == "text_contains":
        return (str(expected).lower() in str(actual).lower()) if actual and expected else False
    else:
        return (actual == expected) if expected is not None else bool(actual)


def _get_useful_predicates(predicates: list[dict]) -> list[dict]:
    """Filter to executable, non-tautological predicates."""
    return [p for p in predicates
            if p.get("executable", True) and not _is_tautological(p)]


def _evaluate_predicates(db_path: str, predicates: list[dict]) -> float:
    """Evaluate predicates against current DB state. Returns Phi in [0, 1]."""
    useful = _get_useful_predicates(predicates)
    if not useful:
        return 0.0

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        return 0.0

    scores = []
    for pred in useful:
        try:
            scores.append(1.0 if _check_predicate(cursor, pred) else 0.0)
        except Exception:
            scores.append(0.0)

    conn.close()
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_predicate_signature(db_path: str, predicates: list[dict]) -> tuple[float, tuple]:
    """Evaluate predicates and return (phi, binary_signature_tuple)."""
    useful = _get_useful_predicates(predicates)
    if not useful:
        return 0.0, ()

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        return 0.0, tuple(0 for _ in useful)

    bits = []
    for pred in useful:
        try:
            bits.append(1 if _check_predicate(cursor, pred) else 0)
        except Exception:
            bits.append(0)

    conn.close()
    phi = sum(bits) / len(bits) if bits else 0.0
    return phi, tuple(bits)


def _compute_progress_reward(
    db_path: str,
    prev_phi: float,
    predicate_sqls: list[dict],
) -> tuple[float, float, tuple]:
    """Compute R_prog = Phi(s_{t+1}) - Phi(s_t).

    Returns (R_prog, current_phi, current_signature).
    """
    current_phi, signature = _evaluate_predicate_signature(db_path, predicate_sqls)
    return current_phi - prev_phi, current_phi, signature
