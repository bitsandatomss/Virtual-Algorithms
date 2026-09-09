from __future__ import annotations

from typing import Any


REQUEST_VOTE = "RequestVote"
REQUEST_VOTE_RESPONSE = "RequestVoteResponse"
PRE_VOTE = "PreVote"
PRE_VOTE_RESPONSE = "PreVoteResponse"
APPEND_ENTRIES = "AppendEntries"
APPEND_ENTRIES_RESPONSE = "AppendEntriesResponse"
CLIENT_COMMAND = "ClientCommand"


def rpc(src: int, dst: int, rpc_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "kind": "rpc",
        "src": src,
        "dst": dst,
        "type": rpc_type,
        **payload,
    }


def event(name: str, **payload: Any) -> dict[str, Any]:
    return {"kind": "event", "event": name, **payload}


def command(name: str, **payload: Any) -> dict[str, Any]:
    return {"kind": "control", "command": name, **payload}
