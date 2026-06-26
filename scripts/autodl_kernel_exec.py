#!/usr/bin/env python3
"""Execute Python code on a remote Jupyter kernel, for AutoDL bring-up."""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import uuid
from typing import Any

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute code on a remote Jupyter kernel")
    parser.add_argument("--base-url", required=True, help="Remote base URL, e.g. https://host:8443")
    parser.add_argument("--token", required=True, help="Jupyter token")
    parser.add_argument("--kernel-id", required=True, help="Kernel ID from /api/sessions or /api/kernels")
    parser.add_argument(
        "--code",
        help="Inline Python code to execute on the remote kernel",
    )
    parser.add_argument(
        "--code-file",
        help="Local file containing Python code to execute on the remote kernel",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Seconds to wait for remote execution before failing",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification",
    )
    return parser.parse_args()


def load_code(args: argparse.Namespace) -> str:
    if bool(args.code) == bool(args.code_file):
        raise SystemExit("Provide exactly one of --code or --code-file")
    if args.code:
        return args.code
    if args.code_file == "-":
        return sys.stdin.read()
    return open(args.code_file, "r", encoding="utf-8").read()


def websocket_url(base_url: str, kernel_id: str, token: str, session_id: str) -> str:
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://") :]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://") :]
    else:
        raise SystemExit("base-url must start with http:// or https://")
    return (
        f"{ws_base}/jupyter/api/kernels/{kernel_id}/channels"
        f"?session_id={session_id}&token={token}"
    )


async def execute_code(
    *,
    ws_url: str,
    code: str,
    timeout: int,
    ssl_context: ssl.SSLContext | None,
    session_id: str,
) -> int:
    msg_id = uuid.uuid4().hex
    request = {
        "header": {
            "msg_id": msg_id,
            "username": "codex",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
        "buffers": [],
    }

    async with websockets.connect(
        ws_url,
        ssl=ssl_context,
        open_timeout=20,
        max_size=2**24,
    ) as ws:
        await ws.send(json.dumps(request))
        stream_out = sys.stdout
        stream_err = sys.stderr
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            data = json.loads(raw)
            parent = data.get("parent_header", {})
            if parent.get("msg_id") != msg_id:
                continue

            msg_type = data.get("header", {}).get("msg_type")
            if msg_type == "stream":
                name = data["content"].get("name", "stdout")
                text = data["content"].get("text", "")
                target = stream_err if name == "stderr" else stream_out
                target.write(text)
                target.flush()
                continue

            if msg_type == "execute_result":
                payload = data["content"].get("data", {}).get("text/plain")
                if payload:
                    stream_out.write(payload + "\n")
                    stream_out.flush()
                continue

            if msg_type == "error":
                traceback = data["content"].get("traceback", [])
                if traceback:
                    stream_err.write("\n".join(traceback) + "\n")
                else:
                    stream_err.write(
                        f"{data['content'].get('ename')}: {data['content'].get('evalue')}\n"
                    )
                stream_err.flush()
                return 1

            if msg_type == "status" and data.get("content", {}).get("execution_state") == "idle":
                return 0


def main() -> None:
    args = parse_args()
    code = load_code(args)
    ssl_context: ssl.SSLContext | None = None
    if args.base_url.startswith("https://"):
        ssl_context = ssl.create_default_context()
        if args.insecure:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
    session_id = uuid.uuid4().hex
    ws_url = websocket_url(args.base_url.rstrip("/"), args.kernel_id, args.token, session_id)
    exit_code = asyncio.run(
        execute_code(
            ws_url=ws_url,
            code=code,
            timeout=args.timeout,
            ssl_context=ssl_context,
            session_id=session_id,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
