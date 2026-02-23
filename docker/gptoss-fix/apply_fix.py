#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch vLLM to propagate default stop_token_ids to per-request SamplingParams.

Fix for https://github.com/vllm-project/vllm/issues/22519

Works with both:
  - Flat layout (older builds): vllm/entrypoints/openai/protocol.py
  - Subdirectory layout (current source):
      vllm/entrypoints/openai/chat_completion/protocol.py
      vllm/entrypoints/openai/completion/protocol.py

Also patches harmony_utils.py with a safety-net stop-token break.
"""
import os
import sys

import vllm

VLLM_BASE = os.path.dirname(vllm.__file__)

FLAT_PROTOCOL = os.path.join(
    VLLM_BASE, "entrypoints", "openai", "protocol.py"
)
CHAT_PROTOCOL = os.path.join(
    VLLM_BASE, "entrypoints", "openai", "chat_completion", "protocol.py"
)
COMPLETION_PROTOCOL = os.path.join(
    VLLM_BASE, "entrypoints", "openai", "completion", "protocol.py"
)
HARMONY_PATH = os.path.join(
    VLLM_BASE, "entrypoints", "openai", "parser", "harmony_utils.py"
)

MERGE_BLOCK = (
    "        # Merge server-default stop_token_ids (e.g., model-specific tokens\n"
    "        # like </call> for gpt-oss) with any request-specified ones\n"
    "        stop_token_ids = self.stop_token_ids or None\n"
    '        default_stop_ids = default_sampling_params.get("stop_token_ids")\n'
    "        if default_stop_ids:\n"
    "            if stop_token_ids is None:\n"
    "                stop_token_ids = list(default_stop_ids)\n"
    "            else:\n"
    "                stop_token_ids = list(set(stop_token_ids) | set(default_stop_ids))\n"
    "\n"
)

MARKER = "default_stop_ids"


def patch_protocol(path: str) -> bool:
    """Inject stop_token_ids merge logic before every
    `prompt_logprobs = self.prompt_logprobs` inside a to_sampling_params
    method, and replace `stop_token_ids=self.stop_token_ids` with
    `stop_token_ids=stop_token_ids` in SamplingParams.from_optional().
    Returns True if any changes were made.
    """
    if not os.path.isfile(path):
        return False

    with open(path, "r") as f:
        content = f.read()

    if MARKER in content:
        print(f"  [skip] {path} already patched")
        return False

    lines = content.split("\n")
    new_lines: list[str] = []
    patched = 0

    for i, line in enumerate(lines):
        if (
            line.strip() == "prompt_logprobs = self.prompt_logprobs"
            and not any(MARKER in lines[j] for j in range(max(0, i - 10), i))
        ):
            new_lines.append(MERGE_BLOCK.rstrip("\n"))
            patched += 1
        new_lines.append(line)

    content = "\n".join(new_lines)
    content = content.replace(
        "stop_token_ids=self.stop_token_ids,",
        "stop_token_ids=stop_token_ids,",
    )

    with open(path, "w") as f:
        f.write(content)
    print(f"  [patched] {path}  ({patched} insertion(s))")
    return True


def patch_harmony(path: str) -> bool:
    """Add stop-token break to parse_output_into_messages."""
    if not os.path.isfile(path):
        print(f"  [skip] {path} not found")
        return False

    with open(path, "r") as f:
        content = f.read()

    old = (
        "def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:\n"
        "    parser = get_streamable_parser_for_assistant()\n"
        "    for token_id in token_ids:\n"
        "        parser.process(token_id)\n"
        "    return parser"
    )
    new = (
        "def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:\n"
        "    parser = get_streamable_parser_for_assistant()\n"
        "    stop_tokens = set(get_stop_tokens_for_assistant_actions())\n"
        "    for token_id in token_ids:\n"
        "        parser.process(token_id)\n"
        "        if token_id in stop_tokens:\n"
        "            break\n"
        "    return parser"
    )

    if new in content:
        print(f"  [skip] {path} already patched")
        return False
    if old not in content:
        print(f"  [warn] {path} does not match expected pattern — skipping")
        return False

    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [patched] {path}")
    return True


def main() -> None:
    print(f"vLLM base: {VLLM_BASE}")
    changed = False

    if os.path.isfile(FLAT_PROTOCOL):
        print("\nDetected flat layout (single protocol.py)")
        changed |= patch_protocol(FLAT_PROTOCOL)
    elif os.path.isfile(CHAT_PROTOCOL) and os.path.isfile(COMPLETION_PROTOCOL):
        print("\nDetected subdirectory layout")
        changed |= patch_protocol(CHAT_PROTOCOL)
        changed |= patch_protocol(COMPLETION_PROTOCOL)
    else:
        print("ERROR: Could not locate protocol.py", file=sys.stderr)
        sys.exit(1)

    print()
    changed |= patch_harmony(HARMONY_PATH)

    if changed:
        print("\nAll fixes applied successfully!")
    else:
        print("\nNothing to patch — fixes already present.")


if __name__ == "__main__":
    main()
