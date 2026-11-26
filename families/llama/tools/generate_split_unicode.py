# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

URL = "https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt"
SHA256 = "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f"


def generate(data: bytes) -> str:
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Expected the pinned Unicode 16.0.0 UnicodeData.txt")
    groups = {"L": [], "N": []}
    first = 0
    for line in data.decode("utf-8").splitlines():
        fields = line.split(";")
        code = int(fields[0], 16)
        if fields[1].endswith(", First>"):
            first = code
            continue
        start = first if fields[1].endswith(", Last>") else code
        category = fields[2][0]
        if category in groups:
            groups[category].extend(range(start, code + 1))
    output = [
        "/*",
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        " * SPDX-License-Identifier: Apache-2.0",
        " */",
        "",
        "// Generated from the pinned UnicodeData.txt by generate_split_unicode.py.",
        "// Note (Jiaxin Deng): Reference properties must not vary with the host ICU version.",
        "// clang-format off",
        "#pragma once",
        "namespace trtmc::llama::split_unicode {",
    ]
    for category, values in groups.items():
        ranges = []
        for value in values:
            if ranges and value == ranges[-1][1] + 1:
                ranges[-1][1] = value
            else:
                ranges.append([value, value])
        pattern = "[" + "".join(
            f"\\x{{{lo:x}}}" + (f"-\\x{{{hi:x}}}" if hi != lo else "") for lo, hi in ranges
        ) + "]"
        output.append(f"inline constexpr char k{category}[] =")
        output.extend("    " + json.dumps(pattern[i:i + 64]) for i in range(0, len(pattern), 64))
        output[-1] += ";"
    output += ["} // namespace trtmc::llama::split_unicode", "// clang-format on", ""]
    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicode-data", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "runtime/split_unicode.h")
    args = parser.parse_args()
    content = args.unicode_data.read_bytes() if args.unicode_data else urllib.request.urlopen(URL).read()
    args.output.write_text(generate(content), encoding="utf-8")
