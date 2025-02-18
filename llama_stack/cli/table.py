# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
import textwrap
from typing import Iterable

from termcolor import cprint


def strip_ansi_colors(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def format_row(row, col_widths):
    def wrap(text, width):
        lines = []
        for line in text.split("\n"):
            if line.strip() == "":
                lines.append("")
            else:
                lines.extend(textwrap.wrap(line, width, break_long_words=False, replace_whitespace=False))
        return lines

    wrapped = [wrap(item, width) for item, width in zip(row, col_widths, strict=False)]
    max_lines = max(len(subrow) for subrow in wrapped)

    lines = []
    for i in range(max_lines):
        line = []
        for cell_lines, width in zip(wrapped, col_widths, strict=False):
            value = cell_lines[i] if i < len(cell_lines) else ""
            line.append(value + " " * (width - len(strip_ansi_colors(value))))
        lines.append("| " + (" | ".join(line)) + " |")

    return "\n".join(lines)


def print_table(rows, headers=None, separate_rows: bool = False, sort_by: Iterable[int] = tuple()):
    def itemlen(item):
        return max([len(line) for line in strip_ansi_colors(item).split("\n")])

    rows = [[x or "" for x in row] for row in rows]

    if sort_by:
        rows.sort(key=lambda x: tuple(x[i] for i in sort_by))

    if not headers:
        col_widths = [max(itemlen(item) for item in col) for col in zip(*rows, strict=False)]
    else:
        col_widths = [
            max(
                itemlen(header),
                max(itemlen(item) for item in col),
            )
            for header, col in zip(headers, zip(*rows, strict=False), strict=False)
        ]
    col_widths = [min(w, 80) for w in col_widths]

    header_line = "+".join("-" * (width + 2) for width in col_widths)
    header_line = f"+{header_line}+"

    if headers:
        print(header_line)
        cprint(format_row(headers, col_widths), "white", attrs=["bold"])

    print(header_line)
    for row in rows:
        print(format_row(row, col_widths))
        if separate_rows:
            print(header_line)

    if not separate_rows:
        print(header_line)
