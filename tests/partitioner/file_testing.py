"""Set of convenience methods for testing file contents"""

import os
import re


def assert_text_file_matches(expected_lines, file_name):
    """Convenience method to read a text file and compare the contents, line for line.

    When file contents get even a little bit big, it can be difficult to see
    the difference between an actual file and the expected contents without
    increased testing verbosity. This helper compares files line-by-line,
    using the provided strings or regular expressions.

    Notes:
        Because we check strings as regular expressions, you may need to escape some
        contents of `expected_lines`.

    Args:
        expected_lines(:obj:`string array`) list of strings, formatted as regular expressions.
        file_name (str): fully-specified path of the file to read
    """
    assert os.path.exists(file_name)
    metadata_file = open(
        file_name,
        "r",
        encoding="utf-8",
    )

    contents = metadata_file.readlines()

    assert len(expected_lines) == len(
        contents
    ), f"files not the same length ({len(contents)} vs {len(expected_lines)})"
    for i, expected in enumerate(expected_lines):
        assert re.match(
            expected, contents[i]
        ), f"files do not match at line {i+1} (actual: [{contents[i]}] vs expected: [{expected}])"

    metadata_file.close()
