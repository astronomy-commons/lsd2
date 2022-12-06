"""Set of convenience methods for testing file contents"""

import os
import re


def assert_text_file_matches(expected_lines, file_name):
    """
    Convenience method to read a text file and compare the contents,
    line for line, against regular expressions.

    It can be easier to see differences in indivudual lines
    when file contents grow to be large.
    """
    assert os.path.exists(file_name)
    metadata_file = open(
        file_name,
        "r",
        encoding="utf-8",
    )

    contents = metadata_file.readlines()

    assert len(expected_lines) == len(contents)
    for i, expected in enumerate(expected_lines):
        assert re.match(expected, contents[i])

    metadata_file.close()
