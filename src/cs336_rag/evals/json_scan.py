"""Extract JSON values embedded in free-form model replies.

Models wrap JSON in code fences, preamble prose, or extra objects. Scanning
for the first value that both parses *and* satisfies the caller's predicate
is more robust than a regex span: it tolerates stray brackets, brackets
inside strings, and leading objects that are not the payload.
"""

import json
from collections.abc import Callable

_OPENERS = {"[": list, "{": dict}


def scan_json[T](raw: str, opener: str, extract: Callable[[object], T | None]) -> T | None:
    """Return the first JSON value at ``opener`` that ``extract`` accepts.

    ``extract`` receives each successfully decoded value and returns the
    converted result, or ``None`` to keep scanning. Returns ``None`` when no
    value in ``raw`` is both parseable and accepted.
    """
    expected = _OPENERS[opener]
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != opener:
            continue
        try:
            data, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, expected):
            continue
        result = extract(data)
        if result is not None:
            return result
    return None
