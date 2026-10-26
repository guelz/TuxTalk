"""
Latin only: keep Parakeet from writing Cyrillic or Greek for someone who
doesn't dictate in those scripts.

Parakeet has no language input. Its 25 languages share one vocabulary of word
pieces, Russian, Ukrainian and Bulgarian among them, and on a short or
distorted bit of audio it sometimes settles on the wrong script: English
"Systems" comes out as "Системс", "Swiss cheese" as "Свис чиз". Type at
pause makes this common, because its chunks are often a word or two long.

The fix is in the decoder: before it picks the next piece, the Cyrillic and
Greek pieces are made impossible (their scores set to minus infinity), so
the best Latin piece wins, which is usually the right word. Speech that
comes out in Latin letters anyway is not changed at all, because the piece
the decoder picks is the same. parakeet_engine.py does this on the
processor; on the graphics chip the engine does it (parakeet_gpu_worker.py,
vulkan-engine-patches/parakeet-latin-only.patch) with the ids found here.
"""
import re

import numpy as np

# Greek, Greek Extended, Cyrillic and Cyrillic Supplement. Latin letters with
# accents (č, ä, ß) are not in these ranges: German, Czech and Polish need them.
NON_LATIN = re.compile("[Ͱ-Ͽἀ-῿Ѐ-ӿԀ-ԯ]")

# The languages Parakeet knows that are written in those scripts.
EXEMPT_LANGUAGES = frozenset({"ru", "uk", "bg", "el"})


def non_latin_ids(vocab) -> list[int]:
    """The ids, in order, of the word pieces in *vocab* (id -> text) with a
    Cyrillic or Greek letter in them, mixed pieces like "Hи" included."""
    return sorted(i for i, text in vocab.items() if NON_LATIN.search(text))


def should_block(language, auto_languages, environ) -> bool:
    """Whether to keep Cyrillic and Greek out, given the language chosen
    (None or "" for auto-detect), auto-detect's list of languages and the
    environment. Not when the person could be dictating one of
    EXEMPT_LANGUAGES: the chosen language, else every language on the list,
    else the desktop's own language from the locale."""
    if language:
        possible = {language}
    elif auto_languages:
        possible = set(auto_languages)
    else:
        possible = {_locale_language(environ)}
    return not possible & EXEMPT_LANGUAGES


def _locale_language(environ) -> str:
    """The language code of the locale in *environ* ("de" for de_CH.UTF-8),
    looked up the way programs do: LC_ALL, then LC_MESSAGES, then LANG."""
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = environ.get(name, "")
        if value:
            return re.split(r"[_.@]", value, maxsplit=1)[0].lower()
    return ""


class LogitMask:
    """Blocks word pieces *ids* in the decoder's scores, and counts the steps
    where a blocked piece would otherwise have been written (hits): the log
    reports that count, so it shows when blocking changed something."""

    def __init__(self, ids):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.hits = 0

    def __bool__(self):
        return bool(self.ids.size)

    def apply(self, logits):
        """*logits* with the blocked pieces at minus infinity, as a copy."""
        out = np.array(logits, copy=True)
        if self.ids.size and np.isin(int(out.argmax()), self.ids):
            self.hits += 1
        out[self.ids] = -np.inf
        return out

    def reset(self):
        self.hits = 0
