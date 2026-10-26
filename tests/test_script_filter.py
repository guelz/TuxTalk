"""Latin only (script_filter.py): which of Parakeet's word pieces are
Cyrillic or Greek, and when they are kept out of a transcription.

Measured Oct 26 2026 on Parakeet Ultra with short Type at pause chunks:
English said into a clipping mic came out as "Системс", "Суксес", "Свис чиз"
(15 of 71 chunks on cbox). With these pieces blocked every one came out in
Latin letters, and no chunk that was Latin already changed.
"""
import numpy as np
import pytest

from tuxtalk.script_filter import EXEMPT_LANGUAGES, LogitMask, non_latin_ids, should_block

# Word pieces as onnx-asr and the engine hand them over. 3 is "с" (Cyrillic)
# followed by a Latin "o", 4 a Latin "H" with a Cyrillic "и".
VOCAB = {0: "<unk>", 1: "▁the", 2: "▁Ол", 3: "сo", 4: "Hи", 5: "č", 6: "ß", 7: "▁λ",
         8: "ἀ", 9: "<blk>"}


def test_non_latin_ids_finds_cyrillic_greek_and_mixed_pieces():
    assert non_latin_ids(VOCAB) == [2, 3, 4, 7, 8]


def test_latin_diacritics_and_specials_stay():
    assert not {0, 1, 5, 6, 9} & set(non_latin_ids(VOCAB))


def test_the_exempt_languages_are_the_cyrillic_and_greek_ones_parakeet_knows():
    assert EXEMPT_LANGUAGES == {"ru", "uk", "bg", "el"}


@pytest.mark.parametrize("language,auto,usual,blocked", [
    ("de", (), {"de"}, True),
    ("en", (), {"ru"}, True),                 # a chosen language wins over the usual ones
    ("ru", (), {"de"}, False),
    ("uk", (), set(), False),
    ("el", (), set(), False),
    (None, ("en", "de"), {"ru"}, True),       # so does auto-detect's list
    (None, ("en", "ru"), {"de"}, False),
    (None, (), {"de"}, True),                 # plain auto: the languages this person uses
    (None, (), {"en", "ru"}, False),          # Russian picked at first run, English desktop
    (None, (), {"bg"}, False),
    ("", (), set(), True),
])
def test_should_block(language, auto, usual, blocked):
    assert should_block(language, auto, usual) is blocked


def test_logit_mask_moves_the_pick_and_counts_it():
    mask = LogitMask([2, 3])
    out = mask.apply(np.array([0.0, 1.0, 5.0, 0.5], dtype=np.float32))
    assert int(out.argmax()) == 1
    assert out[2] == -np.inf and out[3] == -np.inf
    assert mask.hits == 1


def test_logit_mask_leaves_latin_picks_alone():
    mask = LogitMask([2])
    logits = np.array([0.0, 3.0, 1.0], dtype=np.float32)
    out = mask.apply(logits)
    assert int(out.argmax()) == 1
    assert mask.hits == 0
    assert logits[2] == 1.0              # the caller's array is left as it was


def test_logit_mask_reset_starts_the_count_again():
    mask = LogitMask([0])
    mask.apply(np.array([2.0, 1.0], dtype=np.float32))
    mask.reset()
    assert mask.hits == 0


def test_an_empty_mask_is_falsy():
    assert not LogitMask([])
    assert LogitMask([1])
