"""Parakeet engine wiring.

Parakeet is offered as one more model name ("parakeet-v3") so that every model
picker, the config validator and the download dialogs handle it without their
own special cases. These tests pin down the places where it genuinely differs
from Whisper: how "is it downloaded?" is answered and how dictation audio
reaches it. Where its files come from is tests/test_parakeet_ultra.py.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_parakeet_is_offered_valid_and_downloadable():
    from tuxtalk.config import VALID_MODELS
    from tuxtalk.model_helper import MODEL_DISPLAY_SIZES, OFFERED_MODELS
    from tuxtalk.parakeet_engine import PARAKEET_MODEL

    assert PARAKEET_MODEL in OFFERED_MODELS
    assert PARAKEET_MODEL in VALID_MODELS
    assert PARAKEET_MODEL in MODEL_DISPLAY_SIZES


def test_every_offered_model_has_a_label_on_every_picker():
    """The picker indexes a label dict by model id; a missing key crashes the
    screen as it opens. (First run's card has no full picker.)"""
    from tuxtalk.model_helper import OFFERED_MODELS

    for path, dict_name in (("src/tuxtalk/prefs.py", "_MODEL_DISPLAY"),):
        source = (ROOT / path).read_text()
        body = re.search(dict_name + r" = \{(.*?)\n\s*\}", source, re.S).group(1)
        keys = set(re.findall(r'"([^"]+)":', body))
        missing = set(OFFERED_MODELS) - keys
        assert not missing, f"{path} {dict_name} has no label for {sorted(missing)}"


def test_whisper_downloads_are_not_filtered(monkeypatch):
    import huggingface_hub
    from tuxtalk import model_helper

    monkeypatch.setattr(
        huggingface_hub, "list_repo_tree",
        lambda repo_id, recursive=True: [SimpleNamespace(path="model.bin", size=10),
                                         SimpleNamespace(path="README.md", size=1)],
    )
    files = model_helper._files_to_download("Systran/faster-whisper-small")
    assert {p for p, _ in files} == {"model.bin", "README.md"}


def test_cache_checks_use_parakeet_files(monkeypatch):
    from tuxtalk import model_helper

    monkeypatch.setattr(model_helper, "_parakeet_cached_dir", lambda: None)
    assert model_helper.is_model_cached_fast("parakeet-v3") is False

    monkeypatch.setattr(model_helper, "_parakeet_cached_dir", lambda: "/some/dir")
    assert model_helper.is_model_cached_fast("parakeet-v3") is True


def test_dictation_audio_goes_to_parakeet(monkeypatch):
    from tuxtalk import app
    from tuxtalk.parakeet_engine import ParakeetModel

    received = []
    fake = ParakeetModel.__new__(ParakeetModel)  # skip loading the real model
    fake.recognize = lambda audio: received.append(audio) or "Hello there."
    monkeypatch.setattr(app, "model", fake, raising=False)

    audio = np.zeros(16000, dtype=np.float32)
    assert app._transcribe_audio(audio, language="de") == "Hello there."
    assert received and received[0] is audio


def test_parakeet_silence_returns_none(monkeypatch):
    from tuxtalk import app
    from tuxtalk.parakeet_engine import ParakeetModel

    fake = ParakeetModel.__new__(ParakeetModel)
    fake.recognize = lambda audio: ""
    monkeypatch.setattr(app, "model", fake, raising=False)

    assert app._transcribe_audio(np.zeros(8000, dtype=np.float32), language=None) is None


def test_the_download_question_names_the_model_as_the_menus_do(monkeypatch):
    """It said "Download Small Model?" where the menus said "Whisper Small"
    (review of 0.17)."""
    from tuxtalk import model_helper
    asked = []

    class Question:
        def __init__(self, **kwargs):
            asked.append(kwargs["text"])
        def format_secondary_text(self, text): pass
        def set_position(self, position): pass
        def destroy(self): pass
        def run(self):
            return model_helper.Gtk.ResponseType.CANCEL

    monkeypatch.setattr(model_helper, "is_model_cached_fast", lambda m: False)
    monkeypatch.setattr(model_helper, "message_dialog", lambda **k: Question(**k))
    assert model_helper.download_model_with_progress("small") is None
    assert asked == ["Download Whisper Small Model?"]


def test_menus_show_the_device_parakeet_really_uses():
    """Picking Parakeet in Preferences can leave device = "cuda" in the config;
    both the tray and the GNOME extension (via GetDeviceType) must say CPU."""
    from tuxtalk.parakeet_engine import effective_device
    assert effective_device("parakeet-v3", "cuda") == "cpu"
    assert effective_device("large-v3", "cuda") == "cuda"
    assert effective_device("small", "cpu") == "cpu"


def test_parakeet_is_on_the_graphics_chip_only_once_its_vulkan_files_are_there(monkeypatch):
    from tuxtalk import whisper_vulkan as wv
    from tuxtalk.parakeet_engine import effective_device
    monkeypatch.setattr(wv, "is_installed", lambda m: False)
    assert effective_device("parakeet-v3", "vulkan") == "cpu"
    monkeypatch.setattr(wv, "is_installed", lambda m: True)
    assert effective_device("parakeet-v3", "vulkan") == "vulkan"
    assert effective_device("parakeet-v3", "cuda") == "cpu"     # CUDA has no Parakeet

    from tuxtalk.dbus_service import TuxTalkDBusService
    svc = TuxTalkDBusService.__new__(TuxTalkDBusService)
    svc.app = SimpleNamespace(config=SimpleNamespace(model="parakeet-v3", device="cuda"))
    assert TuxTalkDBusService.GetDeviceType(svc) == "cpu"


def _parakeet_saying(monkeypatch, text, nobody_talked):
    from tuxtalk import app, silence
    from tuxtalk.parakeet_engine import ParakeetModel
    fake = ParakeetModel.__new__(ParakeetModel)
    fake.recognize = lambda audio: text
    monkeypatch.setattr(app, "model", fake, raising=False)
    monkeypatch.setattr(silence, "has_no_speech", lambda a, sr: nobody_talked)
    return app._transcribe_audio(np.zeros(16000, dtype=np.float32), language=None)


def test_parakeet_inventing_mm_hmm_in_silence_types_nothing(monkeypatch):
    """0.15.0 testing: an F9 tap with nothing said typed "Mm-hmm." into Konsole."""
    assert _parakeet_saying(monkeypatch, "Mm-hmm.", nobody_talked=True) is None


def test_parakeet_keeps_a_real_mm_hmm(monkeypatch):
    assert _parakeet_saying(monkeypatch, "Mm-hmm.", nobody_talked=False) == "Mm-hmm."


def test_parakeet_keeps_real_words_even_in_a_quiet_recording(monkeypatch):
    """Only stock fillers are judged, never a real sentence."""
    assert _parakeet_saying(monkeypatch, "Hello there.", nobody_talked=True) == "Hello there."


def test_nothing_recognized_plays_the_cancel_beep(monkeypatch):
    """A recording that types nothing ends with the cancel beep, not silence."""
    from tuxtalk import app, silence
    beeps = []
    monkeypatch.setattr(app, "_beep", lambda on, *tone: beeps.append(tone))
    monkeypatch.setattr(silence, "is_speechless_tap", lambda a, sr: False)
    monkeypatch.setattr(app, "_transcribe_audio", lambda *a: None)
    app._transcribe_and_inject([np.ones(16000, dtype=np.int16).tobytes()], 16000, True, False, False)
    assert beeps == [app.CANCEL_BEEP]


def test_the_log_says_your_words_steered_parakeet_but_not_which_word(caplog):
    """Which word Parakeet wrote is something the user said: the log keeps
    none of that unless log_transcripts is on."""
    import logging
    from tuxtalk import parakeet_engine
    caplog.set_level(logging.INFO, logger=parakeet_engine.logger.name)
    parakeet_engine._log_word("ClaudeClaw", 2.5, kept=True)
    logged = " ".join(r.message for r in caplog.records)
    assert "one of your words" in logged
    assert "ClaudeClaw" not in logged


# --- Latin only on the processor (script_filter.py) ---------------------------

# Pieces in onnx-asr's form; 5 is a Cyrillic "Ол" that Parakeet sometimes
# prefers to " also" on a short, distorted chunk.
LATIN_VOCAB = {0: " T", 1: "uck", 2: "ux", 3: " talk", 4: "Talk", 5: " Ол", 6: " also",
               7: "<unk>", 8: "<blk>"}
LATIN_BLANK = 8


class ScriptedAsr:
    """Stands in for onnx-asr's NemoConformerTdt: _decode scores the pieces
    *script(t, tokens)* names (everything else -20, the blank 0), and the
    TDT step is always one frame, as the duration head would say."""
    use_low_precision = False
    _blank_idx = LATIN_BLANK
    _max_tokens_per_step = 10
    _vocab = LATIN_VOCAB
    _vocab_size = len(LATIN_VOCAB)

    def __init__(self, script):
        self.script = script

    def _create_state(self):
        return ()

    def _decode(self, tokens, state, enc):
        logits = np.full(len(LATIN_VOCAB), -20.0, dtype=np.float32)
        logits[LATIN_BLANK] = 0.0
        for pid, score in self.script(int(enc[0]), tokens).items():
            logits[pid] = score
        return logits, 1, (tuple(tokens),)


def tuck_talk_also(t, tokens):
    """Speech heard as "Tuck talk" (TuxTalk two points behind) and then a
    word whose favourite is the Cyrillic piece, " also" one point behind."""
    last = LATIN_VOCAB[tokens[-1]] if tokens else None
    if t == 0:
        return {0: 5}
    if t == 1 and last == " T":
        return {1: 5, 2: 3}
    if t == 2 and last in ("uck", "ux"):
        return {3: 5, 4: 3}
    if t == 3:
        return {5: 6, 6: 5}
    return {}


def _latin_model(asr):
    from tuxtalk import parakeet_engine
    model = parakeet_engine.ParakeetModel.__new__(parakeet_engine.ParakeetModel)
    model._model = SimpleNamespace(asr=asr)
    return model


def _plain_decode(asr, frames=5):
    """onnx-asr's own greedy decoder on *asr*, as recognize() would run it."""
    from onnx_asr.asr import _AsrWithTransducerDecoding
    enc = np.arange(frames, dtype=np.float32).reshape(1, frames, 1)
    tokens, _ts, _lp = next(_AsrWithTransducerDecoding._decoding(asr, enc, np.array([frames])))
    return "".join(LATIN_VOCAB[i] for i in tokens).strip()


def test_latin_only_keeps_cyrillic_out_of_a_transcription():
    asr = ScriptedAsr(tuck_talk_also)
    assert _plain_decode(asr) == "Tuck talk Ол"
    _latin_model(asr).set_latin_only(True)
    assert _plain_decode(asr) == "Tuck talk also"


def test_latin_only_works_with_your_words():
    from tuxtalk import parakeet_bias
    asr = ScriptedAsr(tuck_talk_also)
    model = _latin_model(asr)
    model.set_latin_only(True)
    booster = parakeet_bias.WordBooster(LATIN_VOCAB, ["TuxTalk"], LATIN_BLANK)
    decode = parakeet_bias.boosted_tdt_decoding(asr, booster)
    enc = np.arange(5, dtype=np.float32).reshape(1, 5, 1)
    tokens, _ts, _lp = next(decode(enc, np.array([5])))
    assert "".join(LATIN_VOCAB[i] for i in tokens).strip() == "TuxTalk also"


def test_latin_only_off_gives_parakeet_its_own_decoder_back():
    asr = ScriptedAsr(tuck_talk_also)
    model = _latin_model(asr)
    model.set_latin_only(True)
    model.set_latin_only(False)
    assert "_decode" not in vars(asr)
    assert _plain_decode(asr) == "Tuck talk Ол"


def test_latin_only_asked_again_changes_nothing():
    asr = ScriptedAsr(tuck_talk_also)
    model = _latin_model(asr)
    model.set_latin_only(True)
    wrapped = asr._decode
    model.set_latin_only(True)
    assert asr._decode is wrapped


def test_latin_only_on_an_unexpected_onnx_asr_still_dictates():
    """A future onnx-asr without these internals must not break dictation."""
    class Odd:
        pass
    asr = Odd()
    _latin_model(asr).set_latin_only(True)
    assert "_decode" not in vars(asr)


def test_the_log_counts_the_steering_but_shows_no_text(caplog):
    import logging
    from tuxtalk import parakeet_engine
    asr = ScriptedAsr(tuck_talk_also)
    model = _latin_model(asr)
    model._model.recognize = lambda audio, sample_rate: _plain_decode(asr)
    model.set_latin_only(True)
    caplog.set_level(logging.INFO, logger=parakeet_engine.logger.name)
    assert model.recognize(np.zeros(16000, dtype=np.float32)) == "Tuck talk also"
    logged = [r.message for r in caplog.records if "steered" in r.message]
    assert logged == ["Parakeet: steered 1 steps away from Cyrillic/Greek"]
    assert "also" not in " ".join(logged)


@pytest.mark.parametrize("language,auto_languages,dictation_language,lang_env,expected", [
    (None, (), "de", "de_CH.UTF-8", [True]),             # a Swiss desktop, auto-detect
    ("ru", (), "de", "de_CH.UTF-8", [False]),            # Russian chosen
    (None, ("en", "ru"), "de", "de_CH.UTF-8", [False]),  # Russian among auto-detect's languages
    (None, (), "ru", "en_US.UTF-8", [False]),            # Russian picked at first run
    (None, (), "en", "ru_RU.UTF-8", [False]),            # a Russian desktop
])
def test_dictation_keeps_cyrillic_out_unless_it_could_be_wanted(
        monkeypatch, language, auto_languages, dictation_language, lang_env, expected):
    from tuxtalk import app, parakeet_engine

    seen = []

    class Model(parakeet_engine.ParakeetModel):
        def __init__(self):
            pass

        def set_latin_only(self, on):
            seen.append(on)

        def recognize(self, audio):
            return "Hello there."
    monkeypatch.setattr(app, "model", Model(), raising=False)  # set by main() normally
    monkeypatch.setattr(app, "_auto_languages", auto_languages)
    monkeypatch.setattr(app, "_dictation_language", dictation_language)
    for name in ("LC_ALL", "LC_MESSAGES", "LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", lang_env)
    app._transcribe_audio(np.zeros(16000, dtype=np.float32), language)
    assert seen == expected
