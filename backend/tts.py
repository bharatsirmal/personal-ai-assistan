import os
import uuid
import wave

from piper import PiperVoice


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VOICE_PATH = os.path.join(
    BASE_DIR,
    "voices",
    "en_US-bryce-medium.onnx"
)

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "audio"
)


# ============================================================
# CREATE AUDIO DIRECTORY
# ============================================================

os.makedirs(
    AUDIO_DIR,
    exist_ok=True
)


# ============================================================
# LOAD PIPER VOICE
# ============================================================

_voice = None


def get_voice():
    """
    Load Piper voice once and reuse it.
    """

    global _voice

    if _voice is None:

        if not os.path.exists(VOICE_PATH):
            raise RuntimeError(
                f"Piper voice not found: {VOICE_PATH}"
            )

        print(
            "Loading Piper voice..."
        )

        _voice = PiperVoice.load(
            VOICE_PATH
        )

        print(
            "Piper voice loaded successfully."
        )

    return _voice


# ============================================================
# GENERATE VOICE
# ============================================================

def generate_voice(text: str) -> str:
    """
    Convert text into WAV audio using Piper.
    Returns the generated filename.
    """

    if not text or not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    voice = get_voice()

    # --------------------------------------------------------
    # Clean text for speech
    # --------------------------------------------------------

    clean_text = (
        text
        .replace("*", "")
        .replace("#", "")
        .replace("_", "")
        .replace("`", "")
        .replace("\n", " ")
        .strip()
    )

    # --------------------------------------------------------
    # Unique filename
    # --------------------------------------------------------

    filename = (
        f"bharat_ai_{uuid.uuid4().hex}.wav"
    )

    filepath = os.path.join(
        AUDIO_DIR,
        filename
    )

    # --------------------------------------------------------
    # Generate WAV
    # --------------------------------------------------------

    with wave.open(
        filepath,
        "wb"
    ) as wav_file:

        voice.synthesize_wav(
            clean_text,
            wav_file
        )

    print(
        f"Generated audio: {filename}"
    )

    return filename