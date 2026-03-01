from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decoder import decode_with_lexicon, extract_phrase_from_filename, normalize_phrase, phrase_letters


class TestDecoder(unittest.TestCase):
    def test_phrase_normalization(self):
        self.assertEqual(normalize_phrase("help me"), "HELP ME")
        self.assertEqual(phrase_letters("Help me!"), "HELPME")

    def test_extract_phrase_from_filename(self):
        name = "AB，help me，1组-0205183330(2).zip"
        self.assertEqual(extract_phrase_from_filename(name), "HELP ME")

    def test_lexicon_decode_prefers_word(self):
        # 4 segments, 26 classes with strong mass on H,E,L,P respectively.
        probs = np.full((4, 26), 1e-6, dtype=np.float32)
        for t, ch in enumerate("HELP"):
            probs[t, ord(ch) - ord("A")] = 0.9
        # Add noise making raw argmax still HELP; decoder should keep HELP.
        probs = probs / probs.sum(axis=1, keepdims=True)
        out = decode_with_lexicon(probs, raw_letters="HELP", lexicon=["MOVE", "HELP"])
        self.assertIn(out.decoded_letters, ("HELP",))


if __name__ == "__main__":
    unittest.main()

