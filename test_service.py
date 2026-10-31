"""Contract tests isolate the model; real-model evidence is collected separately."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import app
import audit
import config
import detector
import scoring
import stylometry

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = patch.multiple(config, LOG_DIR=Path(self.tmp.name), AUDIT_LOG=Path(self.tmp.name)/"audit.jsonl")
        self.paths.start()
        self.client = app.app.test_client()
    def tearDown(self):
        self.paths.stop()
        self.tmp.cleanup()
    def test_submit_appeal_audit(self):
        with patch.object(detector, "model_signal", return_value=0.9):
            response = self.client.post("/submit", json={"text":"A sentence. Another sentence. A final sentence.", "creator_id":"writer"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for key in ("content_id", "guess", "confidence", "label", "model_score", "style_score"):
            self.assertIn(key, body)
        original = audit.read_entries()[0]
        for _ in range(10):
            result = self.client.post("/appeal", json={"content_id":body["content_id"], "reasoning":"I wrote this myself."})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.get_json()["status"], "under_review")
        entries = audit.entries_for(body["content_id"])
        self.assertEqual(entries[0], original)
        self.assertEqual(len(entries), 11)
        self.assertTrue(all(e["reasoning"] == "I wrote this myself." and e["creator_id"] == "writer" for e in entries[1:]))
        self.assertEqual(self.client.get("/log").get_json()["count"], 11)
    def test_bad_input_never_calls_detector(self):
        cases = [None, [], "string", {}, {"text":None}, {"text":12}, {"text":" "}, {"text":"ok"}, {"text":"ok", "creator_id":[]}, {"text":"ok", "creator_id":" "}, {"text":"ok", "creator_id":"a"*201}, {"text":"x"*20001,"creator_id":"writer"}]
        with patch.object(detector,"model_signal") as model:
            for payload in cases:
                with self.subTest(payload=str(payload)[:60]):
                    response = self.client.post("/submit", json=payload)
                    self.assertIn(response.status_code, (400,413))
                    self.assertTrue(response.is_json)
            response = self.client.post("/submit",data="{broken",content_type="application/json")
            self.assertEqual(response.status_code,400)
            model.assert_not_called()
        self.assertEqual(audit.read_entries(),[])
    def test_inclusive_text_limit(self):
        with patch.object(detector,"model_signal",return_value=0.5):
            self.assertEqual(self.client.post("/submit",json={"text":"x"*20000,"creator_id":"writer"}).status_code,200)
    def test_invalid_appeals(self):
        for payload in ([], {}, {"content_id":1,"reasoning":"why"}, {"content_id":"x","reasoning":" "}, {"content_id":"x","reasoning":"a"*5001}):
            self.assertEqual(self.client.post("/appeal",json=payload).status_code,400)
        for i in range(10):
            self.assertEqual(self.client.post("/appeal",json={"content_id":str(i),"reasoning":"why"}).status_code,404)
        self.assertEqual(audit.read_entries(),[])
    def test_detector_failure_has_no_decision(self):
        with patch.object(detector,"model_signal",side_effect=detector.DetectorUnavailable("offline")):
            self.assertEqual(self.client.post("/submit",json={"text":"Enough text for a score.","creator_id":"writer"}).status_code,503)
        self.assertEqual(audit.read_entries(),[])
    def test_all_label_boundaries(self):
        seen = set()
        for i in range(1001):
            score = i/1000
            guess,label = scoring.score_to_label(score)
            seen.add(guess)
            self.assertEqual(guess, "human" if score <= .2 else "ai" if score >= .8 else "unsure")
            self.assertTrue(label.endswith("."))
        self.assertEqual(seen,{"human","ai","unsure"})
        self.assertNotEqual(scoring.score_to_label(.51),scoring.score_to_label(.95))
        self.assertEqual(scoring.combine_signals(.9,.1),.65)
        for value in (-1,2,float("nan"),float("inf")):
            with self.assertRaises(ValueError): scoring.score_to_label(value)
    def test_style_measures_vary(self):
        self.assertEqual(stylometry.type_token_ratio("one one two"),2/3)
        self.assertEqual(stylometry.punctuation_density("Hi!"),1/3)
        self.assertEqual(stylometry.sentence_length_spread("One."),0)
        a = "The system works well and provides consistent results. "*5
        b = "Stop! The violet kettle slid beneath my grandmother's crooked kitchen chair, trailing sparks and a ribbon of steam. Why, though? I ran."
        self.assertNotEqual(stylometry.style_signal(a),stylometry.style_signal(b))
        for text in ("",a,b,"???", "One."):
            self.assertTrue(0 <= stylometry.style_signal(text) <= 1)

if __name__ == "__main__":
    unittest.main()
