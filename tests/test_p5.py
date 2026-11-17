"""Synthetic only; no match assets, public GT or run_all.py.

    python -m unittest discover -s tests -p test_p5.py -v
"""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import GAME, half_video_blocks
from mtah import contract, decoder, loader, p2_adapter, ref_decoder, tags, team


def evidence(ball, noball=None, ep=team.EP_LLR_PRIOR):
    ball = np.asarray(ball, dtype=float)
    return dict(llr_ball=ball, llr_noball=np.zeros_like(ball) if noball is None else noball,
                ep_llr=np.array(ep))


def phase(a, b, code="BUILD UP", side="H"):
    return dict(start_frame=a, end_frame=b, code=code, team=side)


class DecoderTests(unittest.TestCase):
    def test_default_is_exactly_p4_with_one_dead_mask_call(self):
        rng = np.random.RandomState(0)
        half = np.r_[np.zeros(10), np.ones(180), np.zeros(20), 2 * np.ones(180),
                     np.zeros(10)].astype(np.int8)
        pt = evidence(rng.normal(size=len(half)), rng.normal(size=len(half)), ep=2.82)
        pd = np.full(len(half), .2)
        pd[80:120] = .9
        expected = ref_decoder.decode(pt["llr_ball"] + pt["llr_noball"],
                                      ref_decoder.dead_mask(pd, half), half,
                                      team.default_switch_cost(2.82))
        with patch.object(ref_decoder, "dead_mask", wraps=ref_decoder.dead_mask) as mask:
            labels, info = decoder.decode_arrays(pt, pd, half, return_info=True)
        np.testing.assert_array_equal(labels, expected)
        mask.assert_called_once()
        self.assertEqual(info["switch_cost"], team.default_switch_cost(2.82))
        self.assertFalse(info["tag_prior"])

    def test_noball_is_added_once_and_p_home_is_not_reused(self):
        pt = evidence(np.full(80, -.3), np.full(80, .1))
        pt.update(p_home=np.ones(80), conf=np.ones(80))
        labels = decoder.decode_arrays(pt, np.zeros(80), np.ones(80, dtype=int))
        self.assertEqual(set(labels), {"A"})

    def test_no_extra_smoothing_or_minimum_duration(self):
        llr = np.full(101, 100.0)
        llr[50] = -1000.0
        labels = decoder.decode_arrays(evidence(llr), np.zeros(101), np.ones(101, dtype=int))
        self.assertEqual(labels[50], "A")
        self.assertEqual(np.sum(labels == "A"), 1)

    def test_dead_mask_never_spills_between_halves(self):
        half = np.r_[np.ones(30), 2 * np.ones(30)].astype(int)
        labels = decoder.decode_arrays(evidence(np.ones(60)), np.r_[np.ones(30), np.zeros(30)], half)
        np.testing.assert_array_equal(labels[:30], "D")
        np.testing.assert_array_equal(labels[30:], "H")

    def test_empty_runs_use_same_half_neighbours(self):
        half = np.r_[np.ones(180), np.zeros(15), 2 * np.ones(180)].astype(int)
        llr = np.zeros(len(half))
        llr[60:70] = -.5
        llr[280:290] = .5
        pd = np.zeros(len(half))
        pd[90:120] = 1
        pd[235:265] = 1
        labels = decoder.decode_arrays(evidence(llr), pd, half)
        self.assertEqual(set(labels[:90]), {"A"})
        self.assertEqual(set(labels[120:180]), {"A"})
        self.assertEqual(set(labels[180:195]), {"X"})
        self.assertEqual(set(labels[195:235]), {"H"})
        self.assertEqual(set(labels[265:]), {"H"})

    def test_entire_empty_half_stays_unresolved_without_borrowing(self):
        half = np.r_[np.ones(40), np.zeros(10), 2 * np.ones(40)].astype(int)
        llr = np.r_[np.full(40, -.3), np.zeros(50)]
        labels, info = decoder.decode_arrays(evidence(llr), np.zeros(90), half, return_info=True)
        self.assertEqual(set(labels[:40]), {"A"})
        self.assertEqual(set(labels[40:]), {"X"})
        self.assertEqual(info["unresolved"], 1)

    def test_no_graded_frames_and_empty_arrays(self):
        for n in (0, 20):
            labels = decoder.decode_arrays(evidence(np.zeros(n)), np.ones(n), np.zeros(n, dtype=int))
            self.assertEqual(labels.shape, (n,))
            self.assertTrue(np.all(labels == "X"))

    def test_bad_evidence_masks_and_scalars_fail(self):
        half, pd = np.ones(4, dtype=int), np.zeros(4)
        for bad in ([0, 0], [0, np.nan, 0, 0], [0, np.inf, 0, 0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decoder.decode_arrays(evidence(bad), pd, half)
        for bad in (np.nan, np.inf, -1, 0, [2.0]):
            with self.subTest(ep=bad), self.assertRaises(ValueError):
                decoder.decode_arrays(evidence(np.ones(4), ep=bad), pd, half)
        for bad in ([1, 0, 1, 0], [2, 2, 1, 1], [1, 3, 1, 1], [1., 1., 1., 1.]):
            with self.subTest(half=bad), self.assertRaises(ValueError):
                decoder.decode_arrays(evidence(np.ones(4)), pd, bad)
        for bad in ([0, 0], [0, np.nan, 0, 0], [-.1, 0, 0, 0], [1.1, 0, 0, 0]):
            with self.subTest(pd=bad), self.assertRaises(ValueError):
                decoder.decode_arrays(evidence(np.ones(4)), bad, half)
        with self.assertRaises(KeyError):
            decoder.decode_arrays({"llr_ball": np.ones(4)}, pd, half)

    def test_missing_ep_uses_p4_prior(self):
        pt = evidence(np.ones(4))
        del pt["ep_llr"]
        _, info = decoder.decode_arrays(pt, np.zeros(4), np.ones(4, dtype=int), return_info=True)
        self.assertEqual(info["switch_cost"], team.default_switch_cost(team.EP_LLR_PRIOR))

    def test_default_does_not_even_load_tag_prior(self):
        requested = []

        def load(name):
            requested.append(name)
            return {"p_team": evidence(np.full(40, -.005)), "p_dead": {"p_dead": np.zeros(40)},
                    "tag_prior": {"llr_tag": np.full(40, 1.0 / 40)}}[name]

        with patch.object(decoder, "load_half", return_value=np.ones(40, dtype=int)), \
                patch.object(contract, "load", side_effect=load):
            self.assertEqual(set(decoder.decode()), {"A"})
            self.assertEqual(requested, ["p_team", "p_dead"])
            self.assertEqual(set(decoder.decode(use_tag_prior=True)), {"H"})
        self.assertIn("tag_prior", requested)

    def test_missing_cache_does_not_build_loader(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(loader, "FRAMES_PATH", str(Path(directory) / "missing.npz")), \
                patch.object(loader, "build") as build:
            with self.assertRaises(FileNotFoundError):
                decoder.decode()
            build.assert_not_called()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, value in (("N_FRAMES", 4), ("DERIVED_DIR", str(self.root))):
            patcher = patch.object(contract, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.source = self.root / "p_dead.csv"
        self.source.write_bytes(b"frame,p_dead\n0,0.0\n1,0.3\n2,0.999\n3,1.0\n")

    def test_roundtrip_provenance_source_preservation_and_idempotence(self):
        before = self.source.read_bytes()
        out = Path(p2_adapter.adapt(self.source))
        result = contract.load("p_dead")
        np.testing.assert_array_equal(result["p_dead"], np.array([0, .3, .999, 1], dtype=np.float32))
        self.assertEqual(str(result["source_sha256"]), hashlib.sha256(before).hexdigest())
        self.assertEqual(self.source.read_bytes(), before)
        stamp, content = out.stat().st_mtime_ns, out.read_bytes()
        p2_adapter.adapt(self.source)
        self.assertEqual(out.stat().st_mtime_ns, stamp)
        self.assertEqual(out.read_bytes(), content)

    def test_existing_npz_needs_explicit_replacement(self):
        output = Path(contract.save("p_dead", p_dead=np.zeros(4)))
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            p2_adapter.adapt(self.source)
        self.assertEqual(output.read_bytes(), before)
        p2_adapter.adapt(self.source, replace=True)
        self.assertGreater(contract.load("p_dead")["p_dead"][2], .99)

    def test_invalid_csv_never_overwrites_npz_even_with_replace(self):
        output = Path(contract.save("p_dead", p_dead=np.zeros(4)))
        before = output.read_bytes()
        valid = self.source.read_text()
        bad_cases = [valid.replace("frame,p_dead", "p_dead,frame"),
                     valid.replace("frame,p_dead", "frame,p_dead,extra"),
                     valid.replace("1,0.3", "0,0.3"), valid.replace("1,0.3", "2,0.3"),
                     valid.replace("0,0.0", "1,0.0"), valid.replace("0,0.0", "0.0,0.0"),
                     valid.replace("1,0.3", '1,"0.3'),
                     "\n".join(valid.splitlines()[:-1]) + "\n", valid + "4,0.5\n", valid + "\n"]
        bad_cases += [valid.replace("1,0.3", "1," + value)
                      for value in ("nan", "inf", "-inf", "-0.1", "1.1", "", "word")]
        for content in bad_cases:
            with self.subTest(case=content.splitlines()[0]):
                self.source.write_text(content)
                original = self.source.read_bytes()
                with self.assertRaises((ValueError, csv.Error)):
                    p2_adapter.adapt(self.source, replace=True)
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(self.source.read_bytes(), original)

    def test_crlf_bom_and_endpoint_probabilities(self):
        self.source.write_bytes(b"\xef\xbb\xbfframe,p_dead\r\n0,0\r\n1,1\r\n2,0\r\n3,1\r\n")
        p2_adapter.adapt(self.source)
        np.testing.assert_array_equal(contract.load("p_dead")["p_dead"], [0, 1, 0, 1])

    def test_source_cannot_alias_destination(self):
        output = Path(contract.path("p_dead"))
        output.write_bytes(self.source.read_bytes())
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            p2_adapter.adapt(output, replace=True)
        self.assertEqual(output.read_bytes(), before)


class PriorTests(unittest.TestCase):
    def test_phase_budget_does_not_grow_with_duration(self):
        half = np.ones(5000, dtype=int)
        for length in (1, 17, 125, 4999):
            with self.subTest(length=length):
                result = tags.build_prior([phase(0, length)], half)
                self.assertLessEqual(np.abs(result["llr_tag"]).sum(), 1 + 1e-12)
                self.assertAlmostEqual(result["llr_tag"].sum(), 1.0)
                self.assertTrue(np.all(np.abs(result["phase_budget"]) <= 1))
                self.assertEqual(result["phase_frames"][0], length)

    def test_attack_and_defence_have_opposite_meaning(self):
        half = np.ones(20, dtype=int)
        for code in tags.ATTACK:
            self.assertGreater(tags.build_prior([phase(2, 10, code)], half)["llr_tag"].sum(), 0)
        for code in tags.DEFENCE:
            self.assertLess(tags.build_prior([phase(2, 10, code)], half)["llr_tag"].sum(), 0)
        self.assertLess(tags.build_prior([phase(2, 10, side="A")], half)["llr_tag"].sum(), 0)

    def test_overlapping_phases_keep_individual_budgets_and_cancel(self):
        half = np.ones(30, dtype=int)
        attack = phase(3, 20)
        defence = phase(3, 20, "DEFENSIVE TRANSITION")
        cancelled = tags.build_prior([attack, defence], half)
        np.testing.assert_allclose(cancelled["llr_tag"], 0, atol=1e-15)
        combined = tags.build_prior([attack, attack], half)
        self.assertAlmostEqual(combined["llr_tag"].sum(), 2)
        np.testing.assert_array_equal(combined["phase_budget"], [1, 1])

    def test_clipping_half_open_boundaries_and_no_halftime_evidence(self):
        half = np.r_[np.zeros(5), np.ones(10), np.zeros(10), 2 * np.ones(10), np.zeros(5)].astype(int)
        result = tags.build_prior([phase(-100, 100)], half)
        self.assertAlmostEqual(result["llr_tag"].sum(), 1)
        np.testing.assert_array_equal(result["llr_tag"][half == 0], 0)
        bounded = tags.build_prior([phase(5, 10)], half)
        self.assertGreater(bounded["llr_tag"][9], 0)
        self.assertEqual(bounded["llr_tag"][10], 0)
        self.assertTrue(np.all(bounded["llr_tag"][10:] == 0))

    def test_na_unknown_and_set_pieces_do_not_create_team_evidence(self):
        half = np.ones(30, dtype=int)
        events = [phase(2, 20, side="N/A"), phase(2, 20, "UNKNOWN"),
                  phase(5, 10, "SET PIECES", "N/A")]
        result = tags.build_prior(events, half)
        np.testing.assert_array_equal(result["prior_home"], .5)
        np.testing.assert_array_equal(result["prior_away"], .5)
        self.assertEqual(result["restart"].sum(), 5)
        self.assertEqual(result["restart"][10], 0)

    def test_invalid_caps_and_intervals_fail(self):
        half = np.ones(10, dtype=int)
        for cap in (-1, 1.01, np.nan, np.inf):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                tags.build_prior([phase(0, 10)], half, cap)
        for a, b in ((5, 3), (0.5, 3), (False, 3)):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                tags.build_prior([phase(a, b)], half)
        np.testing.assert_array_equal(tags.build_prior([], half)["llr_tag"], 0)

    def test_events_csv_build_and_contract_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(contract, "N_FRAMES", 20), \
                patch.object(contract, "DERIVED_DIR", directory), \
                patch.object(loader, "FRAMES_PATH", str(Path(directory) / "frames.npz")):
            np.savez(loader.FRAMES_PATH, half=np.ones(20, dtype=int))
            events = Path(directory) / "events.csv"
            events.write_text("start_frame,end_frame,code,team,side,type,half\n0,20,BUILD UP,H,,,1st Half\n")
            before = events.read_bytes()
            tags.build(events)
            result = contract.load("tag_prior")
            self.assertAlmostEqual(result["llr_tag"].sum(), 1)
            self.assertEqual(result["llr_tag"].dtype, np.float64)
            self.assertEqual(events.read_bytes(), before)
            events.write_text("start_frame,end_frame,code,team\n0.5,20,BUILD UP,H\n")
            with self.assertRaises(ValueError):
                tags.read_events(events)


class SubmissionTests(unittest.TestCase):
    def test_canonical_format_and_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(contract, "N_FRAMES", 4):
            path = Path(directory) / "submission.csv"
            decoder.write_submission(path, list("HADX"))
            self.assertEqual(path.read_bytes(), b"frame,label\n0,H\n1,A\n2,D\n3,X\n")
            decoder.validate_file(path)
            with self.assertRaises(FileExistsError):
                decoder.write_submission(path, list("AAAA"))
            decoder.write_submission(path, list("AAAA"), overwrite=True)
            self.assertTrue(path.read_bytes().endswith(b"3,A\n"))

    def test_bad_labels_do_not_truncate_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(contract, "N_FRAMES", 4):
            path = Path(directory) / "submission.csv"
            decoder.write_submission(path, list("HADX"))
            before = path.read_bytes()
            for labels in (["H", "A", "D"], ["HA", "A", "D", "X"], [None, "H", "D", "X"],
                           ["", "H", "D", "X"], [["H", "A", "D", "X"]]):
                with self.subTest(labels=labels), self.assertRaises(ValueError):
                    decoder.write_submission(path, labels, overwrite=True)
                self.assertEqual(path.read_bytes(), before)

    def test_p2_csv_cannot_be_replaced_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(contract, "N_FRAMES", 4):
            path = Path(directory) / "p_dead.csv"
            path.write_bytes(b"frame,p_dead\n0,0.3\n1,0.3\n2,0.3\n3,0.3\n")
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                decoder.write_submission(path, list("HADX"), overwrite=True)
            self.assertEqual(path.read_bytes(), before)

    def test_official_format_checks_reject_bad_frames_blank_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(contract, "N_FRAMES", 4):
            path = Path(directory) / "submission.csv"
            good = "frame,label\n0,H\n1,A\n2,D\n3,X\n"
            for content in (good.replace("1,A", "0,A"), good.replace("1,A", "1,"),
                            good.replace("1,A", "1,Z"), "\n".join(good.splitlines()[:-1])):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    decoder.validate_file(path)


class CommandTests(unittest.TestCase):
    def test_full_length_cli_is_deterministic_valid_and_preserves_p2_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            half = np.zeros(contract.N_FRAMES, dtype=np.int8)
            for h, start, end in half_video_blocks(GAME):
                half[start:end + 1] = h
            llr = np.where(half == 1, .3, -.3)
            np.savez(root / "frames.npz", half=half)
            with patch.object(contract, "DERIVED_DIR", directory):
                contract.save("p_team", p_home=np.full(len(half), .5), p_away=np.full(len(half), .5),
                              conf=np.zeros(len(half)), llr_ball=llr,
                              llr_noball=np.zeros(len(half)), ep_llr=np.array(2.82))
            source = root / "p_dead.csv"
            with source.open("w", newline="\n") as handle:
                handle.write("frame,p_dead\n")
                handle.writelines("{},0.2\n".format(i) for i in range(len(half)))
            inputs = [source, root / "frames.npz", root / "p_team.npz"]
            before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
            environment = dict(os.environ, MTAH_DERIVED_DIR=directory)
            outputs = [root / "one.csv", root / "two.csv"]
            for output in outputs:
                result = subprocess.run([sys.executable, "-m", "mtah.decoder", "--p-dead-csv",
                                         str(source), "--output", str(output)], cwd=str(ROOT),
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("tag prior OFF", result.stdout)
                self.assertFalse((root / "tag_prior.npz").exists())
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            for path, (content, stamp) in zip(inputs, before):
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mtime_ns, stamp)
            result = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"),
                                     str(outputs[0])], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("199319 rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
