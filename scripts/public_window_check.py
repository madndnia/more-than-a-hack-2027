"""Person 1 ONLY. Score one submission on the public 5 minutes -- as a pipeline sanity check.

    python scripts/public_window_check.py submissions/TEAMNAME.csv

This answers "are frames aligned and do labels mean what we think", nothing more. 17 possessions
cannot rank two versions (EVALUATION.md); do not tune anything on this number and do not post it
in the team chat as a target. Compare versions with mtah.diagnostics instead.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    gt = pm.label_codes(pm.read_label_csv(GT))
    table = pm.read_label_csv(sys.argv[1])
    reason = pm.validate_submission(table, len(gt))
    if reason:
        sys.exit("FAIL " + reason)
    r = pm.evaluate(gt, pm.label_codes(table))
    print("public 5 min  macroF1 {:.4f}  spellF1 {:.4f}  poss {} (gt {})  dur {:.1f}s (gt {:.1f}s)"
          .format(r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"],
                  r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"]))
    print("baseline_smooth50 on the same window: macroF1 0.6207  spellF1 0.3256")
    print("If you are far BELOW the baseline here, suspect a bug (alignment, H/A swapped), "
          "not the model.")
