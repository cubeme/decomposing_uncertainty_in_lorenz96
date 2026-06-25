import sys
from pathlib import Path

# resolve project root (two levels up from this file)
PROJECT_ROOT = Path.cwd().resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# add src/ and notebooks/ to sys.path
for p in (SRC_DIR, NOTEBOOKS_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import numpy as np  # noqa: E402
from absl import app, flags, logging  # noqa: E402
from evaluation.scripts.eval_correlation_long_truth import (  # noqa: E402
    compute_correlation_long_streamed,
    compute_correlation_truth_streamed,  # noqa: E402
)

# -------------------------------- FLAGS ----------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir",
    None,
    "Path to one model directory (e.g., results/ensemble_l96_...).",
)

flags.DEFINE_string("out_dir", None, "Output directory for metrics (npz + parquet).")

flags.DEFINE_integer("max_lag", None, "Maximum lag to compute correlations for.")

flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("out_dir")

FLAGS = flags.FLAGS
FLAGS.log_dir = str(PROJECT_ROOT / "notebooks" / "evaluation" / "scripts" / "logs")


# -------------------------------- Main ----------------------------------------------


def main(argv):
    Path(FLAGS.log_dir).mkdir(parents=True, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(program_name="run")

    logging.info("Logging to %s", FLAGS.log_dir)
    # absl app passes argv; we don't need extra positional args
    if len(argv) > 1:
        raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")

    data_dir = Path(FLAGS.data_dir).resolve()
    out_dir = Path(FLAGS.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting rank histogram computation for model %s.", data_dir.name)

    if "ensemble_l96" in data_dir.name:
        logging.info("Computing correlation for L96 truth.")
        type = "truth"
        df, t = compute_correlation_truth_streamed(data_dir, max_lag=FLAGS.max_lag)
    else:
        logging.info(
            "Computing correlation for reduced model in long setting.",
        )
        type = "long"
        df, t = compute_correlation_long_streamed(data_dir, max_lag=FLAGS.max_lag)

    logging.info("Saving results to %s", out_dir)
    pickle_path = out_dir / f"{data_dir.name}_correlation_{type}.pkl"
    df.to_pickle(pickle_path)

    time_out = out_dir / f"{data_dir.name}_correlation_{type}_time.npy"
    np.save(time_out, t)

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
