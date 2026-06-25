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
from evaluation.scripts.eval_metrics_mix import (  # noqa: E402
    compute_metrics_mix_streamed,  # noqa: E402
)

# -------------------------------- FLAGS ----------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir",
    None,
    "Path to one model directory (e.g., results/ensemble_l96_...).",
)
flags.DEFINE_string(
    "l96_dir",
    None,
    "Path to L96 truth directory.",
)
flags.DEFINE_string("out_dir", None, "Output directory for metrics (npz + parquet).")

flags.DEFINE_integer("n_init_states", 300, "N_init")
flags.DEFINE_integer("n_ens_members", 20, "N_ens")


flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("l96_dir")
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

    model_dir = Path(FLAGS.model_dir).resolve()
    l96_dir = Path(FLAGS.l96_dir).resolve()
    out_dir = Path(FLAGS.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    N = FLAGS.n_init_states
    M = FLAGS.n_ens_members

    logging.info(
        "Starting metrics computation for model %s in mixed setting with N=%d, M=%d.",
        model_dir.name,
        N,
        M,
    )

    df, t = compute_metrics_mix_streamed(model_dir, l96_dir, N, M)

    logging.info("Saving results to %s", out_dir)
    pickle_path = out_dir / f"{model_dir.name}_metrics_mix.pkl"
    df.to_pickle(pickle_path)

    time_out = out_dir / f"{model_dir.name}_metrics_mix_time.npy"
    np.save(time_out, t)

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
