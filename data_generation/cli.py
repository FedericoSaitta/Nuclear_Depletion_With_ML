"""The command line the sweep entry points share.

**The config describes the simulation; the flags describe the machine.** The
machine half is therefore the same flags whichever simulation is being run, and
this is the one place they are spelled out — `datagen.py` and
`quarter_datagen.py` used to carry identical copies, differing only in their
defaults.

`zoom_datagen.py` keeps its own parser on purpose: it is single-process and
continues an existing run, so it has no `-n/-c` and a required `--daily-results`
instead.

Nothing here imports OpenMC, which is what lets an entry point build its parser
at the top of the file and still have `--help` work on a machine with no OpenMC
build — and, more to the point, parse `-t` before `OMP_NUM_THREADS` has to be
set.
"""

import argparse


def sweep_parser(description, config_example, default_runs, default_cores):
    """Parser for an entry point that runs rounds of parallel workers.

    *default_runs* and *default_cores* are the only things that vary: a sampled
    sweep wants many short workers, a measured single history wants one worker
    on every core it has.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, help=f"simulation config, e.g. {config_example}"
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=default_runs,
        help="rounds of parallel workers",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=default_cores,
        help="parallel worker processes per round",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=1, help="OpenMP threads per worker"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="master seed (None = random)"
    )
    parser.add_argument(
        "--out", default=None, help="output directory (default: data_generation/data)"
    )
    return parser
