#!/usr/bin/env python

"""Distributes `run-sim-infer.py` on a SLURM cluster.

This Python script includes code to run a simulation routine, to
accept arguments from the command line to parameterize this function,
and to distribute job submissions of this script to SLURM over
hundreds of combination of parameter settings.

This script sets up a total of 6400 jobs across different parameter
combinations, each of which takes a few hours to run, so it is a
good idea to use HPC to run this.

Example
-------
>>> python run-sim-loci-inference-distributed.py \
>>>   --neff 10000 100000 \
>>>   --ctime 0.1 \
>>>   --nloci 100 250 500 1000 2500 5000 10000 \
>>>   --nsites 2000 10000 \
>>>   --nreps 10 \
>>>   --mut 5e-8 \
>>>   --recomb 0 5e-9 \
>>>   --node-heights 0.01 0.05 0.06 1 \
>>>   --ncores 12 \
>>>   --outdir /moto/eaton/users/de2356/recomb-response/data5 \
>>>   --account eaton \
>>>   --delay 1

Description
-----------
The job above implements a fixed imbalanced 5-taxon species tree
with divergence times described by the `node-heights` parameter.
It will simulate 10000 loci (the largest `nloci` param) of length
2000 and 10000. This will be done in one case with recombination,
and in another case without. These sims are also each repeated
for 10 replicates from random seeds. For each dataset true genealogies
will be saved, and also empirical gene trees will be inferred for
each locus using raxml-ng. A species tree will then be inferred
from each distribution of genealogies or gene trees.

Outputs
--------
>>> outdir/rep[1-10]-concat-subloci[100-10000].nwk
>>> outdir/rep[1-10]-astral-genealogy-subloci[100-10000].nwk
>>> outdir/rep[1-10]-astral-genetree-subloci[100-10000].nwk
"""

from typing import List, Dict, Iterator, Any, Tuple
import sys
import time
import argparse
import shutil
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen, STDOUT, PIPE
from loguru import logger
import numpy as np
import pandas as pd
import ipcoal

logger = logger.bind(name="ipcoal")
ROOT = str(Path(__file__).parent)
SBATCH = """\
#!/bin/sh

#SBATCH --account={account}
#SBATCH --job-name={jobname}
#SBATCH --output={outpath}.out
#SBATCH --error={outpath}.err
#SBATCH --time=11:59:00
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={nthreads}
#SBATCH --mem=12G

# run the command to write and submit a shell script
{python} {root}/run-sim-infer.py \
  --neff {neff} \
  --ctime {ctime} \
  --mut {mut} \
  --recomb {recomb} \
  --nsites {nsites} \
  --nloci {nloci} \
  --phased {phased} \
  --rep {rep} \
  --seed {seed} \
  --outdir {outdir} \
  --ncores {ncores} \
  --nthreads {nthreads} \
  --node-heights {node_heights} \
  --raxml-bin {raxml_bin} \
  --astral-bin {astral_bin}
"""


@dataclass
class SlurmDistribute:
    # user params with one or more values
    neff: List[int]
    ctime: List[int]
    recomb: List[float]
    mut: List[float]
    nsites: List[int]
    nloci: List[int]

    # individual params
    nreps: int
    seed: int
    ncores: int
    nthreads: int
    phased: bool
    outdir: Path
    account: str
    node_heights: List[float]

    # params with defaults, or to be filled
    raxml_bin: Path = None
    astral_bin: Path = None
    delay: float = 0.1

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        bindir = Path(sys.prefix) / "bin"
        if self.astral_bin is None:
            self.astral_bin = bindir / "astral.5.7.1.jar"
        else:
            self.astral_bin = Path(self.astral_bin)
        assert self.astral_bin.exists(), f"cannot find {self.astral_bin}. Use conda instructions."

        if self.raxml_bin is None:
            self.raxml_bin = bindir / "raxml-ng"
        else:
            self.raxml_bin = Path(self.raxml_bin)
        assert self.raxml_bin.exists(), f"cannot find {self.raxml_bin}. Use conda instructions."

    def iter_jobs(self) -> Iterator[Tuple[str, List[Any]]]:
        """Yield Tuples iterating over parameters combinations."""
        combs = product(self.nsites, self.ctime, self.recomb, self.mut, self.neff)
        for nsi, cti, rec, mut, nef in combs:
            # basename of the params used across a set of replicates.
            params_basename = (
                f"neff{int(nef)}_"
                f"ctime{cti}_"
                f"recomb{str(rec)}_"
                f"mut{str(mut)}_"
                f"nloci{max(self.nloci)}_"
                f"nsites{nsi}"
            )
            yield params_basename, [nsi, cti, rec, mut, nef]

    def iter_params(self) -> Iterator[Dict[str, Any]]:
        """Yield Dicts with all params for each job replicate."""
        # all runs use the same set of seeds across replicates
        seeds = np.random.default_rng(self.seed).integers(1e12, size=self.nreps)

        # iterate over jobs to be submitted
        for params_basename, [nsi, cti, rec, mut, nef] in self.iter_jobs():

            # run this set for nreplicate times.
            for rep in range(self.nreps):

                # get name of this job
                jobname = f"res_{params_basename}_rep{rep}"
                outpath = self.outdir / jobname # for .sh, .err, .out files

                # submit job to run...
                kwargs = dict(
                    account=self.account,
                    jobname=jobname,
                    outpath=outpath,
                    ntasks=int(self.ncores * self.nthreads),
                    ncores=self.ncores,
                    nthreads=self.nthreads,
                    root=ROOT,
                    neff=nef,
                    ctime=cti,
                    mut=mut,
                    recomb=rec,
                    nsites=nsi,
                    nloci=" ".join([str(i) for i in self.nloci]),
                    rep=rep,
                    seed=seeds[rep],
                    phased=self.phased,                    
                    outdir=self.outdir,
                    node_heights=" ".join([str(i) for i in self.node_heights]),
                    raxml_bin=self.raxml_bin,
                    astral_bin=self.astral_bin,
                    python=Path(sys.prefix) / 'bin' / 'python',
                )
                yield kwargs

    def iter_slurm_scripts(self) -> Iterator[Tuple[str, str]]:
        """Yield SLURM scripts (bash w/ #HEADER) for all job params."""
        for params in self.iter_params():
            yield params['jobname'], SBATCH.format(**params)

    def submit_subprocess(self, name: str, script: str, cmd: str="sbatch") -> None:
        """Start sh script with bash or sbatch."""
        # b/c the params string name has a '.' in it for decimal ctime.
        tmpfile = self.outdir / f"job_{name}.sh"
        with open(tmpfile, 'w', encoding='utf-8') as out:
            out.write(script)

        # submit job to bash or SLURM job manager
        cmd = [cmd, str(tmpfile)]
        with Popen(cmd, stdout=PIPE, stderr=STDOUT) as proc:
            out, _ = proc.communicate()
        if proc.returncode:
            logger.error(f"{out.decode()}")
        tmpfile.unlink()

    def _count_njobs(self) -> int:
        """Return number of jobs."""
        nlen = len(self.neff)
        rlen = len(self.recomb)
        clen = len(self.ctime)
        slen = len(self.nsites)
        mlen = len(self.mut)
        ilen = self.nreps
        njobs = nlen * rlen * clen * slen * ilen * mlen
        return njobs

    def run(self, cmd: str="sbatch", resume: bool=False, force: bool=False) -> None:
        """Calls submit_subprocess() on jobs in iter_slurm_script().

        This command also enforces the resume/force options to continue
        or restart a set of jobs, and ensures the outdir exists.
        """
        # if outdir exists an error is raised unless the user specified
        # either resume or force. The former will resume the run, the
        # latter will remove any files and restart the run.
        if self.outdir.exists():
            if not (resume or force):
                raise IOError(f"Output directory {self.outdir} exists.\n"
                    "Use --resume to continue running remaining jobs without existing results.\n"
                    "Or use --force to clear the output directory and restart."
                )
        if force and resume:
            raise ValueError("You must select --force or --resume, not both.")

        # outdir must exist.
        self.outdir.mkdir(exist_ok=True)

        # force removes everything inside the outdir.
        if force:
            for path in self.outdir.glob("*_neff*_ctime*_recomb*_nloci*"):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

        # iterate over all jobs to submit
        nfin = len(list(self.outdir.glob("_neff*_ctime*_recomb*_nloci*.csv")))
        njobs = self._count_njobs()

        # if all jobs are finished then run concatenation and end.
        if nfin == njobs:
            self.combine()
            sys.exit(0)

        # else run all jobs
        resuming = "Resuming." if resume else ""
        logger.info(f"Submitting {njobs - nfin} jobs. {resuming}")
        for name, script in self.iter_slurm_scripts():

            # skip if results exist for this rep
            resfile = self.outdir / f"{name}.csv"
            # logger.info(f"{resfile.exists()}, {resfile}")
            if resfile.exists():
                logger.info(f"skipping {name}")
                continue

            # submit job to run and remove .sh file when done.
            logger.info(f"starting job {name}")
            self.submit_subprocess(name, script, cmd)

            # if no exceptions were raised then rm .err and .out files
            logfiles = [self.outdir / f"{name}.err", self.outdir / f"{name}.out"]
            for logf in logfiles:
                if logf.exists():
                    # if not logfile.stat().st_size:
                    logf.unlink()

            # use short delay between job submissions to be nice.
            time.sleep(self.delay)

    def combine(self) -> None:
        """Concatenate all CSVs into a one large file."""
        iter_csvs = self.outdir.glob("*.csv")
        iter_dfs = (pd.read_csv(i, index_col=0) for i in iter_csvs)
        data = pd.concat(iter_dfs, ignore_index=True)
        outfile = self.outdir / "concat.csv"
        data.to_csv(outfile)
        nfiles = data.shape[0]
        logger.info(f"All jobs finished. Wrote {nfiles} results to concat file: {outfile}")

def distributed_command_line_parser():
    """Parse command line arguments and return.

    Example
    -------
    >>> python run-sim-loci-inference-distributed.py  \
    >>>     --ncores 2 \
    >>>     --nreps 100 \
    >>>     --nsites 2000 10000 \
    >>>     --neff 1e4 1e5 \
    >>>     --ctimes 0.1 0.2 0.3 0.4 0.5 0.75 1.0 1.25 \
    >>>     --mut 5e-8 \
    >>>     --recomb 0 5e-9 \
    >>>     --node-heights 0.01 0.05 0.06 1 \
    >>>     --outdir /scratch/recomb/ \
    >>>     --account eaton \
    >>>     --phased True
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--neff', type=int, default=[10000, 100000], nargs="*", help='Effective population sizes')
    parser.add_argument(
        '--ctime', type=float, default=[0.1, 1.5], nargs="*", help='Root species tree height in coalescent units')
    parser.add_argument(
        '--recomb', type=float, default=[0, 5e-9], nargs="*", help='Recombination rates.')
    parser.add_argument(
        '--mut', type=float, default=[5e-8], nargs="*", help='Mutation rates.')
    parser.add_argument(
        '--node-heights', type=float, default=[0.05, 0.055, 0.06, 1], nargs=4, help='Internal relative node heights')
    parser.add_argument(
        '--nsites', type=int, default=[2000], nargs="*", help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, default=[20000], nargs="*", help='number of independent simulated loci.')
    parser.add_argument(
        '--nreps', type=int, default=10, help='number replicate per param setting.')
    parser.add_argument(
        '--outdir', type=Path, default=Path("/tmp/test"), help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--account', type=str, default="free", help='Account name for SLURM job submission')
    parser.add_argument(
        '--ncores', type=int, default=3, help='Number of cores per task')
    parser.add_argument(
        '--nthreads', type=int, default=4, help='Number of threads per task')
    parser.add_argument(
        '--resume', action='store_true', help='Resume an interrupted run with some existing results.')
    parser.add_argument(
        '--force', action='store_true', help='Restart an run overwriting any existing results.')
    parser.add_argument(
        '--delay', type=float, default=0.5, help='Number of seconds delay between SLURM job submissions.')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random number generator seed.')
    parser.add_argument(
        '--phased', type=bool, default=True, help='simulate 5 phased haplotypes (vs 5 unphased diplotypes from 10 simulated haplotypes).')
    parser.add_argument(
        '--cmd', type=str, default="bash", help="Execute 'bash' for local or 'sbatch' for SLURM.")
    parser.add_argument(
        '--log-level', type=str, default="INFO", help="logging level: DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument(
        '--astral-bin', type=Path, default=None, help="Full path to astral binary.")
    parser.add_argument(
        '--raxml-bin', type=Path, default=None, help="Full path to raxml-ng binary.")
    return parser.parse_args()

def main():
    """Command line utility to accept arguments.

    """
    args = distributed_command_line_parser()
    kwargs = vars(args)
    resume = kwargs.pop("resume")
    force = kwargs.pop("force")
    cmd = kwargs.pop("cmd")
    log_level = kwargs.pop("log_level")
    ipcoal.set_log_level(log_level)
    tool = SlurmDistribute(**kwargs)
    tool.run(cmd=cmd, resume=resume, force=force)

if __name__ == "__main__":

    # shutil.rmtree("/tmp/test/")
    main()
