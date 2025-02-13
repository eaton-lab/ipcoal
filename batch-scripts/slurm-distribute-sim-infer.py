#!/usr/bin/env python

"""Distributes `run-sim-loci-inference.py` on a SLURM cluster.

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
>>>   --ncores 2 \
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

from typing import List
import sys
import time
import argparse
from pathlib import Path
from subprocess import Popen, STDOUT, PIPE
import numpy as np


SBATCH = """\
#!/bin/sh

#SBATCH --account={account}
#SBATCH --job-name={jobname}
#SBATCH --output={outpath}.out
#SBATCH --error={outpath}.err
#SBATCH --time=11:59:00
#SBATCH --ntasks={ncores}
#SBATCH --mem=12G

# run the command to write and submit a shell script
python {root}/run-sim-loci-inference.py \
--neff {neff} \
--ctime {ctime} \
--mut {mut} \
--recomb {recomb} \
--nsites {nsites} \
--nloci {nloci} \
--rep {rep} \
--seed {seed} \
--outdir {outdir} \
--ncores {ncores} \
--node-heights {node_heights} \
--raxml-bin {raxml_bin} \
--astral-bin {astral_bin}
"""


def write_and_submit_sbatch_script(
    neff: int,
    ctime: int,
    mut: float,
    recomb: float,
    rep: int,
    seed: int,
    nsites: int,
    nloci: int,
    ncores: int,
    outdir: Path,
    account: str,
    node_heights: List[float],
    raxml_bin: Path,
    astral_bin: Path,
    dry_run: bool,
    ):
    """Submit an sbatch job to the cluster with these params."""
    # build parameter name string
    params = (
        f"neff{neff}-ctime{ctime}-"
        f"recomb{int(bool(recomb))}-rep{rep}-"
        f"nloci{max(nloci)}-nsites{nsites}"
    )

    # expand sbatch shell script with parameters
    sbatch = SBATCH.format(**dict(
        account=account,
        jobname=params,
        ncores=ncores,
        neff=neff,
        ctime=ctime,
        mut=mut,
        recomb=recomb,
        nsites=nsites,
        nloci=" ".join([str(i) for i in nloci]),
        rep=rep,
        seed=seed,
        node_heights=" ".join([str(i) for i in node_heights]),
        raxml_bin=raxml_bin,
        astral_bin=astral_bin,
        outdir=outdir,
        outpath=outdir / params,
        root=str(Path(__file__).parent),
    ))
    # print(sbatch)

    # write the sbatch shell script (careful here using with_suffix
    # b/c the params string name has a '.' in it for decimal ctime.
    tmpfile = (outdir / (params + '.sh'))
    with open(tmpfile, 'w', encoding='utf-8') as out:
        out.write(sbatch)

    # submit job to HPC SLURM job manager
    if not dry_run:
        cmd = ['sbatch', str(tmpfile)]
        with Popen(cmd, stdout=PIPE, stderr=STDOUT) as proc:
            out, _ = proc.communicate()


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
    >>>     --step 1
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--step', type=int, default=1, help='Analysis step: 1=gene tree inference, 2=species tree inference.')
    parser.add_argument(
        '--neff', type=int, default=[10000, 100000], nargs="*", help='Effective population size')
    parser.add_argument(
        '--ctime', type=float, default=[0.1, 1.5], nargs="*", help='Root species tree height in coalescent units')
    parser.add_argument(
        '--recomb', type=float, default=[0, 5e-9], nargs=2, help='Recombination rate.')
    parser.add_argument(
        '--mut', type=float, default=5e-8, help='Mutation rate.')
    parser.add_argument(
        '--node-heights', type=float, default=[0.05, 0.055, 0.06, 1], nargs=4, help='Internal relative node heights')
    parser.add_argument(
        '--nsites', type=int, default=[2000], nargs="*", help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, default=[20000], nargs="*", help='number of independent simulated loci.')
    parser.add_argument(
        '--nreps', type=int, default=100, help='number replicate per param setting.')
    parser.add_argument(
        '--outdir', type=Path, default=Path("."), help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--account', type=str, default="free", help='Account name for SLURM job submission')
    parser.add_argument(
        '--ncores', type=int, default=2, help='Number of cores per job (recommended=2)')
    parser.add_argument(
        '--dry-run', action='store_true', help='Write sbatch scripts but do not submit them.')
    parser.add_argument(
        '--delay', type=float, default=0.5, help='Number of seconds delay between SLURM job submissions.')

    return parser.parse_args()


def _slurm_cleanup(self):
    """Not currently implemented.
    
    For example, a rep with 3 nloci counts will produce 9 files 
    in this folder.
    - rep0-astral-genealogy-subloci20.nwk
    - rep0-astral-genealogy-subloci50.nwk        
    - rep0-astral-genealogy-subloci100.nwk        
    - rep0-astral-genetree-subloci20.nwk
    - rep0-astral-genetree-subloci50.nwk        
    - rep0-astral-genetree-subloci100.nwk        
    - rep0-concat-subloci20.nwk
    - rep0-concat-subloci50.nwk        
    - rep0-concat-subloci100.nwk

    This will be simplified into:
    Neff    ctime    nloci   loclen  rep     c     a1     a2
    10000    0.1      20      1000    0    (...)  (...)  (...)
    10000    0.1      50      1000    0    (...)  (...)  (...)
    10000    0.1     100      1000    0    (...)  (...)  (...)        
    """
    # # cleanup
    # jobparams = (
    #     f"neff{int(neff)}-ctime{ctime}-"
    #     f"recomb{int(bool(recomb))}-rep{rep}-"
    #     f"nloci{max(nloci)}-nsites{nsites}"
    # )
    # sh_file = outdir / (jobparams + ".sh")
    # err_file = outdir / (jobparams + ".err")
    # out_file = outdir / (jobparams + ".out")

    # if not err_file.stat().st_size:
    #     err_file.unlink()
    # if not out_file.stat().st_size:
    #     out_file.unlink()
    # sh_file.unlink()

    # # remove directory for ...
    # tmpdir.rmdir()
    

def main():
    """Parse command line args, write sbatch scripts, and submit to SLURM."""

    # parse command line args
    args = distributed_command_line_parser()

    # build grid of all jobs
    nlen = len(args.neff)
    rlen = len(args.recomb)
    clen = len(args.ctime)
    ilen = args.nreps
    slen = len(args.nsites)
    njobs = nlen * rlen * clen * slen * ilen
    print(f"Submitting {njobs} sbatch jobs at {args.delay} second intervals.")

    # find conda installed packages
    Path(args.outdir).mkdir(exist_ok=True)
    bindir = Path(sys.prefix) / "bin"
    astral_bin = bindir / "astral.5.7.1.jar"
    raxml_bin = bindir / "raxml-ng"
    assert astral_bin.exists(), f"cannot find {astral_bin}. Use conda instructions."
    assert raxml_bin.exists(), f"cannot find {raxml_bin}. Use conda instructions."

    # distribute jobs over all params except NLOCI (pass whole list).
    seeds = np.random.default_rng(123).integers(1e12, size=args.nreps)
    for rep in range(args.nreps):
        for nsites in args.nsites:
            for ctime in args.ctime:
                for recomb in args.recomb:
                    for neff in args.neff:
                        
                        # the folder name where this job will output
                        paramdir = (
                            f"neff{int(neff)}-ctime{ctime}-"
                            f"recomb{int(bool(recomb))}-"
                            f"nloci{max(args.nloci)}-nsites{nsites}"
                        )                        

                        # if step 1 then skip if `...gene_trees.csv` exists
                        path = args.outdir / params / TODO

                        # if step 2 then skip if `...astral-genetree...nwk` exists

                        # TODO
                        # check if this rep's result already exist
                        # paths = [args.outdir / (params + f"-astral-genetree-subloci{i}.nwk") for i in args.nloci]
                        # if all(i.exists() for i in paths):
                        #     njobs -= 1
                        #     print(f"skipping job {params}, result files exist.")
                        #     continue

                        # gtime = int(ctime * 4 * neff)
                        write_and_submit_sbatch_script(
                            neff=neff,
                            ctime=ctime,
                            mut=args.mut,
                            recomb=recomb,
                            nloci=args.nloci,
                            nsites=nsites,
                            rep=rep,
                            seed=seeds[rep],
                            ncores=args.ncores,
                            outdir=args.outdir,
                            account=args.account,
                            node_heights=args.node_heights,
                            raxml_bin=raxml_bin,
                            astral_bin=astral_bin,
                            dry_run=args.dry_run,
                        )
                        time.sleep(args.delay)
    print(f"{njobs} jobs submitted.")


if __name__ == "__main__":
    main()
