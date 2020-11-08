# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """Doesn't run anything; instead, prints each command.
    Useful for testing."""
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def hpc_launcher(commands):
    with open('jobs.txt', 'w') as f:
        f.writelines(commands)
    
    num_tasks = len(commands)
    print("num_tasks", num_tasks)
    with open('submit.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        f.write("#SBATCH --mem-per-cpu=50MB\n")
        f.write("#SBATCH --partition=p40_4,p100_4,v100_sxm2_4,v100_pci_2\n")
        f.write(f"#SBATCH --array=1-{num_tasks}\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --time=1:00:00\n")

        f.write("srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)")
    subprocess.call("sbatch -vv submit.sh", shell=True)

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'hpc': hpc_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
