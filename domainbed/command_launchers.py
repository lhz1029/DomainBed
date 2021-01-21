# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
from datetime import datetime

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
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    with open('jobs_{}.txt'.format(date_time), 'w') as f:
        # singularity_cmd = '''singularity exec --nv --overlay /scratch/wz727/chestXR/data/mimic-cxr.sqsh:ro ''' + \
        #     '''--overlay /scratch/wz727/chestXR/data/chestxray8.sqsh:ro ''' + \
        #     '''--overlay /scratch/lhz209/data/padchest.sqf:ro ''' + \
        #     '''--overlay /scratch/lhz209/data/chexpert.sqf:ro ''' + \
        #     '''--overlay /scratch/lhz209/pytorch1.7.0-cuda11.0.ext3:ro ''' + \
        #     '''/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif ''' + \
        #     '''bash -c 'source /ext3/env.sh; cd /scratch/wz727/chestXR/DomainBed; '''
        # commands = [singularity_cmd + cmd + "; '" for cmd in commands]
        f.write('\n'.join(commands) + '\n')
    num_tasks = len(commands)
    print("num_tasks", num_tasks)
    with open('submit_{}.sh'.format(date_time), 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        # f.write("#SBATCH --mem-per-cpu=1GB\n")
        # f.write("#SBATCH --partition=p40_4,p100_4,v100_sxm2_4,v100_pci_2\n")
        f.write(f"#SBATCH --array=1-{num_tasks}\n")
        f.write("#SBATCH --gres=gpu\n")
        f.write("#SBATCH --mem=64GB\n")
        f.write("#SBATCH --time=5-12:00:00\n")

        # f.write("command=$(head -n $SLURM_ARRAY_TASK_ID jobs_{}.txt | tail -n 1)\n".format(date_time))
        # f.write('srun "${command[@]}"')
        # f.write("srun $(head -n $SLURM_ARRAY_TASK_ID jobs_{}.txt | tail -n 1)".format(date_time))\
        singularity_cmd = '''srun singularity exec --nv --overlay /scratch/wz727/chestXR/data/mimic-cxr.sqsh:ro \
            --overlay /scratch/wz727/chestXR/data/chestxray8.sqsh:ro \
            --overlay /scratch/lhz209/data/padchest.sqf:ro \
            --overlay /scratch/lhz209/data/chexpert.sqf:ro \
            --overlay /scratch/lhz209/pytorch1.7.0-cuda11.0.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c 'source /ext3/env.sh; cd /scratch/wz727/chestXR/DomainBed; \
            $(head -n $SLURM_ARRAY_TASK_ID jobs_{}.txt | tail -n 1)'
            '''.format(date_time)
        f.write(singularity_cmd)
        # f.write("$(head -n $SLURM_ARRAY_TASK_ID jobs_{}.txt | tail -n 1)".format(date_time))
    subprocess.call("sbatch -vv submit_{}.sh".format(date_time), shell=True)

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
