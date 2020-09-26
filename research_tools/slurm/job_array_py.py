import sys
import signal
import argparse as ap 
import os 
import logging
import subprocess
import pathlib

dir_path = pathlib.Path(__file__).parent.absolute()
job_array_loc = os.path.join(dir_path, 'job_array_py.py')

parser = ap.ArgumentParser()
parser.add_argument('--save_dir', type=str, help='Where to save this set of runs. As a default, set to the ID of the slurm job which is currently running.')
parser.add_argument('--jobs_file', type=str, help='Location of the jobs to run.')

# The below arguments are in the case that we need to catch the SIGTERM and rerun. 
parser.add_argument('--preemptable', action='store_true', 
    help='Whether or not this job is preemptable. If it is, then we catch SIGTERM and re-run from the latest checkpoint.')
parser.add_argument('--launcher_file', type=str, help='The name of the bash script used to launch the slurm job.')
parser.add_argument('--partition', type=str, default='tiger', help='Which cluster to run this on.')
parser.add_argument('--gpus', type=int, default=1, help='How many gpus to use.')
parser.add_argument('--mem', type=str, default='12000', help='Memory to allocate for job.')
parser.add_argument('--mincpus', type=int, default=8, help='How many cpus to use.')

global args
global job_arr
global curr_job_ind

args = parser.parse_args()

job_arr = []

with open(args.jobs_file, "r") as f: 
    for line in f:
        job_arr.append(line)

def find_lastest_dir_id():
    max_dir_id = -1
    max_dir_name = None
    for dir_name in os.listdir(save_dir_name):
        complete_dir = os.path.join(save_dir_name, dir_name)
        if os.path.isdir(complete_dir):
            dir_id = int(dir_name.split("_")[1])
            if dir_id > max_dir_id:
                max_dir_id = dir_id
                max_dir_name = complete_dir
    return max_dir_id, max_dir_name

def sigterm_handler(signal, frame):
    # catch the sigterm signal and set up the new job array
    #logging.debug('This message should ALSO go to the log file')
    sys.stdout.write("Caught SIGTERM signal.")
    sys.stdout.flush()
    curr_command = job_arr[curr_job_ind]

    new_job_arr = job_arr[curr_job_ind:]

    # modify this current command to have a resume option
    new_job_arr[0] = new_job_arr[0].rstrip() + " --resume_killed\n" 

    # create a new job file 
    new_job_file = args.jobs_file + "_resume"
    with open(new_job_file, "w") as f:
        for command in new_job_arr:
            f.write(command)

    new_launcher_file = args.launcher_file.split(".sh")[0] + "_resume.sh"
    with open(new_launcher_file, "w") as f:
        f.write("#!/bin/sh\n#SBATCH --output=slurm_outs/slurm-%j.out\n")
        f.write("echo \"PATH=$PATH\"\n")
        f.write("echo \"SHELL=$SHELL\"\n")
        f.write("exec python %s --save_dir %s --jobs_file %s %s--launcher_file %s --partition %s --gpus %d --mem %s --mincpus %d \n" %
            (
                job_array_loc,
                args.save_dir, 
                new_job_file, 
                "--preemptable " if args.preemptable else "", 
                new_launcher_file, 
                args.partition,
                args.gpus,
                args.mem,
                args.mincpus
            ))

    new_command_string = "sbatch --partition %s --gres=gpu:%d --mem %s --mincpus %d --no-requeue" % (args.partition, args.gpus, args.mem, args.mincpus)
    full_command = new_command_string + " %s" % (new_launcher_file)
    print("SIGTERM received. Resuming with command %s" % (full_command))
    sys.stdout.flush()
    os.system(full_command)
    sys.exit(0)


if args.preemptable:
    print("Registered signal handler.")
    sys.stdout.flush()
    signal.signal(signal.SIGTERM, sigterm_handler)

# now we crawl the save dir and find the latest integer for the run ID

curr_job_ind = 0
for job_command in job_arr:
    # argument to be passed into --slurm_id
    job_mod = job_command.rstrip() + " --slurm_dir %s\n" % (args.save_dir)
    subprocess.call(job_mod, shell=True)
    curr_job_ind += 1
