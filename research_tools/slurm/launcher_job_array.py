# a more sophisticated launcher which can run multiple jobs in one script
import argparse as ap 
import os
import pathlib

dir_path = pathlib.Path(__file__).parent.absolute()
job_array_loc = os.path.join(dir_path, 'job_array_py.py')

parser = ap.ArgumentParser()
parser.add_argument("--commands_file", type=str, help="location of file containing commands") # one line for each command
parser.add_argument("--partition", choices=["tiger", "tiger-quick", "tiger-lo"], help="where to run jobs")
parser.add_argument("--gpus", type=int, default=1, help="number of gpus per job")
parser.add_argument("--node", type=str, help="request a specific node")
parser.add_argument("--mem", type=int, default=8000, help="memory per job")
parser.add_argument("--mincpus", type=int, default=8, help="number of cpus to use")
args = parser.parse_args()

user = os.environ.get('USER')
command_string = "sbatch --partition %s --gres=gpu:%d --mem %d --mincpus %d" % (args.partition, args.gpus, args.mem, args.mincpus)

if args.partition == "tiger-lo":
    # prevent from requeueing twice from pre-empting
    command_string += " --no-requeue"
if args.node:
    command_string += " --nodelist=%s" % (args.node)

# take the prefix name of the file
commands_prefix = args.commands_file.split(".")[0].split("/")[-1]

if not os.path.exists("commands_dir"):
    os.makedirs("commands_dir")

if not os.path.exists("job_files_dir"):
    os.makedirs("job_files_dir")

with open(args.commands_file, "r") as f:
    line_collection = []
    file_counter = 1
    for line in f:
        if len(line) == 0 or line.startswith('#'):
            continue
        if "*****" in line:
            # First we write the jobs file. 
            jobs_file = os.path.join("job_files_dir", commands_prefix + "_%d" % (file_counter))
            with open(jobs_file, "w") as f_jobs:
                for ind, new_line in enumerate(line_collection):
                    f_jobs.write(new_line)

            # Next we set up the slurm commands file.
            slurm_commands = os.path.join("commands_dir", commands_prefix + "_%d.sh" % (file_counter))
            with open(slurm_commands, "w") as f_slurm:

                if user == 'kshen6':
                    f_slurm.write("#!/usr/bin/env bash\n#SBATCH --output=slurm_outs/slurm-%j.out\nsource ~/.bashrc\nsource activate replearn\n")
                else:
                    f_slurm.write("#!/usr/bin/env bash\n#SBATCH --output=slurm_outs/slurm-%j.out\nsource ~/.bashrc\nconda activate torch1.3\n")
                f_slurm.write("echo \"PATH=$PATH\"\n")
                f_slurm.write("echo \"SHELL=$SHELL\"\n")
                f_slurm.write("export PYTHONUNBUFFERED=1\n")
                f_slurm.write("exec python %s --save_dir %s --jobs_file %s %s --launcher_file %s --partition %s --gpus %d --mem %s --mincpus %d \n" %
                    (
                        job_array_loc,
                        "$SLURM_JOB_ID", 
                        jobs_file, 
                        "--preemptable " if args.partition == "tiger-lo" else "", 
                        slurm_commands, 
                        args.partition,
                        args.gpus,
                        args.mem,
                        args.mincpus
                    ))
            full_command = command_string + " %s" % (slurm_commands)
            print("Launching", full_command)
            print("Commands", line_collection)
            os.system(full_command)
            file_counter += 1
            line_collection = []
        else:
            line_collection.append(line)


