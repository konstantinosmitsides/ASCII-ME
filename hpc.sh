#!/usr/bin/env bash

# Check if container path is valid
if [ ! -f "$1" ]; then
    echo "ERROR: invalid container path."
    exit 1
fi

# Parse hpc.yaml configuration file
script_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source "$script_dir/parse_yaml.sh"
eval $(parse_yaml apptainer/hpc.yaml)

# Define additional shell variables
repository_name=${PWD##*/}
container_path="$1"
container_name=${container_path##*/}
container_directory=${container_name%.*}
commit=$(echo "$container_directory" | cut -d "_" -f 4)

# Check parsed configuration
if [ -z "$walltime" ]; then
    echo "ERROR: walltime not defined in hpc.yaml."
    exit 1
fi
if [ -z "$nnodes" ]; then
    echo "ERROR: nnodes not defined in hpc.yaml."
    exit 1
fi
if [ -z "$ncpus" ]; then
    echo "ERROR: ncpus not defined in hpc.yaml."
    exit 1
fi
if [ -z "$mem" ]; then
    echo "ERROR: mem not defined in hpc.yaml."
    exit 1
fi
if [ -z "$ngpus" ]; then
    echo "ERROR: ngpus not defined in hpc.yaml."
    exit 1
fi
if [ -z "$gpu_type" ]; then
    echo "ERROR: gpu_type not defined in hpc.yaml."
    exit 1
fi
if [ ! "$njobs" -gt 0 ]; then
    echo "ERROR: njobs needs to be a positive integer."
    exit 1
fi

# Create select shell variable
if [ "$ngpus" -gt 0 ]; then
    select="$nnodes:ncpus=$ncpus:mem=$mem:ngpus=$ngpus:gpu_type=$gpu_type"
else
    select="$nnodes:ncpus=$ncpus:mem=$mem"
fi

# Create array job PBS directive
#if [ "$njobs" -gt 1 ]; then
#    pbs_array="#PBS -J 1-$njobs"
#else

pbs_array=""

# Create queue shell variable
if [ "$queue" == "null" ]; then
    queue=""
fi

# Create temporary directory
tmp_dir=$(mktemp -d)

# Save git log output to track container
git log --decorate --color -10 "$commit" > "$tmp_dir/git-log.txt"

# Send container and git log to the HPC
ssh km2120@login.cx3.hpc.imperial.ac.uk "mkdir -p ~/$repository_name/$container_directory/"
rsync --verbose --ignore-existing --progress -e ssh "$container_path" "$tmp_dir/git-log.txt" km2120@login.cx3.hpc.imperial.ac.uk:~/"$repository_name"/"$container_directory"/

# Define hyperparameters
# clip_params=(0.2)
# stds=(2)
# learning_rates=(0.003)
no_epochs=(4 8 16 32)
envs=(ant_uni walker2d_uni hopper_uni ant_omni anttrap_omni)

common_params="HPC=L40S algo=mcpg_me env.episode_length=250 num_iterations=4000 batch_size=256 algo.proportion_mutation_ga=0.5 init=orthogonal algo.cosine_similarity=True algo.learning_rate=3e-3 algo.std=2 algo.clip_param=0.2"

# Generate all configurations
configurations=()

for epochs in "${no_epochs[@]}"; do
    for env in "${envs[@]}"; do
        config="$common_params env=$env algo.no_epochs=$epochs"
        configurations+=("$config")
    done
done


# Multiply configurations by 3 to run each with a random seed 3 times
expanded_configurations=()
for cfg in "${configurations[@]}"; do
    for i in {1..5}; do
        expanded_configurations+=("$cfg seed=\$RANDOM")
    done
done
configurations=("${expanded_configurations[@]}")

# Total number of experiments
total_experiments=${#configurations[@]}  # Should be 2880
experiments_per_job=$(( (total_experiments + njobs - 1) / njobs ))  # Ensure all experiments are included

# Create jobscripts
table="|Job ID|Job Name|Job Script|Status|Number of Experiments\n"

for ((i=0; i<njobs; i++)); do
    start_index=$((i * experiments_per_job))
    end_index=$((start_index + experiments_per_job))
    if [ "$end_index" -gt "$total_experiments" ]; then
        end_index=$total_experiments
    fi
    num_experiments=$((end_index - start_index))
    job_configs=("${configurations[@]:$start_index:$num_experiments}")

    # Build jobscript from template
    tmp_jobscript=$(mktemp -p "$tmp_dir")
    sed "s/{{ job_name }}/$job_name/g" "$script_dir/template.job" > "$tmp_jobscript"
    sed -i "s/{{ walltime }}/$walltime/g" "$tmp_jobscript"
    sed -i "s/{{ select }}/$select/g" "$tmp_jobscript"
    sed -i "s/{{ pbs_array }}/$pbs_array/g" "$tmp_jobscript"
    sed -i "s/{{ repository_name }}/$repository_name/g" "$tmp_jobscript"
    sed -i "s/{{ container_directory }}/$container_directory/g" "$tmp_jobscript"
    sed -i "s/{{ container_name }}/$container_name/g" "$tmp_jobscript"
    sed -i "s/{{ commit }}/$commit/g" "$tmp_jobscript"

    # Append the configs array to the job script
    echo "configs=(" >> "$tmp_jobscript"
    for cfg in "${job_configs[@]}"; do
        printf "\"%s\"\n" "$cfg" >> "$tmp_jobscript"
    done
    echo ")" >> "$tmp_jobscript"

    # Append the rest of the job script

cat << EOF >> "$tmp_jobscript"

# Transfer files from server to compute node
stagein()
{
    # Create output/ directory on server
    cmd="mkdir -p \$HOME/$repository_name/$container_directory/output/"
    echo \$(date +"%Y-%m-%d %H:%M:%S") stagein: \$cmd
    eval \$cmd

    # Create symbolic link to container from server in compute node
    cmd="ln -s \$HOME/$repository_name/$container_directory/$container_name"
    echo \$(date +"%Y-%m-%d %H:%M:%S") stagein: \$cmd
    eval \$cmd
}

# Run container
runprogram()
{
    # Check if the symbolic link exists
    if [ -L "./output" ]; then
        # Check if it points to the correct target
        TARGET_LINK="\$(readlink ./output)"
        EXPECTED_TARGET="\$HOME/$repository_name/$container_directory/output/"
        if [ "\$TARGET_LINK" != "\$EXPECTED_TARGET" ]; then
            # Update the symbolic link to point to the correct target
            cmd="ln -sf \$EXPECTED_TARGET ./output"
            echo \$(date +"%Y-%m-%d %H:%M:%S") runprogram: "Updating symbolic link './output' to point to '\$EXPECTED_TARGET'"
            eval \$cmd
        else
            echo \$(date +"%Y-%m-%d %H:%M:%S") runprogram: "Symbolic link './output' already points to the correct target."
        fi
    else
        # Create the symbolic link
        cmd="ln -s \$HOME/$repository_name/$container_directory/output/ ./output"
        echo \$(date +"%Y-%m-%d %H:%M:%S") runprogram: "Creating symbolic link './output' pointing to '\$HOME/$repository_name/$container_directory/output/'"
        eval \$cmd
    fi

    # Create a temporary working directory
    WORKDIR=\$(mktemp -d -p "\$(pwd)" tmp.XXXXXXXXXX)

    # Run the experiments
    for args in "\${configs[@]}"; do
        echo "Running experiment with args: \$args"
        # Expand \$RANDOM
        eval_args=\$(eval echo "\$args")
        # Run your command within the container
        cmd="time SINGULARITYENV_PBS_JOB_INDEX=\$PBS_JOB_INDEX SINGULARITYENV_PBS_ARRAY_INDEX=\$PBS_ARRAY_INDEX apptainer run --bind ./output/:/workspace/src/output/ --cleanenv --containall --home /tmp/ --no-home --nv --pwd /workspace/src/ --workdir \$WORKDIR $container_name +commit=$commit \$eval_args"
        echo \$(date +"%Y-%m-%d %H:%M:%S") runprogram: \$cmd
        eval \$cmd
    done
}

# Transfer files from compute node to server and exit
stageout()
{
    cmd="exit"
    echo \$(date +"%Y-%m-%d %H:%M:%S") stageout: \$cmd
    eval \$cmd
}

stagein
runprogram
stageout

EOF

    # Send jobscript to the HPC
    rsync --quiet --progress -e ssh "$tmp_jobscript" km2120@login.cx3.hpc.imperial.ac.uk:~/"$repository_name"/"$container_directory"/

    # Submit the job
    jobid=$(ssh km2120@login.cx3.hpc.imperial.ac.uk "cd ~/$repository_name/$container_directory/ && /opt/pbs/bin/qsub $queue ${tmp_jobscript##*/} 2> /dev/null")

    # Handle job submission result
    if [ $? -eq 0 ]; then
        ssh km2120@login.cx3.hpc.imperial.ac.uk "mv ~/$repository_name/$container_directory/${tmp_jobscript##*/} ~/$repository_name/$container_directory/${job_name}_${jobid%.*}.job"
        table+="$((i+1))|${jobid%.*}|$job_name|${job_name}_${jobid%.*}.job|Queued|$num_experiments\n"
    else
        ssh km2120@login.cx3.hpc.imperial.ac.uk "rm ~/$repository_name/$container_directory/${tmp_jobscript##*/}"
        table+="$((i+1))|-|-|-|Failed|$num_experiments\n"
    fi
done

rm -rf "$tmp_dir"
echo
echo -e "$table" | column -s '|' -t