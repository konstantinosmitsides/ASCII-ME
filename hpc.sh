#!/usr/bin/env bash

# Check if container path is valid
if [ ! -f $1 ]; then
	echo ERROR: invalid container path.
	exit 1
fi

# Parse hpc.yaml configuration file
script_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source $script_dir/parse_yaml.sh
eval $(parse_yaml apptainer/hpc.yaml)

# Define additional shell variables
repository_name=${PWD##*/}
container_path=$1
container_name=${container_path##*/}
container_directory=${container_name%.*}
commit=$(echo $container_directory | cut -d "_" -f 4)

# Check parsed configuration
if [ -z "$walltime" ]; then
	echo ERROR: walltime not defined in hpc.yaml.
	exit 1
fi
if [ -z "$nnodes" ]; then
	echo ERROR: nnodes not defined in hpc.yaml.
	exit 1
fi
if [ -z "$ncpus" ]; then
	echo ERROR: ncpus not defined in hpc.yaml.
	exit 1
fi
if [ -z "$mem" ]; then
	echo ERROR: mem not defined in hpc.yaml.
	exit 1
fi
if [ -z "$ngpus" ]; then
	echo ERROR: ngpus not defined in hpc.yaml.
	exit 1
fi
if [ -z "$gpu_type" ]; then
	echo ERROR: gpu_type not defined in hpc.yaml.
	exit 1
fi
if [ ! $njobs -gt 0 ]; then
	echo ERROR: njobs needs to be a positive integer.
	exit 1
fi

# Create select shell variable
if [ $ngpus -gt 0 ]; then
	select="$nnodes:ncpus=$ncpus:mem=$mem:ngpus=$ngpus:gpu_type=$gpu_type"
else
	select="$nnodes:ncpus=$ncpus:mem=$mem"
fi

# Create array job PBS directive
if [ $njobs -gt 1 ]; then
	pbs_array="#PBS -J 1-$njobs"
fi

# Create queue shell variable
if [ "$queue" == "null" ]; then
	queue=""
fi

# Create temporary directory
tmp_dir=$(mktemp -d)

# Save git log output to track container
git log --decorate --color -10 $commit > $tmp_dir/git-log.txt

# Send container and git log to the HPC
ssh hpc "mkdir -p ~/$repository_name/$container_directory/"
rsync --verbose --ignore-existing --progress -e ssh $container_path $tmp_dir/git-log.txt hpc:~/$repository_name/$container_directory/

# Create jobscripts
table="|Job ID|Job Name|Job Script|Status|args\n"
counter=1
for args in $args_; do
	# Expand args
	args=$(eval echo \$${args})

	# Build jobscript from template
	tmp_jobscript=$(mktemp -p $tmp_dir)
	sed "s/{{ job_name }}/$job_name/g" $script_dir/template.job > $tmp_jobscript
	sed -i "s/{{ walltime }}/$walltime/g" $tmp_jobscript
	sed -i "s/{{ select }}/$select/g" $tmp_jobscript
	sed -i "s/{{ pbs_array }}/$pbs_array/g" $tmp_jobscript
	sed -i "s/{{ repository_name }}/$repository_name/g" $tmp_jobscript
	sed -i "s/{{ container_directory }}/$container_directory/g" $tmp_jobscript
	sed -i "s/{{ container_name }}/$container_name/g" $tmp_jobscript
	sed -i "s/{{ commit }}/$commit/g" $tmp_jobscript
	sed -i "s/{{ args }}/$args/g" $tmp_jobscript

	# Send jobscript to the HPC
	rsync --quiet --progress -e ssh $tmp_jobscript hpc:~/$repository_name/$container_directory/
	jobid=$(ssh hpc "cd ~/$repository_name/$container_directory/ && /opt/pbs/bin/qsub $queue ~/$repository_name/$container_directory/${tmp_jobscript##*/} 2> /dev/null")

	# Rename jobscript to $jobid.job
	if [ $? == 0 ]; then
		ssh hpc "mv ~/$repository_name/$container_directory/${tmp_jobscript##*/} ~/$repository_name/$container_directory/${job_name}_${jobid%.*}.job"
		table+="$counter|${jobid%.*}|$job_name|${job_name}_${jobid%.*}.job|Queued|$args\n"
	else
		ssh hpc "rm ~/$repository_name/$container_directory/${tmp_jobscript##*/}"
		table+="$counter|-|-|-|Failed|$args\n"
	fi

	counter=$((counter + 1))
done

rm -rf $tmp_dir
echo
echo -e $table | column -s '|' -t
