
import os,glob
import ruamel.yaml as yaml

configfile: 'experiment_params.yaml'
EPISODES=config['episodes']
template_files= [os.path.join(dirpath,file) for dirpath,_,files in os.walk("Template") for file in files]
req_gen=lambda x: [os.path.join(os.path.dirname(i),x) for i in glob.glob("grid/**/.flag")]
#PROJECT-SPECIFIC
DATESPY='python' if config['location']=='local' else config['singularity_image']
PLOTPY=lambda x:'frameworkpython.sh {}'.format(x) if config['location']=='local' else 'module load application/anaconda3 && python3 {} && module rm application/anaconda3'.format(x)

#-- set some location stuff

if config['location']=='local':
	PY='python'
	FPY='frameworkpython'
	PYMOD='python'
	RW='~/.virtualenvs/tf3/bin/record_writer.py'
	MODULE_CMD='echo none'
elif config['location']=='slurm':
	PY=config['singularity_image']
	FPY=PY
	PYMOD='sbatch {}'.format(SLURMSCRIPT)
	RW='~/.local/bin/record_writer.py'
	MODULE_CMD='module add application/singularity'
elif config['location']=='cluster':
	PY=config['singularity_image']
	FPY=PY
	PYMOD=config['singularity_image']
	RW='~/.local/bin/record_writer.py'
	MODULE_CMD='module add application/singularity'

rule INIT:
	input: dynamic("grid/{manip}/.flag")

rule TRAIN:
	input: dynamic("grid/{manip}/output.txt")


rule DATES:
	input: dynamic("grid/{manip}/experimenttestGDPtime.csv"), dynamic("grid/{manip}/experimenttrainGDPtime.csv")
rule PLOTS:
	input: dynamic("plots/{manip}_abs_err.pdf"), dynamic("plots/{manip}_preds_v_targets.pdf")

rule SUMMARY:
	input: dynamic("grid/{manip}/experiment.preds"), dynamic("grid/{manip}/experiment.targets")
	shell: "{PY} utils/test_summary.py --grid_folder='grid' --output_file='test_summary.csv'"


rule _INIT_GRID:
	input: "utils/build_grid.py", template_files, "experiment_params.yaml"
	output: dynamic("grid/{manip}/.flag")
	run:
		shell("mkdir -p grid")
		shell("utils/build_grid.py")

# rule _GRID: 
# 	output: "grid"
# 	shell: "mkdir -p grid"



rule _TRAIN:
	input: "grid/{manip}/.flag", "grid/{manip}/.flag"
	output:
		targets="grid/{manip}/output.txt"
	run:
		econfig=yaml.safe_load(open("grid/{manip}/experiment.yaml".format(manip=wildcards.manip),'r'))
		print(econfig)

		fdict=dict(PY=PY,sdir="grid/{}".format(wildcards.manip))
		fdict.update(**econfig)
		shell("{PY} {sdir}/model_build.py --subdir {sdir} --env {gym} --episodes {episodes} ".format(**fdict))





