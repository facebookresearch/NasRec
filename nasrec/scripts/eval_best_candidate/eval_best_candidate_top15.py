import os
from typing import List
from time import sleep
import secrets

class HyperParams:
    learning_rate: float = 0.12
    weight_decay: float = 0
    batch_size: int = 512           # 512 for Criteo/Avazu, 1024 for KDD.
    test_interval: int = 10000
    
class BenchMark:
    dataset: str = "avazu-kaggle"
    search_space: str = "xlarge"
    train_limit: int = -1
    test_limit: int = -1
    top_k: int = 15 
    shot: str = "1shot"
    
    trainval_train_limit : int = -1
    trainval_val_limit : int = -1
    traintest_train_limit: int = -1
    traintest_test_limit: int = -1
    
class Items:
    trainval_train_limit: int = -1
    trainval_val_limit: int = -1
    traintest_train_limit: int = -1    
    traintest_test_limit: int = -1
  
class CriteoKaggleItems(Items):
    trainval_train_limit: int = 36672495
    trainval_val_limit: int = 4584061
    traintest_train_limit: int = 41256556    
    traintest_test_limit: int = 4584061


class KDDItems(Items):
    trainval_train_limit: int = 119711284    
    trainval_val_limit: int = 14963910
    traintest_train_limit: int = 134675194
    traintest_test_limit: int = 14963910


class AvazuItems(Items):
    trainval_train_limit: int = 32343175
    trainval_val_limit: int = 4042896
    traintest_train_limit: int = 36386071 
    traintest_test_limit: int = 4042896


class CommandGenerator:
    def __init__(self):
        self.content: List[str] = []

    def __str__(self):
        return "\n".join(self.content)
    
    def push(self, new_str):
        self.content.append(new_str)
        
    def write(self, file):
        with open(file, 'w') as fp:
            fp.write(str(self))
            

#TODO: Add a slurm partition name.
SLURM_PARTITION = ""
#TODO: Add a slurm account name.
SLURM_ACCOUNT_NAME = ""
#TODO: You can also launch a plain version by modifying this script.

def get_slurm_header(command_gen_obj: CommandGenerator, job_name: str, entitlement=None):
    command_gen_obj.push("#!/usr/bin/sh")
    command_gen_obj.push("#SBATCH --job-name={}".format(job_name))
    command_gen_obj.push("#SBATCH --cpus-per-task 8")
    command_gen_obj.push("#SBATCH --gpus-per-task 1")
    command_gen_obj.push("#SBATCH -e {}.err".format(job_name))
    command_gen_obj.push("#SBATCH -o {}.out".format(job_name))
    command_gen_obj.push("#SBATCH --partition {}".format(entitlement))
    command_gen_obj.push("#SBATCH --account {}".format(SLURM_ACCOUNT_NAME))
    return command_gen_obj


dataset_to_items_dict = {
    'kdd': KDDItems,
    'criteo-kaggle': CriteoKaggleItems,
    "avazu": AvazuItems,
}


def copy_benchmark_train_test_limits(benchmark: BenchMark, dataset = "kdd") -> BenchMark:
    item_obj = dataset_to_items_dict[dataset]
    benchmark.traintest_test_limit = item_obj.traintest_test_limit
    benchmark.traintest_train_limit = item_obj.traintest_train_limit
    benchmark.trainval_train_limit = item_obj.trainval_train_limit
    benchmark.trainval_val_limit = item_obj.trainval_val_limit
    return benchmark


data_style_mapping = {
    "criteo-kaggle": "criteo_kaggle",
    "kdd": "kdd_kaggle",
    "avazu": "avazu_kaggle",
}

def main():
    hparmas = HyperParams()
    benchmarks = BenchMark()
    benchmarks.dataset = "avazu"
    benchmarks.search_space = "xlarge"
    copy_benchmark_train_test_limits(benchmarks, benchmarks.dataset)
    data_style = data_style_mapping[benchmarks.dataset]
    for job_id in range(benchmarks.top_k):
        tmp_folder_id = secrets.randbits(32)
        tmp_folder = "/tmp/{}".format(tmp_folder_id)
        ea_root_dir = "ea-{}-{}-best-{}".format(
            benchmarks.dataset, benchmarks.search_space, benchmarks.shot)
        job_name = "ea_{}_{}_n{}".format(benchmarks.dataset, benchmarks.search_space, job_id)
        os.makedirs(tmp_folder, exist_ok=True)
        script_path = os.path.join(tmp_folder, "run.sh") 
        command_liner = CommandGenerator()
        _ = get_slurm_header(command_liner, job_name, SLURM_PARTITION)
        # Now, generating PyThon scripts.
        command_liner.push("python -u nasrec/main_train.py \\")
        command_liner.push("--root_dir ./data/{}_autoctr/ \\".format(data_style))
        command_liner.push("--net supernet-config \\")
        command_liner.push("--supernet_config {}/best_config_{}.json \\".format(ea_root_dir, job_id))
        command_liner.push("--num_epochs 1 \\")
        command_liner.push("--learning_rate {} \\".format(hparmas.learning_rate))
        command_liner.push("--train_batch_size {} \\".format(hparmas.batch_size))
        command_liner.push("--wd {} \\".format(hparmas.weight_decay))
        command_liner.push("--logging_dir ./experiments-www/{}-config-{} \\".format(ea_root_dir, job_id))
        command_liner.push("--gpu 0 \\")
        command_liner.push("--test_interval {} \\".format(hparmas.test_interval))
        command_liner.push("--dataset {} --train_limit {} \\".format(benchmarks.dataset, benchmarks.trainval_train_limit))
        command_liner.push("--test_limit {} \\".format(benchmarks.trainval_val_limit))
        # Very Important. You should validate the best architecture on validation split.=
        command_liner.push("--train_split train --validate_split val")
        print("Dump command to {}!".format(script_path))
        # Dump files.
        command_liner.write(script_path)
        # Run the sbatch command.
        os.system("sbatch {}".format(script_path))
        # Clean up a tmp file.
        print("Clean up...")
        sleep(1)
        os.system("rm -rf {}".format(tmp_folder))


if __name__ == "__main__":
    main()
