"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Important imports
from copy import deepcopy
import argparse

# Other imports
import numpy as np
import torch.multiprocessing as mp

# Project Imports.
from nasrec.searcher.searcher_utils import (
    create_model_train_and_get_results_helper,
    get_device_id,
)
from nasrec.searcher.tokenizer import Tokenizer
from nasrec.supernet.supernet import ops_config_lib

class Searcher(object):
    """
    A searcher to carry search and obtain the best network.
    Args:
        eval_fn (ptr): Evaluation function.
        model (nn.Module): PyTorch module.
        args (argparse.Namespace): arguments to control the search process.
        checkpoint: Model checkpoint.
    """

    def __init__(
        self,
        eval_fn,
        args: argparse.Namespace,
    ):
        self._eval_fn = eval_fn
        self._args = args
        self.all_results = []
        self._tokenizer = Tokenizer(
            num_blocks=args.num_blocks, ops_config=ops_config_lib[args.config]
        )
        self._checkpoint = None

    @staticmethod
    def _sort_results_with_criterion(
        results: np.ndarray,
        criterion: str = "test_loss",
        **kwargs,
    ) -> np.ndarray:
        # Check requirements and incorporate latency measurements.
        objs = []
        for idx in range(len(results)):
            if criterion == "test_loss_penalty_lat":
                # In the original TuNAS, we use absolute value to force exact latency.
                # Here we relax this constraint to favor efficient models.
                objective = results[idx]["test_loss"] + kwargs["beta"] * (
                    results[idx]["latency"] / kwargs["target_latency"] - 1
                )
            else:
                objective = results[idx][criterion]
            objs.append(objective)

        objs = np.asarray(objs).flatten()
        sorted_indices = np.argsort(objs)
        if criterion in ["test_acc", "test_auroc"]:
            return results[sorted_indices[::-1]]
        else:
            return results[sorted_indices]

    def _sampler(
        self, population: np.ndarray, num_samples: int, criterion: str = "test_loss"
    ) -> np.ndarray:
        sampled_indices = np.random.choice(len(population), num_samples, replace=False)
        return population[sampled_indices]

    def random_search_from_supernet(
        self,
        budget: int = 200,
        criterion: str = "test_loss",
        top_k: int = 5,
        num_parallel_workers: int = 1,
        on_cpu: bool = False,
        sorted: bool = True,
        **kwargs,
    ) -> np.ndarray:
        assert num_parallel_workers >= 1, "Should have at least 1 worker!"
        assert top_k <= budget, "Should have 'top_k' smaller than 'budget'."
        if on_cpu:
            assert num_parallel_workers == 1, ValueError(
                "Can only use 'num_parallel_workers=1' when on CPU."
            )
        assert criterion in [
            "test_loss",
            "test_acc",
            "test_auroc",
            "test_loss_penalty_lat",
        ], NotImplementedError("Criterion {} is not supported!".format(criterion))

        kwargs["beta"] = 0.0 if "beta" not in kwargs else kwargs["beta"]
        kwargs["target_latency"] = (
            -1 if "target_latency" not in kwargs else kwargs["target_latency"]
        )
        kwargs["latency_batch_size"] = (
            512 if "latency_batch_size" not in kwargs else kwargs["latency_batch_size"]
        )

        if kwargs["target_latency"] == -1 and kwargs["beta"] != 0:
            kwargs["target_latency"] = 0.0 # TODO: Add baseline latency to achieve latency-aware search.

        self.all_results = []
        idx = 0
        # The only way to bypass is to load checkpoint indirectly in the execution function.
        # Do not touch manifold operations before executing manifold operations.
        ckpt_holder = mp.Manager().dict()
        while idx < budget:
            print("Evaluating {} of {} random networks!".format(idx, budget))
            num_jobs = min(num_parallel_workers, budget - idx)
            processses = []
            # Empty result dict first.
            return_dict = mp.Manager().dict()
            # Start process and join to go back to the main process.
            for job_id in range(num_jobs):
                p = mp.Process(
                    target=create_model_train_and_get_results_helper,
                    args=(
                        deepcopy(self._args),
                        get_device_id(job_id, on_cpu),
                        deepcopy(self._eval_fn),
                        deepcopy(self._tokenizer),
                        None,
                        return_dict,
                        ckpt_holder,
                        kwargs,
                    ),
                )
                # Then, start process.
                p.start()
                processses.append(p)
            for p in processses:
                p.join()
            # Collect results.
            for key in list(return_dict.keys()):
                self.all_results.append(return_dict[key])
            idx += num_jobs

        self.all_results = np.asarray(self.all_results)
        # Now, get the best results.
        if sorted:
            return self._sort_results_with_criterion(
                self.all_results, criterion, **kwargs
            )[:top_k]
        else:
            return self.all_results[:top_k]

    def regularized_evolution_from_supernet(
        self,
        n_generations: int = 50,
        n_childs: int = 16,
        init_population: int = 100,
        sample_size: int = 5,
        criterion: str = "test_loss",
        skip_random: bool = False,
        num_parallel_workers: int = 1,
        on_cpu: bool = False,
        top_k: int = 2,
        **kwargs,
    ) -> np.ndarray:
        assert criterion in [
            "test_loss",
            "test_acc",
            "test_auroc",
            "test_loss_penalty_lat",
        ], NotImplementedError("Criterion {} is not supported!".format(criterion))
        assert top_k <= sample_size, ValueError(
            "You must maintain more than 'top_k' children to append 'top_k' archs to history."
        )
        assert sample_size < init_population, ValueError(
            "Sample size must be no greater than the number of population ('init_population')!"
        )
        assert num_parallel_workers >= 1, ValueError("Should have at least 1 worker!")
        if init_population < n_childs:
            print(
                "WARNING: For the best effect, you should have more initial population than children!"
            )
        if on_cpu:
            assert num_parallel_workers == 1, ValueError(
                "Can only use 'num_parallel_workers=1' when on CPU."
            )

        kwargs["beta"] = 0.0 if "beta" not in kwargs else kwargs["beta"]
        kwargs["target_latency"] = (
            -1 if "target_latency" not in kwargs else kwargs["target_latency"]
        )
        kwargs["latency_batch_size"] = (
            512 if "latency_batch_size" not in kwargs else kwargs["latency_batch_size"]
        )

        if kwargs["target_latency"] == -1 and kwargs["beta"] != 0:
            kwargs["target_latency"] = 0.0 # TODO: Add baseline latency to achieve latency-aware search.

        # Firstly, sample a bunch of initial populations.
        all_populations = self.random_search_from_supernet(
            budget=init_population,
            criterion=criterion,
            top_k=init_population,
            num_parallel_workers=num_parallel_workers,
            on_cpu=on_cpu,
            sorted=False,
            **kwargs,
        )
        print("Done random sample!")
        # Gathering all populations.
        all_populations = np.asarray(all_populations)
        history = []
        print(all_populations)
        visited_hash_tokens = []
        ckpt_holder = mp.Manager().dict()
        for n_gen in range(n_generations):
            # Now, sample the current population and best architecture.
            sampled_population = self._sampler(all_populations, sample_size, criterion)
            sorted_sampled_population = self._sort_results_with_criterion(
                sampled_population, criterion, **kwargs
            )
            # Parent arch selected.
            parent_arch = sorted_sampled_population[0]
            print("Parent Arch: {}".format(parent_arch))
            # Empty result dict first.
            return_dict = mp.Manager().dict()
            child_id = 0
            # Mutate more at the start, and less at the end. If less than 20 generations, mutations are set to 1.
            num_mutations = (n_generations - n_gen) // (max(20, n_generations // 5)) + 1
            while child_id < n_childs:
                num_jobs = min(n_childs - child_id, num_parallel_workers)
                print("Generation {}, Child {}!".format(n_gen, child_id))
                processses = []
                # Empty return dict first.
                return_dict = mp.Manager().dict()
                # Start process and join to go back to the main process.
                for job_id in range(num_jobs):
                    mutated_choice = deepcopy(parent_arch["choice"])
                    while True:
                        for _ in range(num_mutations):
                            mutated_choice = self._tokenizer.mutate_spec(mutated_choice)
                        mutated_token = self._tokenizer.tokenize(mutated_choice)
                        mutated_hash_token = self._tokenizer.hash_token(mutated_token)
                        if mutated_hash_token not in visited_hash_tokens:
                            visited_hash_tokens.append(mutated_hash_token)
                            break
                    p = mp.Process(
                        target=create_model_train_and_get_results_helper,
                        args=(
                            deepcopy(self._args),
                            get_device_id(job_id, on_cpu),
                            deepcopy(self._eval_fn),
                            deepcopy(self._tokenizer),
                            mutated_choice,
                            return_dict,
                            ckpt_holder,
                            kwargs,
                        ),
                    )
                    # Start process.
                    p.start()
                    processses.append(p)
                for p in processses:
                    p.join()
                # Collect results.
                for key in list(return_dict.keys()):
                    all_populations = np.append(all_populations, return_dict[key])
                child_id += num_parallel_workers

            # Add the top_k (5) indices into the history.
            sorted_current_gen_populations = self._sort_results_with_criterion(
                all_populations[-n_childs:], criterion, **kwargs
            )
            new_best_childs = [
                sorted_current_gen_populations[idx] for idx in range(top_k)
            ]
            history += new_best_childs

            # Now, remove the initial first $n_child$ archs.
            all_populations = all_populations[n_childs:]
        return np.asarray(history)
