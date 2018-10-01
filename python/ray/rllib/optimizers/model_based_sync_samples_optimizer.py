from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.utils.filter import RunningStat
from ray.rllib.utils.timer import TimerStat


class ModelBased_SyncSamplesOptimizer(PolicyOptimizer):
    """A simple synchronous RL optimizer.

    In each step, this optimizer pulls samples from a number of remote
    evaluators, concatenates them, and then updates a local model. The updated
    model weights are then broadcast to all remote evaluators.
    """

    def _init(self, num_sgd_iter=1, train_batch_size=1):
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.throughput = RunningStat()
        self.num_sgd_iter = num_sgd_iter
        self.train_batch_size = train_batch_size
        self.learner_stats = {}

    def set_env_model(self, env_model):
        # TODO: add validation check
        self.env_model = env_model

    def step(self):
        with self.update_weights_timer:
            if self.remote_evaluators:
                weights = ray.put(self.local_evaluator.get_weights())
                for e in self.remote_evaluators:
                    e.set_weights.remote(weights)

        with self.sample_timer:
            samples = []
            while sum(s.count for s in samples) < self.train_batch_size:
                if self.remote_evaluators:
                    samples.extend(
                        ray.get([
                            e.sample.remote() for e in self.remote_evaluators
                        ]))
                else:
                    samples.append(self.local_evaluator.sample())
            samples = SampleBatch.concat_samples(samples)
            self.sample_timer.push_units_processed(samples.count)
            # print("\n\nhkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk samples", samples.keys())
            # print("hkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk samples obs", samples["obs"].shape)
            # print("hkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk samples new_obs", samples["new_obs"].shape)
            # print("hkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk samples actions", samples["actions"].shape)
            # import numpy
            # transition_state = numpy.stack([samples["obs"], samples["new_obs"]],  axis=2)
            # print("hkkkkk ", transition_state.shape)
            # print("hkkkkk ", transition_state.reshape(transition_state.shape[0], -1).shape)

            # print("samples[obs]", samples["obs"])
            # print("samples[actions]", samples["actions"])

        with self.grad_timer:

            new_samples = self.env_model.process(samples)
            # print("new_samples[obs]", new_samples["obs"])
            # print("new_samples[actions]", new_samples["actions"])
            for i in range(self.num_sgd_iter):
                fetches = self.local_evaluator.compute_apply(new_samples)
                if "stats" in fetches:
                    self.learner_stats = fetches["stats"]
                if self.num_sgd_iter > 1:
                    print(i, fetches)
            # self.grad_timer.push_units_processed(new_samples.count)
            self.grad_timer.push_units_processed(len(samples["obs"]))

        # self.num_steps_sampled += new_samples.count
        # self.num_steps_trained += new_samples.count
        self.num_steps_sampled += len(samples["obs"])
        self.num_steps_trained += len(samples["obs"])
        return fetches

    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean,
                                        3),
                "opt_peak_throughput": round(self.grad_timer.mean_throughput,
                                             3),
                "sample_peak_throughput": round(
                    self.sample_timer.mean_throughput, 3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })
