# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology #
# zdravko.marinov@kit.edu #
# Further code extension and modification by B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology #
# matthiashadlich@posteo.de #

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import torch
from ignite.engine import Events
from monai.engines.utils import IterationEvents
from monai.utils.profiling import ProfileHandler, WorkflowProfiler

from sw_fastedit.api import get_trainer, oom_observer
from sw_fastedit.utils.argparser import parse_args, setup_environment_and_adapt_args
from sw_fastedit.utils.helper import GPU_Thread, TerminationHandler, get_gpu_usage
from sw_fastedit.utils.tensorboard_logger import init_tensorboard_logger

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.") # Ignore this verbose warning 

logger = logging.getLogger("sw_fastedit")


def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")



    gpu_thread = GPU_Thread(1, "Track_GPU_Usage", os.path.join(args.output_dir, "usage.csv"), device)
    logger.info(f"Logging GPU usage to {args.output_dir}/usage.csv")

    try:
        wp = WorkflowProfiler()
        (
            trainer,
            evaluator,
            key_train_metric,
            additional_train_metrics,
            key_val_metric,
            additional_val_metrics,
        ) = get_trainer(args, resume_from=args.resume_from)
        train_metric_names = list(key_train_metric.keys()) + list(additional_train_metrics.keys())
        val_metric_names = list(key_val_metric.keys()) + list(additional_val_metrics.keys())

        tb_logger = init_tensorboard_logger(
            trainer,
            evaluator,
            trainer.optimizer,
            train_metric_names,
            val_metric_names,
            network=trainer.network,
            output_dir=args.output_dir,
        )

        gpu_thread.start()
        terminator = TerminationHandler(args, tb_logger, wp, gpu_thread)

        with tb_logger:
            with wp:
                start_time = time.time()
                for t, name in [(trainer, "trainer"), (evaluator, "evaluator")]:
                    for event in [
                        ["Epoch", wp, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED],
                        [
                            "Iteration",
                            wp,
                            Events.ITERATION_STARTED,
                            Events.ITERATION_COMPLETED,
                        ],
                        [
                            "Batch generation",
                            wp,
                            Events.GET_BATCH_STARTED,
                            Events.GET_BATCH_COMPLETED,
                        ],
                        [
                            "Inner Iteration",
                            wp,
                            IterationEvents.INNER_ITERATION_STARTED,
                            IterationEvents.INNER_ITERATION_COMPLETED,
                        ],
                        ["Whole run", wp, Events.STARTED, Events.COMPLETED],
                    ]:
                        event[0] = f"{name}: {event[0]}"
                        ProfileHandler(*event).attach(t)

                try:
                    if not args.eval_only:
                        trainer.run()
                    else:
                        evaluator.run()
                except torch.cuda.OutOfMemoryError:
                    oom_observer(device, None, None, None)
                    logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))

                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        # Got a cuDNN error
                        pass
                    oom_observer(device, None, None, None)
                    logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
                finally:
                    logger.info("Total Training Time {}".format(time.time() - start_time))
    finally:
        terminator.cleanup()
        terminator.join_threads()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        logger.info(f"\n{wp.get_times_summary_pd()}")


def main():
    global logger

    args = parse_args()
    args, logger = setup_environment_and_adapt_args(args)

    run(args)


if __name__ == "__main__":
    main()
