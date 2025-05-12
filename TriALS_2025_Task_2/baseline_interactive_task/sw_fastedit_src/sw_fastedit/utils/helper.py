from __future__ import annotations

import gc
import logging
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from functools import wraps
from typing import List

import cupy as cp
import pandas as pd
import psutil
import SimpleITK
import torch
from monai.data.meta_tensor import MetaTensor
from pynvml import (
    NVMLError,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)

logger = logging.getLogger("sw_fastedit")

gb_divisor = 1024**3
memory_unit = "GB"


def get_actual_cuda_index_of_device(device: torch.device):
    try:
        cuda_indexes = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    except KeyError:
        return int(device.index)
    return int(cuda_indexes[device.index])


def gpu_usage(device: torch.device, used_memory_only=False, nvml_handle=None):
    shutdown = False
    cuda_index = get_actual_cuda_index_of_device(device)
    if nvml_handle is None:
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(cuda_index)
        shutdown = True
    else:
        h = nvml_handle

    try:
        info = nvmlDeviceGetMemoryInfo(h)
        util = nvmlDeviceGetUtilizationRates(h)
        nv_total, nv_free, nv_used = (
            info.total / gb_divisor,
            info.free / gb_divisor,
            info.used / gb_divisor,
        )
        util_gpu = util.gpu / 100
        util_memory = util.memory / 100

        torch_reserved = torch.cuda.memory_reserved(device) / gb_divisor

        with cp.cuda.Device(device.index):
            mempool = cp.get_default_memory_pool()
            cupy_total = mempool.total_bytes() / gb_divisor
            cupy_used = mempool.used_bytes() / gb_divisor
    except NVMLError:
        return []
    finally:
        if shutdown:
            nvmlShutdown()

    if not used_memory_only:
        return (cuda_index, util_gpu, util_memory, nv_total, nv_free, nv_used, torch_reserved, cupy_total, cupy_used)
    else:
        return nv_used


def gpu_usage_per_process(device: torch.device, nvml_handle=None) -> List:
    cuda_index = get_actual_cuda_index_of_device(device)
    if nvml_handle is None:
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(cuda_index)
        shutdown = True
    else:
        h = nvml_handle
        shutdown = False

    try:
        process_list = []
        for proc in nvmlDeviceGetComputeRunningProcesses(h):
            process_list.append(
                (
                    cuda_index,
                    psutil.Process(proc.pid).name(),
                    proc.usedGpuMemory / gb_divisor,
                )
            )
        if shutdown:
            nvmlShutdown()
    except NVMLError:
        return []
    finally:
        if shutdown:
            nvmlShutdown()

    return process_list


def get_gpu_usage(
    device: torch.device,
    used_memory_only=False,
    context="",
    csv_format=False,
    nvml_handle=None,
):
    if device.type == "cpu":
        # important for the sw_cpu_output flag
        return ""

    (cuda_index, util_gpu, util_memory, nv_total, nv_free, nv_used, torch_reserved, cupy_total, cupy_used) = gpu_usage(
        device=device, nvml_handle=nvml_handle
    )
    usage = ""

    if csv_format and used_memory_only:
        raise NotImplementedError

    if csv_format:
        header = (
            "device,context,time,gpu util (%),memory util (%),total memory ({0}),"
            "free memory ({0}),used memory ({0}),memory reserved by torch ({0}),cupy total ({0}),cupy used ({0})"
        ).format(memory_unit)
        usage += "{},{},{},{:.2f},{:.2f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}".format(
            cuda_index,
            context,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            util_gpu,
            util_memory,
            nv_total,
            nv_free,
            nv_used,
            torch_reserved,
            cupy_total,
            cupy_used,
        )
        return (header, usage)
    else:
        if used_memory_only:
            usage += "{} Device: {} --- used:  {:.0f} {}".format(context, cuda_index, nv_used, memory_unit)
        else:
            usage += "\ndevice: {} context: {}\ngpu util (%):{:.2f} memory util (%): {:.2f}\n".format(
                cuda_index, context, util_gpu, util_memory
            )
            usage += (
                "total memory ({0}): {1:.0f} free memory ({0}): {2:.0f} used memory ({0}): {3:.0f}"
                "memory reserved by torch ({0}): {4:.0f} cupy total ({0}): {5:.0f}\n"
            ).format(memory_unit, nv_total, nv_free, nv_used, torch_reserved, cupy_total)
    return usage


def print_tensor_gpu_usage(a: torch.Tensor):
    if a.cuda:
        logger.info("Tensor GPU memory: {} MB".format(a.element_size() * a.nelement() / (1024**2)))
    else:
        logger.info("Tensor is not on the GPU.")


def print_all_tensor_gpu_memory_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                logger.info(type(obj), obj.size())
        except Exception:
            pass


def print_amount_of_tensors():
    """
    Care this function can lead to unexpected crashed, I guess due to accessing deallocated memory.
    Please use this only for debugging purposes!
    """
    counter = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                counter += 1
        except Exception:
            pass
    print(f"#################################### Amount of tensors: {counter}")
    return counter


def get_total_size_of_all_tensors(data):
    size = 0
    if type(data) is dict:
        for key in data:
            size += get_total_size_of_all_tensors(data[key])
    elif type(data) is list:
        for element in data:
            size += get_total_size_of_all_tensors(element)
    elif type(data) is torch.Tensor or type(data) is MetaTensor:
        size += data.element_size() * data.nelement()

    return size


def describe(t: torch.Tensor):
    """
    Analouge to Pandas describe function, this function prints similiar statistics for torch Tensors.
    """
    return "mean: {} \nmin: {}\nmax: {} \ndtype: {} \ndevice: {}".format(
        torch.mean(t), torch.min(t), torch.max(t), t.dtype, t.device
    )


def describe_batch_data(batchdata: dict, total_size_only=False):
    batch_data_string = ""
    if total_size_only:
        batch_data_string += (
            f"Total size of all tensors in batch data: {get_total_size_of_all_tensors(batchdata)/ (1024**2)} MB\n"
        )
    else:
        batch_data_string += f"Type of batch data: {type(batchdata)}\n"
        for key in batchdata:
            if type(batchdata[key]) is torch.Tensor or type(batchdata[key]) is MetaTensor:
                extra_info = torch.unique(batchdata[key]).tolist()
                if len(extra_info) > 20:
                    extra_info = f"too many unique (len): {len(extra_info)}"

                batch_data_string += (
                    f"- {key}({batchdata[key].__class__.__qualname__}) size: {batchdata[key].size()} "
                    f"size in MB: {batchdata[key].element_size() * batchdata[key].nelement() / (1024**2)}MB "
                    f"device: {batchdata[key].device} "
                    f"dtype: {batchdata[key].dtype} "
                    f"sum: {torch.sum(batchdata[key])} "
                    f"unique values: {extra_info}"
                    "\n"
                )
            if type(batchdata[key]) is MetaTensor:
                batch_data_string += f"  Meta: {batchdata[key].meta}\n" ""
            elif type(batchdata[key]) is dict:
                batch_data_string += f"- {key}(dict)\n"
                for key2 in batchdata[key]:
                    if type(batchdata[key][key2]) is torch.Tensor or type(batchdata[key][key2]) is MetaTensor:
                        batch_data_string += (
                            f"    - {key}/{key2}(Tensor/MetaTensor) "
                            f"size: {batchdata[key][key2].size()} "
                            f"size in MB: {batchdata[key][key2].element_size() * batchdata[key][key2].nelement() / (1024**2)}MB "
                            f"device: {batchdata[key][key2].device} "
                            f"dtype: {batchdata[key][key2].dtype}\n"
                        )
                    else:
                        batch_data_string += f"    - {key}/{key2}: {batchdata[key][key2]} \n"
            elif type(batchdata[key]) is list:
                batch_data_string += f"- {key}(list)\n"
                for item in batchdata[key]:
                    batch_data_string += f"    - {item}\n"
            else:
                batch_data_string += f"- {key}({type(batchdata[key])})\n"
    return batch_data_string


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        # try:
        #     device = args[0].device
        # except AttributeError:
        #     device = None

        # if device is not None:
        #     gpu1 = gpu_usage(device=device, used_memory_only=True)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        # if device is not None:
        #     gpu2 = gpu_usage(device=device, used_memory_only=True)
        total_time = end_time - start_time
        name = None
        try:
            name = func.__qualname__
        except AttributeError:
            # Excepting it to be an object now
            try:
                name = func.__class__.__qualname__
            except AttributeError:
                logger.error("Timeit Wrapper got unexpected element (not func, class or object). Please fix!")
            pass
        # if device is not None:
        #     logger.info(f'Function {name}() took {total_time:.3f} seconds and reserved {(gpu2 - gpu1) / 1024**2:.1f} MB GPU memory')
        # else:
        logger.debug(f"{name}() took {total_time:.3f} seconds")

        return result

    return timeit_wrapper


def get_git_information():
    stream = os.popen("git branch;git rev-parse HEAD")
    git_info = stream.read()
    return git_info


class TerminationHandler:
    def __init__(self, args, tb_logger, wp, gpu_thread):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.args = args
        self.tb_logger = tb_logger
        self.wp = wp
        self.gpu_thread = gpu_thread

    def exit_gracefully(self, *args):
        logger.critical("#### RECEIVED TERM SIGNAL - ABORTING RUN ############")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        if self.wp is not None:
            logger.info(f"\n{self.wp.get_times_summary_pd()}")
        self.cleanup()
        self.join_threads()
        sys.exit(99)

    def join_threads(self):
        if self.tb_logger is not None:
            self.tb_logger.close()
        self.gpu_thread.stopFlag.set()
        self.gpu_thread.join()

    def cleanup(self):
        logger.info(f"#### LOGGED ALL DATA TO {self.args.output_dir} ############")
        # Cleanup
        if self.args.throw_away_cache:
            logger.info(f"Cleaning up the cache dir {self.args.cache_dir}")
            shutil.rmtree(self.args.cache_dir, ignore_errors=True)
        else:
            logger.info("Leaving cache dir as it is..")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logger.critical(torch.cuda.memory_summary())


class GPU_Thread(threading.Thread):
    def __init__(self, thread_id: int, name: str, output_file: str, device: torch.device):
        super().__init__(daemon=True)
        self.thread_id = thread_id
        self.name = name
        self.device = device
        self.csv_file = open(f"{output_file}", "w")
        header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
        self.csv_file.write(header)
        self.csv_file.write("\n")
        self.csv_file.flush()
        self.stopFlag = threading.Event()

        nvmlInit()
        cuda_index = get_actual_cuda_index_of_device(self.device)
        self.nvml_handle = nvmlDeviceGetHandleByIndex(cuda_index)

    def __del__(self):
        self.csv_file.flush()
        self.csv_file.close()

    def run(self):
        while not self.stopFlag.wait(1):
            header, usage = get_gpu_usage(
                self.device,
                used_memory_only=False,
                context="",
                csv_format=True,
                nvml_handle=self.nvml_handle,
            )
            self.csv_file.write(usage)
            self.csv_file.write("\n")
            self.csv_file.flush()
            # print(gpu_usage_per_process(self.device, nvml_handle=self.nvml_handle))


def get_tensor_at_coordinates(t: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    assert len(coordinates) == len(t.shape)
    if len(coordinates) == 4:
        assert coordinates.shape == (4, 2)
        return t[
            coordinates[0, 0] : coordinates[0, 1],
            coordinates[1, 0] : coordinates[1, 1],
            coordinates[2, 0] : coordinates[2, 1],
            coordinates[3, 0] : coordinates[3, 1],
        ]
    elif len(coordinates) == 3:
        assert coordinates.shape == (3, 2)
        return t[
            coordinates[0, 0] : coordinates[0, 1],
            coordinates[1, 0] : coordinates[1, 1],
            coordinates[2, 0] : coordinates[2, 1],
        ]
    else:
        raise UserWarning("Not implemented for this lenghts of coordinates")


def get_global_coordinates_from_patch_coordinates(
    current_coordinates: List, patch_coordinates: torch.Tensor
) -> torch.Tensor:
    assert len(current_coordinates) == len(patch_coordinates)
    assert patch_coordinates.shape == (len(current_coordinates), 2)
    # Start at the second entry since the first contains other information
    for _ in range(1, len(current_coordinates)):
        current_coordinates[_] += patch_coordinates[_, 0]
    return current_coordinates


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_mha_to_nii(mha_input_path, nii_out_path):
    img = SimpleITK.ReadImage(mha_input_path)
    # reader = sitk.ImageFileReader()

    SimpleITK.WriteImage(img, nii_out_path, True)


def convert_nii_to_mha(nii_input_path, mha_out_path):
    img = SimpleITK.ReadImage(nii_input_path)

    SimpleITK.WriteImage(img, mha_out_path, True)


def is_docker():
    path = "/proc/self/cgroup"
    return os.path.exists("/.dockerenv") or os.path.isfile(path) and any("docker" in line for line in open(path))
