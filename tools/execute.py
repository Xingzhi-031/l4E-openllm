import argparse
import os, os.path as osp
import gc
import sys
try:
    import resource
except ImportError:
    resource = None
import concurrent.futures
import time
import platform
import random
from copy import deepcopy
import contextlib
import multiprocessing
import pickle
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

"""Adapted from code_eval (@link https://huggingface.co/spaces/evaluate-metric/code_eval)"""


def get_memory_usage():
    return sys.getsizeof(sys.modules[__name__])


@contextlib.contextmanager
def set_memory_limit(maximum_memory_bytes=None):
    try:
        if maximum_memory_bytes is not None and resource is not None:
            _rlimit_data = resource.getrlimit(resource.RLIMIT_DATA)
            current_memory_usage = get_memory_usage()
            memory_limit = int(current_memory_usage + maximum_memory_bytes)
            max_allowed_memory = _rlimit_data[1]
            if memory_limit > max_allowed_memory:
                memory_limit = max_allowed_memory
            resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, _rlimit_data[1]))
        yield
    finally:
        if maximum_memory_bytes is not None and resource is not None:
            resource.setrlimit(resource.RLIMIT_DATA, _rlimit_data)

class TimeoutException(Exception):
    pass

def timeout_signal_handler(signum, frame):
    raise TimeoutException("Timed out!")

@contextlib.contextmanager
def set_time_limit(seconds):
    import signal
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, timeout_signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

import io

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise OSError
    def readline(self, *args, **kwargs):
        raise OSError
    def readlines(self, *args, **kwargs):
        raise OSError
    def readable(self, *args, **kwargs):
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

@contextlib.contextmanager
def create_tempdir():
    import tempfile
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

@contextlib.contextmanager
def reliability_guard():
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the testcode (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    with create_tempdir():
        with swallow_io():
            try:

                import faulthandler

                faulthandler.disable()

                import builtins, os, shutil, subprocess

                os.environ["OMP_NUM_THREADS"] = "1"

                _keys = dict(
                    builtins = ('exit', 'quit'),
                    os = ('kill', 'system', 'putenv', 'remove', 'removedirs', 'rmdir', 'fchdir', 'setuid', 'fork', 'forkpty', 'killpg', 'rename', 'renames', 'truncate', 'replace', 'unlink', 'fchmod', 'fchown', 'chmod', 'chown', 'chroot', 'lchflags', 'lchmod', 'lchown', 'getcwd', 'chdir'),
                    shutil = ('rmtree', 'move', 'chown'),
                    subprocess = ('Popen',),
                )
                _baks = dict()
                for lib, keys in _keys.items():
                    obj = locals()[lib]
                    _bak = dict()
                    for key in keys:
                        if hasattr(obj, key):
                            _bak[key] = getattr(obj, key)
                    _baks[lib] = _bak

                #__builtins__["help"] = None

                yield
            finally:
                for lib, keys in _keys.items():
                    obj = locals()[lib]
                    for key, val in _baks[lib].items():
                        setattr(obj, key, val)

def unsafe_execute(program: str, exec_globals: dict):
    try:
        gc_bak = gc.isenabled()
        gc.disable()
        with reliability_guard():
            exec(program, exec_globals)
    finally:
        if gc_bak:
            gc.enable()

def unsafe_execute2(program: str):
    try:
        gc_bak = gc.isenabled()
        gc.disable()
        output_capture = io.StringIO()
        sys.stdout = output_capture  # 重定向标准输出到StringIO
        with reliability_guard():
            exec(program)
            sys.stdout = sys.__stdout__  # 恢复标准输出
            test_result = output_capture.getvalue()  # 获取exec执行过程中产生的输出
            return test_result
    finally:
        if gc_bak:
            gc.enable()

def unsafe_execute_easy(program: str):
    try:
        gc_bak = gc.isenabled()
        gc.disable()
        with reliability_guard():
            exec(program)
    finally:
        if gc_bak:
            gc.enable()

def unsafe_timed_execute(program: str, exec_globals: dict, time_limit_seconds: float):
    def run_program():
        exec(program, exec_globals)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        gc_bak = gc.isenabled()
        gc.disable()
        future = executor.submit(run_program)
        try:
            future.result(timeout=time_limit_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutException
            # print(f"Execution timed out after {time_limit_seconds} seconds")
        except Exception as e:
            raise e
        finally:
            if gc_bak:
                gc.enable()
# def unsafe_timed_execute(program: str, exec_globals: dict, maximum_memory_bytes: float, time_limit_seconds: float):
#     try:
#         gc_bak = gc.isenabled()
#         gc.disable()
#         with reliability_guard():
#             with set_memory_limit(maximum_memory_bytes):
#                 with set_time_limit(time_limit_seconds):
#                     exec(program, exec_globals)
#     finally:
#         if gc_bak:
#             gc.enable()
