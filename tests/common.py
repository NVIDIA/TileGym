# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import collections
import dataclasses
import functools
import gc
import inspect
import itertools
import multiprocessing
import numbers
import os
import pathlib
import random
import unittest

import pytest

# import pandas as pd
import torch
from functools import wraps
from .config import Config

current_process = multiprocessing.current_process()
if current_process.name == 'MainProcess' and not Config.quiet:
    print(Config.args)
    print(f'seed = {Config.seed}')
    print(f'torch = {torch.__version__}')
    if torch.cuda.is_available():
        print(f'device = {torch.cuda.get_device_name()}')


def get_dtype_tolerances(dtype):
    """
    Get default relative and absolute tolerances based on data type.

    Args:
        dtype: torch.dtype - The data type of tensors being compared

    Returns:
        dict: Dictionary with 'rtol' and 'atol' keys
    """
    # Define tolerance mappings based on precision
    tolerance_map = {
        # High precision types
        torch.float64: {'rtol': 1e-12, 'atol': 1e-15},
        torch.complex128: {'rtol': 1e-12, 'atol': 1e-15},
        # Standard precision types
        torch.float32: {'rtol': 1e-5, 'atol': 1e-8},
        torch.complex64: {'rtol': 1e-5, 'atol': 1e-8},
        # Half precision types
        torch.float16: {'rtol': 1e-2, 'atol': 1e-2},
        torch.bfloat16: {'rtol': 1e-2, 'atol': 2e-2},
        torch.complex32: {'rtol': 1e-2, 'atol': 1e-2},
        # 8-bit float types (lower precision)
        torch.float8_e4m3fn: {'rtol': 1e-1, 'atol': 1e-1},
        torch.float8_e5m2: {'rtol': 5e-1, 'atol': 5e-1},
        # Integer types (exact comparison)
        torch.int8: {'rtol': 0, 'atol': 0},
        torch.int16: {'rtol': 0, 'atol': 0},
        torch.int32: {'rtol': 0, 'atol': 0},
        torch.int64: {'rtol': 0, 'atol': 0},
        torch.uint8: {'rtol': 0, 'atol': 0},
        torch.bool: {'rtol': 0, 'atol': 0},
    }

    # Return specific tolerance or default for unknown types
    return tolerance_map.get(dtype, {'rtol': 1e-5, 'atol': 1e-8})


def get_location(offset=2):
    frame = inspect.currentframe()
    for _ in range(offset):
        frame = frame.f_back
    location = f'{frame.f_code.co_filename}:{frame.f_lineno}'
    return location


def get_tensor_alignment(tensor):
    address = tensor.data_ptr()
    alignment = (((address - 1) ^ address) + 1) >> 1
    return alignment


class PyTestCase:
    r'''
    Base class for TileGym unit tests.
    '''

    @pytest.fixture(autouse=True)
    def setup_test(self, request):
        self.request = request
        self.test_name = request.node.name
        self.use_csv = Config.csv
        if self.use_csv:
            self.file = Config.file

    def setUp(self):
        r'''
        Automatically resets random seed to the value provided in config before
        running each test case.
        '''
        torch.manual_seed(Config.seed)
        random.seed(Config.seed)

    def __str__(self):
        cls = self.__class__
        return f'{cls.__module__}.{cls.__qualname__}.{self._testMethodName}'

    def assertCorrectness(
        self,
        test_fn,
        ref_fn,
        kwargs,
        extra_test_kwargs=None,
        extra_ref_kwargs=None,
        gradient=None,
        rtol=None,
        atol=None,
        equal_nan=False,
        check_stride=True,
        multiple_outputs=False,
        test_index=None,
        ref_index=None,
        output_processor=None,
    ):
        r'''
        Check that a specified test function matches the reference.

        Relative and absolute tolerance for comparing gradients of inputs w.r.t
        the reference can be customized for each input tensor.

        Args:
            test_fn: test function
            ref_fn: reference function
            kwargs: keyword arguments common for both test and ref function
            extra_test_kwargs: optional extra keyword arguments for test
                function
            extra_ref_kwargs: optional extra keyword arguments for reference
                function
            gradient: optional input gradient w.r.t. the first output of
                functions
            rtol: relative tolerance for output of forward, and default
                relative tolerance for gradients of inputs. If None, will be
                auto-detected based on the data type of input tensors.
            atol: absolute tolerance for output of forward, and default
                absolute tolerance for gradients of inputs. If None, will be
                auto-detected based on the data type of input tensors.
            equal_nan: if ``True``, then two ``NaNs`` will be considered equal
            check_stride: if ``True``, then check strides of all compared
                tensors
            multiple_outputs: if ``True`` then model has multiple outputs which
                will be iterated through to confirm correctness.  Can only
                be used on Forward, will error if inputs require gradient.
            test_index: Compare a specific tensor from the test output tuple
                at this index incase of multiple outputs. Default is None
            ref_index: Compare a specific tensor from the reference output tuple
                at this index incase of multiple outputs. Default is None
        '''
        passed = True
        all_msgs = []
        failed_msgs = []

        if extra_test_kwargs is None:
            extra_test_kwargs = {}

        if extra_ref_kwargs is None:
            extra_ref_kwargs = {}

        # Auto-detect tolerances based on data type if not provided
        if rtol is None or atol is None:
            # Find the first tensor in kwargs to determine dtype
            detected_dtype = None
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    detected_dtype = value.dtype
                    break
                elif isinstance(value, TestParam) and isinstance(value.tensor, torch.Tensor):
                    detected_dtype = value.tensor.dtype
                    break

            # If no tensor found in kwargs, check extra_test_kwargs
            if detected_dtype is None:
                for value in extra_test_kwargs.values():
                    if isinstance(value, torch.Tensor):
                        detected_dtype = value.dtype
                        break
                    elif isinstance(value, TestParam) and isinstance(value.tensor, torch.Tensor):
                        detected_dtype = value.tensor.dtype
                        break

            # Get default tolerances for the detected dtype
            if detected_dtype is not None:
                default_tols = get_dtype_tolerances(detected_dtype)
                if rtol is None:
                    rtol = default_tols['rtol']
                if atol is None:
                    atol = default_tols['atol']
            else:
                # Fallback to standard defaults if no dtype detected
                if rtol is None:
                    rtol = 1e-5
                if atol is None:
                    atol = 1e-8

        fn_kwargs = {k: v.tensor if isinstance(v, TestParam) else v for k, v in kwargs.items()}

        tensor_args_with_grad = {
            k: v for k, v in fn_kwargs.items() if isinstance(v, torch.Tensor) and v.requires_grad
        }
        for k, v in extra_test_kwargs.items():
            if k in extra_ref_kwargs and isinstance(v, torch.Tensor) and v.requires_grad:
                assert (
                    isinstance(extra_ref_kwargs[k], torch.Tensor)
                    and extra_ref_kwargs[k].requires_grad
                )
                tensor_args_with_grad[k] = v
        assert not (len(tensor_args_with_grad) and multiple_outputs)
        test_out = test_fn(**fn_kwargs, **extra_test_kwargs)
        ref_out = ref_fn(**fn_kwargs, **extra_ref_kwargs)

        if test_index is not None:
            test_out = test_out[test_index]

        if ref_index is not None:
            ref_out = ref_out[ref_index]

        if isinstance(ref_out, torch.Tensor):
            ref_check = [ref_out]
            test_check = [test_out]
        else:
            ref_check = ref_out
            test_check = test_out
        assert len(test_check) == len(ref_check)
        for ind, (ro, to) in enumerate(zip(ref_check, test_check)):
            if output_processor is not None:
                to = output_processor(ind, to, fn_kwargs, extra_test_kwargs, extra_ref_kwargs)
                ro = output_processor(ind, ro, fn_kwargs, extra_test_kwargs, extra_ref_kwargs)
            out_close, msg = compare_tensors(to, ro, rtol, atol, equal_nan, check_stride)
            if not out_close:
                passed = False
                prefix = (
                    f'*** OUTPUT {ind} DID NOT MATCH THE REFERENCE '
                    f'(rtol={rtol}, atol={atol}) ***'
                )
                failed_msgs.append(prefix)
                failed_msgs.extend(msg)
            else:
                prefix = (
                    f'*** OUTPUT {ind} MATCHED THE REFERENCE ' f'(rtol={rtol}, atol={atol}) ***'
                )
            all_msgs.append(prefix)
            all_msgs.extend(msg)
        if not multiple_outputs and test_out.requires_grad:
            if gradient is None:
                gradient = torch.ones_like(test_out)
            elif callable(gradient):
                gradient = gradient(test_out)

            ref_out.backward(gradient)

            ref_grads = {
                name: arg.grad.detach().clone() for name, arg in tensor_args_with_grad.items()
            }

            for arg in tensor_args_with_grad.values():
                arg.grad = None

            test_out.backward(gradient)

            test_grads = {
                name: arg.grad.detach().clone() for name, arg in tensor_args_with_grad.items()
            }

            for arg in tensor_args_with_grad.values():
                arg.grad = None

            for name in ref_grads:
                if name in kwargs:
                    d = kwargs
                else:
                    d = extra_test_kwargs
                if isinstance(d[name], TestParam):
                    grad_rtol = d[name].rtol
                    grad_atol = d[name].atol
                else:
                    grad_rtol = rtol
                    grad_atol = atol

                grad_close, msg = compare_tensors(
                    test_grads[name],
                    ref_grads[name],
                    grad_rtol,
                    grad_atol,
                    equal_nan,
                    check_stride,
                )

                if not grad_close:
                    passed = False
                    prefix = (
                        f'*** GRAD FOR: {name} DID NOT MATCH THE REFERENCE '
                        f'(rtol={grad_rtol}, atol={grad_atol}) ***'
                    )
                    failed_msgs.append(prefix)
                    failed_msgs.extend(msg)
                else:
                    prefix = (
                        f'*** GRAD FOR: {name} MATCHED THE REFERENCE '
                        f'(rtol={grad_rtol}, atol={grad_atol}) ***'
                    )
                all_msgs.append(prefix)
                all_msgs.extend(msg)

        all_msgs = '\n'.join([f'\t{msg}' for msg in all_msgs])
        failed_msgs = '\n'.join([f'\t{msg}' for msg in failed_msgs])

        assert passed, f'\n{failed_msgs}'

        if Config.print_matching:
            test = self._subtest if self._subtest is not None else self
            if not Config.verbose:
                print(test)
            print(all_msgs)

    def assertDeterministic(
        self,
        fn,
        kwargs,
        gradient=None,
        rtol=None,
        atol=None,
        equal_nan=False,
        iters=100,
    ):
        r'''
        Checks that a specified function returns deterministic results.

        On first iteration it stores returned output and gradients w.r.t. input
        arguments as reference values, in all subsequent iterations it compares
        outputs and gradients with the reference.

        Relative and absolute tolerance for comparing gradients of inputs w.r.t
        the reference can be customized for each input tensor.

        Example::

            self.assertDeterministic(
                test_fn
                {
                    'inp1': common.TestParam(tensor1, rtol=1e-3, atol=1e-3),
                    'inp2': tensor2,
                },
            )

        Set ``rtol`` and ``atol`` to zero to test for bitwise identical
        results.

        Args:
            fn: function to be benchmarked
            kwargs: dictionary of keyword arguments for ``fn``
            gradient: optional input gradient w.r.t. the first output of ``fn``
            rtol: relative tolerance for output of forward, and default
                relative tolerance for gradients of inputs
            atol: absolute tolerance for output of forward, and default
                absolute tolerance for gradients of inputs
            equal_nan: if ``True``, then two ``NaNs`` will be considered equal
            iters: number of test iterations to perform
        '''
        passed = True

        # Auto-detect tolerances based on data type if not provided
        if rtol is None or atol is None:
            # Find the first tensor in kwargs to determine dtype
            detected_dtype = None
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    detected_dtype = value.dtype
                    break
                elif isinstance(value, TestParam) and isinstance(value.tensor, torch.Tensor):
                    detected_dtype = value.tensor.dtype
                    break

            # Get default tolerances for the detected dtype
            if detected_dtype is not None:
                default_tols = get_dtype_tolerances(detected_dtype)
                if rtol is None:
                    rtol = default_tols['rtol']
                if atol is None:
                    atol = default_tols['atol']
            else:
                # Fallback to standard defaults if no dtype detected
                if rtol is None:
                    rtol = 1e-5
                if atol is None:
                    atol = 1e-8

        ref_out = None
        ref_grads = None

        fn_kwargs = {k: v.tensor if isinstance(v, TestParam) else v for k, v in kwargs.items()}

        tensor_args_with_grad = {
            k: v
            for k, v in fn_kwargs.items()
            if isinstance(v, (torch.Tensor, TestParam)) and v.requires_grad
        }

        for i in range(iters):
            out = fn(**fn_kwargs)

            if ref_out is None:
                ref_out = out
            else:
                out_close = torch.allclose(out, ref_out, rtol, atol, equal_nan)
                if not out_close:
                    print(
                        '*** OUTPUT DID NOT MATCH THE REFERENCE ' f'(rtol={rtol}, atol={atol}) ***'
                    )
                    passed = False

            if out.requires_grad:
                if gradient is None:
                    gradient = torch.ones_like(out)
                out.backward(gradient)

                if ref_grads is None:
                    ref_grads = {
                        name: arg.grad.detach().clone()
                        for name, arg in tensor_args_with_grad.items()
                    }
                else:
                    current_grads = {name: arg.grad for name, arg in tensor_args_with_grad.items()}

                    for name in ref_grads:
                        if isinstance(kwargs[name], TestParam):
                            grad_rtol = kwargs[name].rtol
                            grad_atol = kwargs[name].atol
                        else:
                            grad_rtol = rtol
                            grad_atol = atol

                        grad_close = torch.allclose(
                            current_grads[name],
                            ref_grads[name],
                            grad_rtol,
                            grad_atol,
                            equal_nan,
                        )
                        if not grad_close:
                            print(
                                f'*** GRAD FOR: {name} DID NOT MATCH THE REFERENCE '
                                f'(rtol={grad_rtol}, atol={grad_atol}) ***'
                            )
                            passed = False

                for arg in tensor_args_with_grad.values():
                    arg.grad = None

            if not passed:
                break

        assert passed

    def assertAllClose(
        self,
        input,
        reference,
        rtol=None,
        atol=None,
        equal_nan=False,
        check_stride=True,
    ):
        # Auto-detect tolerances based on data type if not provided
        if rtol is None or atol is None:
            detected_dtype = input.dtype if isinstance(input, torch.Tensor) else None
            if detected_dtype is not None:
                default_tols = get_dtype_tolerances(detected_dtype)
                if rtol is None:
                    rtol = default_tols['rtol']
                if atol is None:
                    atol = default_tols['atol']
            else:
                # Fallback to standard defaults
                if rtol is None:
                    rtol = 1e-5
                if atol is None:
                    atol = 1e-8

        allclose, msgs = compare_tensors(input, reference, rtol, atol, equal_nan, check_stride)
        assert allclose, msgs

    def assertDiffsClose(self, input, reference_fp16, reference_fp32, tolerance):
        input_fp32_diff = (input - reference_fp32).abs()
        max_input_fp32_diff = input_fp32_diff.max().item()

        ref_fp16_fp32_diff = (reference_fp16 - reference_fp32).abs()
        max_ref_fp16_fp32_diff = ref_fp16_fp32_diff.max().item()

        diffs_close = max_input_fp32_diff <= tolerance * max_ref_fp16_fp32_diff

        close_mask = input_fp32_diff <= tolerance * ref_fp16_fp32_diff
        not_close_mask = close_mask.logical_not()
        close_count = close_mask.sum().item()
        total_count = input.numel()
        close_percent = close_count / total_count * 100

        msg = (
            f'\nClose: {close_count} / {total_count} [{close_percent:.2f}%],'
            f'\nMax Input Ref-fp32 difference: {max_input_fp32_diff:.4e}'
            f'\nMax Ref-fp16 Ref-fp32 difference: {max_ref_fp16_fp32_diff:.4e}'
            f'\nScaled Max Ref-fp16 Ref-fp32 difference: {tolerance * max_ref_fp16_fp32_diff:.4e}'
            f'\nMin Tolerance Possible: {max_input_fp32_diff / max_ref_fp16_fp32_diff:.4e}'
            f'\nMismatched indices:\n{not_close_mask.nonzero()}'
        )

        if Config.print_matching:
            test = self._subtest if self._subtest is not None else self
            verbose_msg = f'{test}\ndiffsclose: {diffs_close}{msg}\n'
            print(verbose_msg)
        assert diffs_close, msg


def save_arguments():

    def outer_wrapper(func):

        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            self, *rest = args
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            arguments = collections.OrderedDict(bound_arguments.arguments)
            arguments.popitem(last=False)
            self.arguments = arguments
            func(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def unroll_generators(params):
    unrolled = []
    for param in params:
        generators = {k: v for k, v in param.items() if inspect.isgenerator(v)}
        if generators:
            generator_keys = generators.keys()
            for gen_vals in itertools.product(*generators.values()):
                update = zip(generator_keys, gen_vals)
                current_param = param.copy()
                current_param.update(update)
                unrolled.append(current_param)
        else:
            unrolled.append(param)
    return unrolled


def df_write_descriptor(fmt):
    descriptor = {
        'csv': {'method': 'to_csv', 'kwargs': {'index': False}},
        'html': {'method': 'to_html', 'kwargs': {'index': False}},
        'json': {'method': 'to_json', 'kwargs': {'orient': 'records'}},
        'md': {'method': 'to_markdown', 'kwargs': {'index': False}},
        'txt': {'method': 'to_string', 'kwargs': {'index': False}},
    }
    return descriptor[fmt]


def flatten_keys(l0):
    res = {}
    for name, l1 in l0.items():
        for direction, l2 in l1.items():
            selected = {(name, direction, metric): l2[metric] for metric in Config.fields}
            res.update(selected)
    return res


def convert_non_primitive_to_repr(arguments):
    primitive_types = (numbers.Number, bool, str)
    arguments = {k: v if isinstance(v, primitive_types) else repr(v) for k, v in arguments.items()}
    return arguments


def find_modes_metrics(df):
    modes = []
    metrics = []
    for col in df.columns:
        mode = col[1]
        if mode and mode not in modes:
            modes.append(mode)
        metric = col[2]
        if metric and metric not in metrics:
            metrics.append(metric)
    return modes, metrics


def load_previous(name, argument_names, expanded_argument_names):
    load_dir = pathlib.Path(Config.load_dir)
    load_name = pathlib.Path(f'{name}.{Config.load}')
    load_path = load_dir / load_name
    loaded_df = pd.read_csv(
        load_path,
        header=[0, 1, 2],
    )
    loaded_df = loaded_df.rename(columns=lambda x: x if 'Unnamed' not in str(x) else '')

    loaded_df.set_index(expanded_argument_names)
    selected_cols = expanded_argument_names + [
        c for c in loaded_df.columns if c[0] in Config.load_names
    ]

    loaded_df = loaded_df[selected_cols]

    loaded_names = [c[0] for c in loaded_df.columns]
    names2 = [n for n in loaded_names if n not in argument_names]
    loaded_df = loaded_df.rename(columns=lambda x: f'loaded_{x}' if x in names2 else x)
    return loaded_df


def dump_results(df, name):
    dump_dir = pathlib.Path(Config.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    dump_format = Config.dump
    write_desc = df_write_descriptor(dump_format)
    res = getattr(df, write_desc['method'])(**write_desc['kwargs'])
    dump_name = pathlib.Path(f'{name}.{dump_format}')
    dump_path = dump_dir / dump_name
    with open(dump_path, 'w') as f:
        f.write(res)


def add_relative_columns(df, main_name, other_names):
    non_relative_metrics = {'rel_std', 'nrep'}

    modes, metrics = find_modes_metrics(df)

    relative_metrics = [m for m in metrics if m not in non_relative_metrics]

    for other_name, mode, metric in itertools.product(other_names, modes, relative_metrics):
        df[(f'{other_name}/{main_name}', mode, metric)] = (
            df[(f'{other_name}', mode, metric)] / df[(f'{main_name}', mode, metric)]
        )


def print_results(df, name):
    formatters = {}
    for col in df.columns:
        if col[1] in {'forward', 'backward'}:
            if col[2] == 'rel_std':
                formatters[col] = '{:.1f}%'.format
            elif col[2] == 'nrep':
                formatters[col] = '{:d}'.format
            else:
                formatters[col] = '{:.2e}'.format
        if '/' in col[0]:
            formatters[col] = '{:.3f}'.format

    print()
    print(name)
    print(
        df.to_string(
            index=False,
            formatters=formatters,
        )
    )
    print()


@dataclasses.dataclass
class TestParam:
    r'''
    Class to specify per-tensor relative and absolute tolerances.
    '''

    tensor: torch.Tensor
    rtol: float
    atol: float


def compare_tensors(
    test,
    reference,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_stride=True,
    msg_prefix='\t',
):
    # Auto-detect tolerances based on data type if not provided
    if rtol is None or atol is None:
        detected_dtype = test.dtype if isinstance(test, torch.Tensor) else None
        if detected_dtype is not None:
            default_tols = get_dtype_tolerances(detected_dtype)
            if rtol is None:
                rtol = default_tols['rtol']
            if atol is None:
                atol = default_tols['atol']
        else:
            # Fallback to standard defaults
            if rtol is None:
                rtol = 1e-5
            if atol is None:
                atol = 1e-8

    if test.shape != reference.shape:
        msgs = f'shape mismatch, test: {test.shape}, ' f'reference: {reference.shape}'
        raise RuntimeError(msgs)

    if check_stride and test.stride() != reference.stride():
        msgs = f'stride mismatch, test: {test.stride()}, ' f'reference: {reference.stride()}'
        raise RuntimeError(msgs)

    if test.dtype != reference.dtype:
        msgs = f'dtype mismatch, test: {test.dtype}, ' f'reference: {reference.dtype}'
        raise RuntimeError(msgs)

    dtype = test.dtype
    input = test.to(torch.float32)
    reference = reference.to(torch.float32)
    input = torch.where(torch.isnan(reference), float('nan'), input)

    allclose = torch.allclose(input, reference, rtol, atol, equal_nan)

    abs_diff = (input - reference).abs()

    is_close = abs_diff <= (atol + rtol * reference.abs())
    close_mask = is_close == True
    not_close_mask = close_mask.logical_not()
    close_count = close_mask.sum().item()
    total_count = input.numel()
    close_percent = close_count / total_count * 100

    rel_change_denom = reference.abs()
    rel_change = abs_diff / rel_change_denom
    rel_change[rel_change_denom == 0] = 0

    max_mean_change_denom = torch.max(input.abs(), reference.abs())
    max_mean_change = abs_diff / max_mean_change_denom
    max_mean_change[max_mean_change_denom == 0] = 0

    arith_mean_change_denom = 0.5 * (input + reference).abs()
    arith_mean_change = abs_diff / arith_mean_change_denom
    arith_mean_change[arith_mean_change_denom == 0] = 0

    max_abs_diff = abs_diff.max()
    max_rel_change = rel_change.max()
    max_max_mean_change = max_mean_change.max()
    max_arith_mean_change = arith_mean_change.max()

    abs_ref = reference.abs()
    abs_input = input.abs()

    msgs = [
        f'allclose: {allclose}',
        f'matched: {close_count} / {total_count} [{close_percent:.2f}%]',
        f'ref range:    {reference.min():11.4e} : {reference.max():11.4e}',
        f'test range:   {input.min():11.4e} : {input.max():11.4e}',
        f'|ref| range:  {abs_ref.min():11.4e} : {abs_ref.max():11.4e}',
        f'|test| range: {abs_input.min():11.4e} : {abs_input.max():11.4e}',
        f'max absolute difference: {max_abs_diff:11.4e}',
        f'max relative change:     {max_rel_change:11.4e}',
        f'max max mean change:     {max_max_mean_change:11.4e}',
        f'max arith mean change:   {max_arith_mean_change:11.4e}',
        f'shape: {input.shape} stride: {input.stride()} dtype: {dtype}',
        f'mismatched indices:{not_close_mask.nonzero().cpu()}',
    ]
    if msg_prefix is not None:
        msgs = [f'{msg_prefix}{msg}' for msg in msgs]

    return allclose, msgs


def any_output_requires_grad(fn):
    out = fn()
    if not isinstance(out, tuple):
        out = (out,)

    any_requires = any(i.requires_grad for i in out if isinstance(i, torch.Tensor))
    return any_requires


def markif(condition_func, mark):
    """
    condition_func: a function without parameters, return bool, decide whether to add mark
    mark: pytest.mark.xxx
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Initialize _markif as a list if it doesn't exist
        if not hasattr(wrapper, "_markif"):
            wrapper._markif = []

        # Add the new condition and mark to the list
        wrapper._markif.append((condition_func, mark))
        return wrapper

    return decorator
