# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import os
import sys


class FromEnvironment(argparse.Action):
    def __init__(
        self,
        envvar,
        nargs=1,
        required=True,
        default=None,
        **kwargs,
    ):
        if envvar in os.environ:
            default = os.environ[envvar]
            if nargs in ['+', '*']:
                default = default.split(',')
        if required and default is not None:
            required = False
        if 'choices' in kwargs:
            if nargs == 1:
                if default is not None and default not in kwargs['choices']:
                    raise RuntimeError(
                        f'argument {envvar}: invalid choice: {default} ' f'(choose from {kwargs["choices"]})'
                    )
            elif nargs in ['+', '*']:
                for item in default:
                    if item not in kwargs['choices']:
                        raise RuntimeError(
                            f'argument {envvar}: invalid choice: {item} ' f'(choose from {kwargs["choices"]})'
                        )

        super().__init__(required=required, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class CacheMeta(type):
    def __getattr__(self, name):
        return getattr(self.args, name)


class Config(metaclass=CacheMeta):
    args = None

    def __new__(cls):
        raise RuntimeError(f'class {cls} should never be instantiated')

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser(
            prog='TileGym Tests',
            description='TileGym Tests',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        parser.add_argument(
            '--seed',
            envvar='SEED',
            action=FromEnvironment,
            default=0,
            type=int,
            help=('set SEED to use manually provided random seed, if unset uses ' 'a random time-based seed'),
        )
        parser.add_argument(
            '--quiet',
            envvar='QUIET',
            action=FromEnvironment,
            required=False,
            default=False,
            type=bool,
            help=('set QUIET to supress printing of environment info'),
        )
        parser.add_argument(
            '--params',
            envvar='PARAMS',
            action=FromEnvironment,
            required=False,
            default=None,
            type=str,
            help=(
                'set PARAMS to manually specify test parameters, parameters '
                'are passed as a string containing a dictionary of keyword '
                'arguments for a given test case'
            ),
        )
        parser.add_argument(
            '--dump',
            envvar='DUMP',
            action=FromEnvironment,
            required=False,
            default=None,
            choices=['txt', 'html', 'csv', 'json', 'md'],
            type=str,
            help=('set DUMP to store results of performance tests in a ' 'specified format'),
        )
        parser.add_argument(
            '--dump-dir',
            envvar='DUMP_DIR',
            action=FromEnvironment,
            required=False,
            default='/tmp/',
            type=str,
            help=('set DUMP_DIR to specify directory to store results of ' 'performance tests'),
        )
        parser.add_argument(
            '--load',
            envvar='LOAD',
            action=FromEnvironment,
            required=False,
            default=None,
            choices=['txt', 'html', 'csv', 'json', 'md'],
            type=str,
            help=('set LOAD to load results of performance tests in a ' 'specified format'),
        )
        parser.add_argument(
            '--load-dir',
            envvar='LOAD_DIR',
            action=FromEnvironment,
            required=False,
            default='/tmp/',
            type=str,
            help=('set LOAD_DIR to specify directory to load results of ' 'performance tests'),
        )
        parser.add_argument(
            '--load-names',
            envvar='LOAD_NAMES',
            action=FromEnvironment,
            nargs='*',
            default=['cutile'],
            type=str,
            help=('set LOAD_NAMES to select which test series should be loaded'),
        )
        parser.add_argument(
            '--warmup',
            envvar='WARMUP',
            action=FromEnvironment,
            default=100,
            type=float,
            help=('set WARMUP to specify the warmup duration for performance ' 'tests [ms]'),
        )
        parser.add_argument(
            '--rep',
            envvar='REP',
            action=FromEnvironment,
            default=50,
            type=float,
            help=('set REP to specify the duration for measured iterations of ' 'performance tests [ms]'),
        )
        parser.add_argument(
            '--min-rep',
            envvar='MIN_REP',
            action=FromEnvironment,
            default=2,
            type=int,
            help=('set MIN_REP to specify the minimum number of measured ' 'iterations for performance tests'),
        )
        parser.add_argument(
            '--initial-rep',
            envvar='INITIAL_REP',
            action=FromEnvironment,
            default=5,
            type=int,
            help=(
                'set INITIAL_REP to specify number of initial iterations for '
                'performance tests, initial iterations are to establish '
                'approximate runtime and to compute number of warmup and '
                'benchmark iterations given a budget expressed in milliseconds'
            ),
        )
        parser.add_argument(
            '--mode',
            envvar='MODE',
            action=FromEnvironment,
            required=False,
            default='auto',
            choices=['forward', 'backward', 'auto'],
            type=str,
            help=(
                'set MODE to specify operating mode for performance tests: '
                '"forward" runs only the forward pass, '
                '"backward" runs only the backward pass, '
                '"auto" runs forward pass and runs backward pass if any '
                'output tensor has requires_grad attribute set to True'
            ),
        )
        parser.add_argument(
            '--csv',
            envvar='CSV',
            action=FromEnvironment,
            required=False,
            default=False,
            type=bool,
            help=('Record to csv file'),
        )
        parser.add_argument(
            '--file',
            envvar='FILE',
            action=FromEnvironment,
            required=False,
            default='out',
            type=str,
            help=("When use cudagraph & csv, can specify file path"),
        )
        parser.add_argument(
            '--config',
            envvar='CONFIG',
            action=FromEnvironment,
            required=False,
            default='dev',
            type=str,
            help=('set CONFIG to specify the name of a set of hyperparameters ' 'to be launched by automatic tests'),
        )
        parser.add_argument(
            '--fields',
            envvar='FIELDS',
            action=FromEnvironment,
            nargs='*',
            default=['median', 'rel_std'],
            type=str,
            # choices=[
            #     'mean',
            #     'std',
            #     'rel_std',
            #     'median',
            #     'min',
            #     'max',
            #     'nrep',
            #     'peak_mem_mb',
            # ],
            help=(
                'set FIELDS to specify metrics reported by performance tests: '
                '"mean" - mean runtime, '
                '"std" - standard deviation of runtime, '
                '"rel_std" - mean / standard deviation, '
                '"median" - median runtime, '
                '"min" - min runtime, '
                '"max" - max runtime, '
                '"peak_mem_mb" - max GPU MB used, '
                'all time metrics are reported in milliseconds'
            ),
        )
        parser.add_argument(
            '--print-matching',
            envvar='PRINT_MATCHING',
            action=FromEnvironment,
            default=False,
            type=bool,
            help=('set PRINT_MATCHING to always print info from assertAllClose'),
        )
        parser.add_argument(
            '--verbose',
            envvar='VERBOSE',
            action=FromEnvironment,
            default=False,
            type=bool,
            help=('set VERBOSE to enable verbose prints'),
        )
        parser.add_argument(
            '--help',
            envvar='HELP',
            action=FromEnvironment,
            default=False,
            type=bool,
            help=('set HELP to any value to print help and exit'),
        )

        cls.args = parser.parse_args(args=[])

        if cls.args.help:
            parser.print_help()
            sys.exit(0)
