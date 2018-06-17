# -*- coding: utf-8 -*-


import ast
import collections
import copy
import csv
import enum
import sys

import numpy as np


FILE = './leaf.csv'


class Fields(enum.Enum):
    species                     = 0
    specimen_number             = enum.auto()
    eccentricity                = enum.auto()
    aspect_ratio                = enum.auto()
    elongation                  = enum.auto()
    solidity                    = enum.auto()
    stochastic_convexity        = enum.auto()
    isoperimetric_factor        = enum.auto()
    maximal_indentation_depth   = enum.auto()
    lobedness                   = enum.auto()
    average_intensity           = enum.auto()
    average_contrast            = enum.auto()
    smoothness                  = enum.auto()
    third_moment                = enum.auto()
    uniformity                  = enum.auto()
    entropy                     = enum.auto()


def load(*args, filename=None, dialect='excel', **kwargs):
    """Load dataset."""
    field_names, field_values = _parse_args(*args)
    dataclass = collections.namedtuple('Data', field_names)
    dataset = collections.namedtuple('Dataset', ['data', 'target'])

    data_list = list()
    target_list = list()
    with open(filename or FILE) as csvfile:
        for csv_line in csv.reader(csvfile, dialect, **kwargs):
            # convert integers
            arg_list = list()
            int_line = [ ast.literal_eval(x) for x in csv_line ]

            # fetch values
            for field in field_values:
                arg_list.append(int_line[field])

            # duplications
            data = dataclass(*arg_list)
            for _ in range(int_line[1]):
                data_list.append(copy.deepcopy(data))
                target_list.append(int_line[0])

    return dataset(np.array(data_list), np.array(target_list))


def _parse_args(*args):
    """Parse parameter range."""
    min_step = 0
    max_step = len(args) - 1

    field_names = list()
    field_values = list()

    for i, x in enumerate(args):
        if x is Ellipsis:
            last_step = args[i-1] if i > min_step else 0
            next_step = args[i+1] if i < max_step else len(Fields)
            for step in range(last_step+1, next_step):
                field_names.append(Fields(step).name)
                field_values.append(Fields(step).value)
        else:
            field_names.append(Fields(x).name)
            field_values.append(Fields(x).value)

    if args:
        # print(field_names, '\n')
        return (field_names, field_values)
    else:
        return ([ field.name for field in Fields ], [ field.value for field in Fields ])


def matrixify(target, *, species=36):
    """Maxtrixify target list."""
    array = [ [-1]*species for _ in target ]
    for i, x in enumerate(target):
        array[i][x-1] = 1
    return np.array(array)


if __name__ == '__main__':
    sys.exit(load(2, ...))
