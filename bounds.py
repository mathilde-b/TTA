#!/usr/bin/env python3.6

from typing import Any, List

import torch
from torch import Tensor
import pandas as pd
from utils import eq


class ConstantBounds():
    def __init__(self, **kwargs):
        self.C: int = kwargs['C']
        self.const: Tensor = torch.zeros((self.C, 1, 2), dtype=torch.float32)

        for i, (low, high) in kwargs['values'].items():
            self.const[i, 0, 0] = low
            self.const[i, 0, 1] = high

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        return self.const

class PredictionBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.dir: str = kwargs['dir']
        self.mode = "percentage" 
        self.sizefile: float = kwargs['sizefile']
        self.sep = kwargs['sep']
        self.sizes = pd.read_csv(self.sizefile,sep=self.sep)
        self.predcol: bool = kwargs['predcol']
        
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c,w,h=target.shape 
        pred_size_col = self.predcol
        #print(self.sizes.loc[:,pred_size_col])
        #print(self.sizes.val_ids)
        #print(self.sizes.loc[self.sizes.val_ids == filename])
        try:
            value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        except:
            print(filename,filename[1:])
            value = eval(self.sizes.loc[self.sizes.val_ids == filename[1:], pred_size_col].values[0])
            #print('not eval')
            #print(self.sizes[pred_size_col])
            #print(self.sizes.columns)
            #print(filename)
            #value = self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0]
        value = torch.tensor([value]).squeeze(0)
        #with_margin: Tensor = torch.stack([value, value], dim=-1)
        #assert with_margin.shape == (*value.shape, 2), with_margin.shape

        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        else:
            raise ValueError("mode")

        if self.dir == "both":
            with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        elif self.dir == "high":
            with_margin: Tensor = torch.stack([value, value + margin], dim=-1)
        elif self.dir == "low":
            with_margin: Tensor = torch.stack([value-margin, value], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        #res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        res = torch.max(with_margin, torch.zeros(*value.shape, 2).type(torch.long)).type(torch.float32)
        #print(res.shape,'res.shape')
        return res


class PreciseBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        #self.power: int = kwargs['power']
        self.power: int = 1
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            #value: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
            value: Tensor = self.__fn__(target[None, ...].type(torch.float32))[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        #print(res.shape,"in bounds")
        return res


class PreciseTags(PreciseBounds):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.neg_value: List = kwargs['neg_value']

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0

        res = super().__call__(image, target, weak_target, filename)

        masked = res[...]
        masked[positive_class == 0] = torch.Tensor(self.neg_value)

        return masked


class BoxBounds():
    def __init__(self, **kwargs):
        self.margins: Tensor = torch.Tensor(kwargs['margins'])
        assert len(self.margins) == 2
        assert self.margins[0] <= self.margins[1]

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c = len(weak_target)
        box_sizes: Tensor = torch.einsum("cwh->c", [weak_target])[..., None].type(torch.float32)

        bounds: Tensor = box_sizes * self.margins

        res = bounds[:, None, :]
        assert res.shape == (c, 1, 2)
        assert (res[..., 0] <= res[..., 1]).all()
        return res


def CheckBounds(**kwargs):
        sizefile: float = kwargs['sizefile']
        sizes = pd.read_csv(sizefile,sep=kwargs['sep'])
        predcol: str = kwargs['predcol']
        #print(predcol, 'pred_size_col')
        #print(sizes.columns, 'self.sizes.columns')
        if predcol in sizes.columns:
            return True
        else:
            print('size pred not in file')
            print(sizes.columns, 'self.sizes.columns')
            print(sizes.shape,"size file shape")
            return False
