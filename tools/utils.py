# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   utils.py
# @Time   :   2021/7/26 17:50
import numpy as np


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8f}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
        #     "{:<80s}: {:<16s}({:8d}) ({})".format(
        #         p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
        #     )
        # )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params / 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )
