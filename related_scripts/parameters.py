#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: parameters.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 6th April 2020 4:01:27 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--str_parameter', type=str, default="default", help='str parameter')
    parser.add_argument('--str_parameter_choices', type=str, default="default", choices=['choice1', 'choice2','choice3'], help='str parameter with choices list')
    parser.add_argument('--int_parameter', type=int, default=0, help='int parameter')
    parser.add_argument('--float_parameter', type=float, default=0.0, help='float parameter')
    parser.add_argument('--bool_parameter', type=str2bool, default='True', choices=['True', 'False'], help='bool parameter')
    parser.add_argument('--list_str_parameter', type=str, nargs='+', default=["element1","element2"], help='str list parameter')
    parser.add_argument('--list_int_parameter', type=int, nargs='+', default=[0,1], help='int list parameter')
    return parser.parse_args()