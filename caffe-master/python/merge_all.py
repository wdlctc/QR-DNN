#!/usr/bin/env python
import sys
import os
from argparse import ArgumentParser

import merge_fixed_point
import merge_power_of_two

def main(args):

    if args.model is None:
        raise NameError("Undefined model")
        
    if (args.solver_first is None) and (args.weight is None):
        raise NameError("Undefined solver")
        
    if args.gpu is None:
        args.gpu = 0
    else:
        args.gpu = int(args.gpu)

    if args.mode not in ["Incremental","dynamic_fixed_point","integer_power_of_2_weights"] :
        print "Unused mode, use Incremental mode instead"
        args.mode = "Incremental"
        
    if args.mode == "dynamic_fixed_point":
    
        if args.width_data is None:
            args.width_data = 8
        if args.width_weight is None:
            args.width_weight = 8
            
        merge_fixed_point.main(args)
        
    elif args.mode == "integer_power_of_2_weights":
    
        if args.width_data is None:
            args.width_data = 8
        if args.width_weight is None:
            args.width_weight = 4
            
        merge_power_of_two.main(args)    	

if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('--model', help="The net definition prototxt")
    parser.add_argument('--weight', help="The pre-train weight")  
    parser.add_argument('--solver_first', help='The first solver')  
    parser.add_argument('--solver_second', help='The second solver')  
    parser.add_argument('--mode', help="The mode of quantization, including Fixed-point, Power-of-tow and Incremental")  
    parser.add_argument('--width_data', help="The bit-width of activations used for quantization")
    parser.add_argument('--width_weight', help="The bit-width of weights used for quantization")
    parser.add_argument('--gpu', help="Which gpu used for training/inference")
    parser.add_argument('--skip_first_phase', help="Whether to skip first phase for ResNets or other batch-normalized networks")
    args = parser.parse_args()

    main(args)
