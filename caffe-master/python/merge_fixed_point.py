#!/usr/bin/env python
import numpy as np
import sys
import os
import os.path as osp
import google.protobuf as pb
import google.protobuf.text_format
from argparse import ArgumentParser
import sys
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import argparse
import time

import gen_merged_model
import ristretto
import option

def change_SolverParameter(solver, model):

    SolverParameter = caffe_pb2.SolverParameter()
    f = open(solver, 'r')
    SolverParameter = text_format.Merge(str(f.read()), SolverParameter)
    SolverParameter.net = model
    if option.DEBUG is False: SolverParameter.snapshot = SolverParameter.max_iter
    f.close()
    f = open(solver, 'w')
    f.write(text_format.MessageToString(SolverParameter))
    f.close()
    return SolverParameter

def create_args(model, weight, phase, model_output=None, weight_output=None):
    
    args = argparse.Namespace()
    args.model = model
    args.weights = weight
    if phase != None:
        file_name = args.model.split('.')[0] 
        args.output_model = file_name + phase + '.prototxt'
        file_name = args.weights.split('.')[0]
        args.output_weights = file_name + phase + '.caffemodel'
    else:
        args.output_model = model_output
        args.output_weights = weight_output
    
    return args,args.output_model,args.output_weights 

def test_accuracy(net):

    caffe.set_mode_gpu()
    accuracy = 0; 
    batch_size = net.blobs['data'].num
    test_iters = option.DATASIZE / batch_size
    
    for i in range(test_iters):
    
        net.forward()
        accuracy += net.blobs['accuracy'].data
    accuracy /= test_iters
    return accuracy
    

def add_bn_scale(src_model, dst_model):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    model_dst = caffe.proto.caffe_pb2.NetParameter()

    pd = True     

    for i, layer in enumerate(model.layer):
	
	top = layer.top[0]
	name = layer.name
        #if layer.type == 'Convolution' :
            # Add bias layer if needed
	    #del layer.param[1]
            #if layer.convolution_param.bias_term == True:
                #layer.convolution_param.bias_term = False

	if layer.type == 'LRN':
	    bottom = layer.bottom[0]
	    model.layer[i+1].bottom[0] = bottom
	    continue

	if layer.type == 'Dropout':
	    bottom = layer.bottom[0]
	    model.layer[i+1].bottom[0] = bottom
	    continue

	model_dst.layer.add().CopyFrom(layer)

	if layer.type == 'BatchNorm':
	    pd = False
	    break

	if layer.type == 'Convolution':
	    bn = model_dst.layer.add()
	    bn.CopyFrom(caffe.layers.BatchNorm(use_global_stats=False).to_proto().layer[0])
	    bn.name = name.replace('conv','bn')
	    bn.top[0] = top
	    bn.bottom.append(top)
	    scale = model_dst.layer.add()
	    scale.CopyFrom(caffe.layers.Scale(bias_term=True).to_proto().layer[0])
	    scale.name = layer.name.replace('conv','scale')
	    scale.bottom.append(top)
	    scale.top[0] = top

    with open(dst_model, 'w') as f:
	if pd is True:
            f.write(pb.text_format.MessageToString(model_dst))
	else:
	    f.write(pb.text_format.MessageToString(model))

def main(args):

    if args.skip_first_phase is not None:
        file_name = osp.splitext(args.model)[0]
        model_first = args.model
        print args.weight
        if args.weight is None :
            SolverParameter = change_SolverParameter(args.solver_first, model_first) 
            caffe.set_device(args.gpu)
            caffe.set_mode_gpu()
            #solver = caffe.SGDSolver(args.solver_first)
            #solver.solve()
            weight_first = str(SolverParameter.snapshot_prefix)  + '_iter_' + str(SolverParameter.max_iter) +  '.caffemodel'
        else:
            weight_first = args.weight
        print "First phase skip!"

    else:
        
        file_name = osp.splitext(args.model)[0]
        model_first = file_name + '.First' + '.prototxt'   
        
        SolverParameter = change_SolverParameter(args.solver_first, model_first) 
        
        add_bn_scale(args.model, model_first)

        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()
        #solver = caffe.SGDSolver(args.solver_first)
        #solver.solve()
        
        weight_first = str(SolverParameter.snapshot_prefix)  + '_iter_' + str(SolverParameter.max_iter) +  '.caffemodel'

        print "First phase complete!"
        
        
    kargs, model_second, weight_second = create_args(model_first,weight_first,".Second")
    gen_merged_model.main(kargs)
    print kargs
           
    print "Second phase complete!"
    if option.DEBUG is False: 
        os.remove(model_first)
        os.remove(weight_first)
        
    kargs, model_third, weight_third = create_args(model_second,weight_second, None, file_name+ '.Final.prototxt', file_name+ '.Final.caffemodel')
    kargs.width_data = args.width_data
    kargs.width_weight = args.width_weight
    ristretto.main(kargs)
        
    print "Third phase complete!"
    if option.DEBUG is False: 
        os.remove(model_second)
        os.remove(weight_second)
        
    print "All phases complete, end fixed-point quantization process"
        
    net = caffe.Net(model_third,weight_third,caffe.TEST)
    accuracy = test_accuracy(net)
        
    print "The final accuracy of quantized network is %f "% accuracy
    


