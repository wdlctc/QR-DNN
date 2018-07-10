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


def add_bn_scale(src_model, dst_model):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    model_dst = caffe.proto.caffe_pb2.NetParameter()

    pd = True     

    top0 = None
    top1 = None

    for i, layer in enumerate(model.layer):
	
	top = layer.top[0]
	
	try:
	    bottom = layer.bottom[0]
	except:
	    bottom = None
	
	if top0 != None:
	    
	    if top == top0:
	        model.layer[i].top[0] = top + layer.type
	        model.layer[i+1].bottom[0] = top + layer.type
	    else :
	        top0=None
	    
	    
	name = layer.name

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
	    bn.name = name + 'bn'
	    bn.top[0] = top + 'bn'
	    bn.bottom.append(top)
	    scale = model_dst.layer.add()
	    scale.CopyFrom(caffe.layers.Scale(bias_term=True).to_proto().layer[0])
	    scale.name = layer.name + 'scale'
	    scale.bottom.append(top + 'bn')
	    scale.top[0] = top + 'scale'
	    model.layer[i+1].bottom[0] = top + 'scale'
            top0 = top

	if layer.type == 'InnerProduct' and layer.inner_product_param.num_output != 1000  and layer.inner_product_param.num_output != 10: 
	    bn = model_dst.layer.add()
	    bn.CopyFrom(caffe.layers.BatchNorm(use_global_stats=False).to_proto().layer[0])
	    bn.name = name + 'bn'
	    bn.top[0] = top + 'bn'
	    bn.bottom.append(top)
	    scale = model_dst.layer.add()
	    scale.CopyFrom(caffe.layers.Scale(bias_term=True).to_proto().layer[0])
	    scale.name = layer.name + 'scale'
	    scale.bottom.append(top + 'bn')
	    scale.top[0] = top + 'scale'
	    model.layer[i+1].bottom[0] = top + 'scale'
	    top0 = top

    with open(dst_model, 'w') as f:
	if pd is True:
	    print model_dst
            f.write(pb.text_format.MessageToString(model_dst))
	else:
	    print model
	    f.write(pb.text_format.MessageToString(model))

def main(args):
    # Set default output file names
    if args.output_model is None:
        file_name = osp.splitext(args.model)[0]
        args.output_model = file_name + '_inference.prototxt'

    add_bn_scale(args.model, args.model + '.temp.pt')

    if args.sh <> None:
        os.system(args.sh)


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('model', help="The net definition prototxt")
    parser.add_argument('--sh')
    parser.add_argument('--output_model')
    args = parser.parse_args()
    main(args)
