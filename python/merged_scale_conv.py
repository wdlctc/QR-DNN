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


def load_and_fill_biases(src_model, src_weights, dst_model, dst_weights):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution': # or layer.type == 'Scale':
            # Add bias layer if needed
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    caffe.set_mode_cpu()
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)
    net_dst = caffe.Net(dst_model, caffe.TEST)
    for key in net_src.params.keys():
        for i in range(len(net_src.params[key])):
            net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]

    if dst_weights is not None:
        # Store params
        pass

    return net_dst

def merge_conv_and_bn(net, i_conv, i_scale):
    # This is based on Kyeheyon's work
    assert(i_conv != None)
    assert(i_scale != None)

    def copy_double(data):
        return np.array(data, copy=True, dtype=np.double)

    key_conv = net._layer_names[i_conv]
    key_scale = net._layer_names[i_scale] 

    # Copy
    print 'Combine {:s} + {:s}'.format(key_conv, key_scale)
    scale_weight = copy_double(net.params[key_scale][0].data)
    scale_bias = copy_double(net.params[key_scale][1].data)
    net.params[key_scale][0].data[:] = 1
    net.params[key_scale][1].data[:] = 0

    weight = copy_double(net.params[key_conv][0].data)
    bias = copy_double(net.params[key_conv][1].data)
    alpha = scale_weight 
    net.params[key_conv][1].data[:] = bias * alpha + (scale_bias)
    for i in range(len(alpha)):
        net.params[key_conv][0].data[i] = weight[i] * alpha[i]
    thread = np.max(abs(net.params[key_conv][0].data[:])) / 64
    #print alpha
    #print (net.params[key_conv][0].data[:])    
    #print np.sum(np.sum((abs(net.params[key_conv][0].data[:]) < thread) * (abs(net.params[key_conv][0].data[:]) != 0))) ,  np.sum(np.sum(abs(net.params[key_conv][0].data[:]) * 0 + 1))
    #net.params[key_conv][0].data[:] = (abs(net.params[key_conv][0].data[:]) >= thread) * (net.params[key_conv][0].data[:])
    

def merge_Scales_in_net(net):
    # for each Scale
    for i, layer in enumerate(net.layers):
        if layer.type != 'Scale':
            continue

        l_name = net._layer_names[i]

        l_bottom = net.bottom_names[l_name]
        assert(len(l_bottom) == 1)
        l_bottom = l_bottom[0]
        l_top = net.top_names[l_name]
        assert(len(l_top) == 1)
        l_top = l_top[0]

        can_be_absorbed = True
	print l_name,l_bottom,l_top
        # Search all (bottom) layers
        for j in xrange(i - 1, -1, -1):
            tops_of_j = net.top_names[net._layer_names[j]]
            if l_bottom in tops_of_j:
                if net.layers[j].type not in ['Convolution', 'InnerProduct']:
                    can_be_absorbed = False
                else:
                    # There must be only one layer
                    conv_ind = j
                    break

        if not can_be_absorbed:
            continue

        merge_conv_and_bn(net, conv_ind, i)

    return net


def process_model(net, src_model, dst_model, func_loop, func_finally):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)


    for i, layer in enumerate(model.layer):
        map(lambda x: x(layer, net, model, i), func_loop)

    map(lambda x: x(net, model), func_finally)

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i):
    if layer.type not in ['BatchNorm', 'Scale']:
        return

    bottom = layer.bottom[0]
    top = layer.top[0]

    if (bottom != top):
        # Not supperted yet
        return

    if layer.type == 'BatchNorm':
        zero_mean = np.all(net.params[layer.name][0].data == 0)
        one_var = np.all(net.params[layer.name][1].data == 1)
        length_is_1 = (net.params[layer.name][2].data == 1) or (net.params[layer.name][2].data == 0)

        if zero_mean and one_var and length_is_1:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)

    if layer.type == 'Scale':
        no_scaling = np.all(net.params[layer.name][0].data == 1)
        zero_bias = np.all(net.params[layer.name][1].data == 0)

        if no_scaling and zero_bias:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)

def remove_empty_layers(net, model):
    map(model.layer.remove, to_delete_empty)


# A function to add 'engine: CAFFE' param into 1x1 convolutions
def set_engine_caffe(layer, net, model, i):
    if layer.type == 'Convolution':
        if layer.convolution_param.kernel_size == 1\
            or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):
            layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']

def main(args):
    # Set default output file names
    if args.output_model is None:
        file_name = osp.splitext(args.model)[0]
        args.output_model = file_name + '_inference.prototxt'
    if args.output_weights is None:
        file_name = osp.splitext(args.weights)[0]
        args.output_weights = file_name + '_inference.caffemodel'

    net = load_and_fill_biases(args.model, args.weights, args.model + '.temp.pt', None)

    net = merge_Scales_in_net(net)

    process_model(net, args.model + '.temp.pt', args.output_model,
                  [pick_empty_layers, set_engine_caffe],
                  [remove_empty_layers])

    net.save(args.output_weights)

if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('model', help="The net definition prototxt")
    parser.add_argument('weights', help="The weights caffemodel")
    parser.add_argument('--output_model')
    parser.add_argument('--output_weights')
    args = parser.parse_args()
    main(args)
