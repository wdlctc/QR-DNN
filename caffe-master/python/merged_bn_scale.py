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
        
    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution': # or layer.type == 'Scale':
            # Add bias layer if needed
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0

    if dst_weights is not None:
        # Store params
        pass

    return net_dst

def merge_conv_and_bn(net, i_conv, i_bn, i_scale):
    # This is based on Kyeheyon's work
    assert(i_conv != None)
    assert(i_bn != None)

    def copy_double(data):
        return np.array(data, copy=True, dtype=np.double)

    key_conv = net._layer_names[i_conv]
    key_bn = net._layer_names[i_bn]
    key_scale = net._layer_names[i_scale] if i_scale else None

    # Copy
    bn_mean = copy_double(net.params[key_bn][0].data)
    bn_variance = copy_double(net.params[key_bn][1].data)
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    if net.params.has_key(key_scale):
        print 'Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale)
        scale_weight = copy_double(net.params[key_scale][0].data)
        scale_bias = copy_double(net.params[key_scale][1].data)
        alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + 1e-5)
	bias = (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
	
	#mid = sum(alpha)/len(alpha)
	#alpha_power_of_2 = 2**np.round(np.log2(mid)) * (alpha*0+1)
	#scale = alpha / alpha_power_of_2
	#print mid, alpha_power_of_2
	
	
	alpha_power_of_2 = 2**np.round(np.log2((alpha)*np.sign(alpha)))*np.sign(alpha)
	#bias_power_of_2 = 2**np.round(np.log2((bias)*np.sign(bias)))*np.sign(bias)
	scale = alpha / alpha_power_of_2
        #print sum(alpha)/len(alpha)
        net.params[key_scale][0].data[:] = alpha_power_of_2
        del(net.params[key_scale][1])
        #net.params[key_scale][1].data[:] = bias
    else:
        print 'Combine {:s} + {:s}'.format(key_conv, key_bn)
        scale_weight = 1
        scale_bias = 0

    
    weight = copy_double(net.params[key_conv][0].data)
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + 1e-5)
    #print len(conv_bias),len(scale),len(weight[0])
    try:
        conv_bias = copy_double(net.params[key_conv][1].data)
	net.params[key_conv][1].data[:] = conv_bias *  scale + bias / alpha_power_of_2
    except:
	pass
    for i in range(len(alpha)):
        net.params[key_conv][0].data[i] = weight[i] *  scale[i]
        #print np.max(weight[i])

    
def merge_batchnorms_in_net(net):
    # for each BN
    for i, layer in enumerate(net.layers):
        if layer.type != 'BatchNorm':
            continue

        l_name = net._layer_names[i]

        l_bottom = net.bottom_names[l_name]
        assert(len(l_bottom) == 1)
        l_bottom = l_bottom[0]
        l_top = net.top_names[l_name]
        assert(len(l_top) == 1)
        l_top = l_top[0]
	
        can_be_absorbed = True

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

        # find the following Scale
        scale_ind = None
        for j in xrange(i + 1, len(net.layers)):
            bottoms_of_j = net.bottom_names[net._layer_names[j]]
            if l_top in bottoms_of_j:
                if scale_ind:
                    # Followed by two or more layers
                    scale_ind = None
                    break

                if net.layers[j].type in ['Scale']:
                    scale_ind = j

                    top_of_j = net.top_names[net._layer_names[j]][0]
                    if top_of_j == bottoms_of_j[0]:
                        # On-the-fly => Can be merged
                        break

                else:
                    # Followed by a layer which is not 'Scale'
                    scale_ind = None
                    break
	print net._layer_names[conv_ind],net._layer_names[i],net._layer_names[scale_ind]
        merge_conv_and_bn(net, conv_ind, i, scale_ind)
	

    return net


def process_model(net, src_model, dst_model, func_loop, func_finally):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)


    for i, layer in enumerate(model.layer):
        map(lambda x: x(layer, net, model, i), func_loop)

    map(lambda x: x(net, model), func_finally)

    output=pb.text_format.MessageToString(model)
    output=output.replace('scale_param','param {lr_mult: 0}\n  scale_param')
    print output

    with open(dst_model, 'w') as f:
        f.write(output)


# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i):
    #print layer.type
    if layer.type not in ['BatchNorm', 'Scale']:
        return
    
    print layer.bottom[0],layer.top[0],layer.name,i
    print model.layer[i+1].name

    bottom = layer.bottom[0]
    top = layer.top[0]

    if layer.type == 'Scale':
    
        model.layer[i].scale_param.bias_term = False    

    if layer.type == 'BatchNorm':

	if (bottom != top):
	    model.layer[i+1].bottom[0] = bottom

	#print net.params[layer.name][0].data,net.params[layer.name][1].data,net.params[layer.name][2].data
        zero_mean = np.all(net.params[layer.name][0].data == 0)
        one_var = np.all(net.params[layer.name][1].data == 1)
        length_is_1 = (net.params[layer.name][2].data == 1) or (net.params[layer.name][2].data == 0)

        if zero_mean and one_var and length_is_1:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)
"""
    if layer.type == 'Scale':
        no_scaling = np.all(net.params[layer.name][0].data == 1)
        zero_bias = np.all(net.params[layer.name][1].data == 0)

        if no_scaling and zero_bias:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)
"""
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

    net = merge_batchnorms_in_net(net)

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
