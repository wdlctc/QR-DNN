#!/usr/bin/env python
# coding=-utf8
import caffe,struct
import numpy as np
#from PIL import Image
from math import log
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format
from argparse import ArgumentParser
import sys
import os
import os.path as osp

def write_bias(name, fl):

    if not os.path.exists(ROOT+'/bias'): os.makedirs(ROOT+'/bias')

    bias=open(ROOT+'/bias/'+output+'_bias.txt','w+',0)
    bias.write('layer_name:'+' '+layers[i].name+' ')
    bias.write('layer_type: '+layers[i].type+' ')
    bias.write('fl_params: '+str(fl_params)+' ')
    bias.write('\n') 
    d=net.params[name][1].data.flatten()
    scale = 2 ** fl_params
    for l in d:
	bias.write(bias_data(l,scale, output == 'co'))
	bias.write('\n')
	#f_all.write(change_data(l,scale))
	#f_all.write('\n')
    bias.close()

def write(datas, fl, bw, f):
    scale = 2 ** fl
    for data in datas:
        data = int (data * scale)
        data = data if(data>=0) else 2**bw + data
        s = bin(data)[2:]
        s = str(s).zfill(bw)
        f.write(s + '\n')

def write_weight(net, layers, path):

    f = path + '/weight/'
    if not os.path.exists(f): os.makedirs(f)
    
    for j, layer in enumerate(layers):
        if 'Convolution' in layer.type or 'InnerProduct' in layer.type:
            name = layer.name
            weight = open(f + name + '_weight.txt','w+',0)
            weight.write('layer_name: '+layer.name+' \n')
            weight.write('layer_type: '+layer.type+' \n')
            weight.write('fl_params: '+str(layer.quantization_param.fl_params)+' \n')
            w = net.params[name][0].data.flatten()
            write (w, layer.quantization_param.fl_params, layer.quantization_param.bw_params, weight)
            weight.close()       
            
            if len(net.params[name]) > 1 :
                bias = open(f + name + '_bias.txt','w+',0)
                bias.write('layer_name: '+layer.name+' \n')
                bias.write('layer_type: '+layer.type+' \n')
                bias.write('fl_params: '+str(layer.quantization_param.fl_layer_out)+' \n')
                b = net.params[name][1].data.flatten()
                write (b, layer.quantization_param.fl_layer_out, layer.quantization_param.bw_layer_out, bias)
                bias.close()       
            

def main(args):
    # Set default output file names
    if args.output_model is None:
        file_name = osp.splitext(args.model)[0]
        args.output_model = file_name + '_fixed_point.prototxt'
    if args.output_weights is None:
        file_name = osp.splitext(args.weights)[0]
        args.output_weights = file_name + '_fixed_point.caffemodel'
    if args.width_data is None:
        args.width_data = 8
    if args.width_weight is None:
        args.width_weight = 8

    caffe.set_mode_gpu()
    net = caffe.Net(args.model,args.weights,caffe.TEST)
    model = caffe_pb2.NetParameter()
    f=open(args.model,'rb')
    pb.text_format.Merge(f.read(), model)
    f.close()
    layers=model.layer
    
    max_in = []
    max_out = []
    
    for j in range(len(layers)):
        max_in.append(0)
        max_out.append(0)
    
    for i in range(10):
        caffe.set_mode_gpu()
        net.forward()
        for j, layer in enumerate(model.layer):
            if 'Convolution' in layer.type or 'InnerProduct' in layer.type:
                 top = layer.top[0]
                 bottom = layer.bottom[0]
                 #print top, bottom, layers[j].name
                 max_in[j] = max( np.max(net.blobs[bottom].data) , max_in[j])
                 max_out[j] = max( np.max(net.blobs[top].data) , max_out[j])
        #print [(k,np.max(v.data)) for k,v in net.blobs.items()]
	
    for j, layer in enumerate(model.layer):
        if 'Convolution' in layer.type or 'InnerProduct' in layer.type:
            name=layer.name
            #print name
            fl_layer_in = int(np.ceil(np.log2(max_in[j]))) ;
            fl_layer_out = int(np.ceil(np.log2(max_out[j]))) ;
            fl_param = int(np.ceil(np.log2(np.max(net.params[name][0].data)))) + 1 ;
            #print fl_param, np.max(net.params[name][0].data)
            
            #if 'Convolution' in layer.type and name != "conv1" : layer.type = 'ConvolutionInq'
            #if 'InnerProduct' in layer.type : layer.type = 'FcInq'
            layer.quantization_param.precision = 0
            layer.quantization_param.bw_layer_in = args.width_data
            layer.quantization_param.fl_layer_in = args.width_data - fl_layer_in
            layer.quantization_param.bw_layer_out = args.width_data
            layer.quantization_param.fl_layer_out = args.width_data - fl_layer_out
            layer.quantization_param.bw_params = args.width_weight
            layer.quantization_param.fl_params = args.width_weight - fl_param
            
            scale = 2 ** layer.quantization_param.fl_params
            scale_bias = 2 ** layer.quantization_param.fl_layer_out
            try :
                print np.max(np.round( net.params[name][0].data[:] * scale))
                net.params[name][0].data[:] = np.round( net.params[name][0].data[:] * scale) / scale
                net.params[name][1].data[:] = np.round( net.params[name][1].data[:] * scale_bias) / scale_bias
                pass
            except :
                pass
            #print np.max(net.params[name][1].data[:]*scale_bias)
            
    with open(args.output_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))
        
    net.save(args.output_weights);
    
    #write_weight(net , model.layer, os.path.split(args.output_model)[0])



if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('model', help="The net definition prototxt")
    parser.add_argument('weights', help="The weights caffemodel")
    parser.add_argument('--output_model')
    parser.add_argument('--output_weights')
    parser.add_argument('--width_data')
    parser.add_argument('--width_weight')
    args = parser.parse_args()
    main(args)


