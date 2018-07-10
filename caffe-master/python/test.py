

def bias_data(data,scale,flag =0):
    s = int (data * scale)
    s = (s if(s>=0) else 65536 + s)
    s = bin(s)[2:]
    s = str(s).zfill(16)
    if flag: print data,scale,int(data * scale),s
    return s

def weight_data(data,scale,flag =0):
    s = int (data * scale)
    s = (s if(s>=0) else 256 + s)
    s = bin(s)[2:]
    s = str(s).zfill(8)
    if flag: print data,scale,int(data * scale),s
    return s

def activation_data(data,div,flag = 0):
    new2_data=int(data*(2**div))
    new_data=(new2_data if (new2_data>=0) else 256 +new2_data)
    new_data=bin(new_data)[2:]
    new_data=str(new_data).zfill(8)
    if flag:
	print data,div,new_data
    return new_data

def write(name, fl):

    print name,'activation'
    if not os.path.exists(ROOT+'/data'): os.makedirs(ROOT+'/data')

    activation=open(ROOT+'/data/'+name+'.txt','w+',0)
    activation.write(r'//')
    activation.write('input_blob name: ' + name + ' fl : ' + str(fl))
    activation.write('\n')
    """
    try:
       size = channel_size = net.blobs[str(name)].data.shape[0]
       channel_size = net.blobs[str(name)].data.shape[1]
       row = net.blobs[str(name)].data.shape[2]
       col = net.blobs[str(name)].data.shape[3]
       print name,'activation',size,channel_size,row,col
    except:     
       return
    """   
    d=net.blobs[str(name)].data.flatten()

    for l in d:
       activation.write(activation_data(l,fl,name=='res'))
       activation.write('\n')
       #f_all.write(change_data(l,fl))  
       #f_all.write('\n')     

def write_bias(name, fl):

    print name,'bias'
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

def write_weight(name, fl):

    print name,'weight'
    if not os.path.exists(ROOT+'/weight'): os.makedirs(ROOT+'/weight')

    weight=open(ROOT+'/weight/'+output+'_weight.txt','w+',0)
    weight.write('layer_name:'+' '+layers[i].name+' ')
    weight.write('layer_type: '+layers[i].type+' ')
    weight.write('fl_params: '+str(fl_params)+' ')
    weight.write('\n') 
    d=net.params[name][0].data.flatten()
    scale = 2 ** fl_params
    for l in d:
	weight.write(weight_data(l,scale, output == 'res'))
	weight.write('\n')
	#f_all.write(change_data(l,scale))
	#f_all.write('\n')
    weight.close()

for i in range(len(layers)):
	if 'Convolution' in layers[i].type or 'Fc' in layers[i].type:
	    output=layers[i].name
	    print output
    	    fl_layer_in=layers[i].quantization_param.fl_layer_in
            fl_layer_out=layers[i].quantization_param.fl_layer_out
            fl_params=layers[i].quantization_param.fl_params

	    for j,layer in enumerate(net.layers):
		if net._layer_names[j] == layers[i].name:
			break

	    scale = 2 ** fl_params
	    scale_bias = 2 ** fl_layer_out
	
	    for j in range(len(net.params[output])):
		if j == 0:
            		net.params[output][j].data[:] = np.round( layer.blobs[j].data[:] * scale) / scale
			print j,np.max(net.params[output][j].data[:] * scale)
		if j == 1:
            		net.params[output][j].data[:] = np.round( layer.blobs[j].data[:] * scale_bias) / scale_bias
			print j,np.max(net.params[output][j].data[:] * scale_bias)
		

net.save(ROOT+'/final.caffemodel');

for i in range(len(layers)):
	if 'Convolution' in layers[i].type or 'InnerProduct' in layers[i].type:
	    output=layers[i].name
	    print output
    	    fl_layer_in=layers[i].quantization_param.fl_layer_in
            fl_layer_out=layers[i].quantization_param.fl_layer_out
            fl_params=layers[i].quantization_param.fl_params

	    for j,layer in enumerate(net.layers):
		if net._layer_names[j] == layers[i].name:
			break

	    scale = 2 ** fl_params

	    #for j in range(len(net.params[output])):
		#print np.max(net.params[output][j].data[:] * scale)
#net.forward()

"""
	    if(len(net.params[output]) >= 2):
	    	write_bias(output,fl_params)
            write_weight(output,fl_params)
	    write(layers[i].bottom[0],fl_layer_in)
    	    write(layers[i].top[0],fl_layer_out)
"""




#f_all.close()

