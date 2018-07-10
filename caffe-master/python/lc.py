import numpy as np 
from math import log
from PIL import Image
import caffe


net = caffe.Net('models/cnn-models-master/my_alexnet/final.prototxt',
		'models/cnn-models-master/my_alexnet/final.caffemodel' , 
		caffe.TEST)
print dir(net)
#net.forward()
#print [(k,v.data) for k,v in net.blobs.items()]
#print [(k,v[0].data, v[1].data) for k,v in net.params.items()]
#net.save('final.caffemodel');

for k,v in net.params.items():
	if  (~k.find('conv')):
		print  k, np.max(v[0].data)

"""
for k,v in net.blobs.items():
	if  (~k.find('res2a_branch1')):
		bias_max=np.max(abs(v.data ))
            	quan_max=2 ** np.ceil (log(bias_max , 2))
	    	scale = 128 / quan_max
	    	fl = int( 7 - np.ceil (log(bias_max , 2)))
		print k
		print  v.data[0][3][:] #, np.max(abs(v.data )), np.max(abs(v.data )) *scale ,fl

for k,v in net.params.items():
	if  (~k.find('res2a_branch1')):
		print  v[0].data[3][:] #, np.max(abs(v.data )), np.max(abs(v.data )) *scale ,fl
		print  v[1].data


for k,v in net.params.items():
	if  (~k.find('res2a_branch1')):
		bias_max=np.max(abs(v[1].data)) 
            	quan_max=2 ** np.ceil (log(bias_max , 2))
	    	scale = 128 / quan_max
	    	fl = int( 7 - np.ceil (log(bias_max , 2)))
		print k
		print  v[1].data, np.max(abs(v[1].data)) ,fl

for k,v in net.blobs.items():
	if  (~k.find('res2a_branch1')):
		bias_max=np.max(abs(v.data ))
            	quan_max=2 ** np.ceil (log(bias_max , 2))
	    	scale = 128 / quan_max
	    	fl = int( 7 - np.ceil (log(bias_max , 2)))
		print k
		print  v.data, np.max(abs(v.data )) ,fl


for k,v in net.params.items():
	if  (~k.find('res')):
	    	bias_max=np.max(abs(v[1].data))
            	quan_max=2 ** np.ceil (log(bias_max , 2))
	    	scale = 128 / quan_max
		print k
		print  v[1].data * scale, np.max(abs(v[1].data * scale))  , bias_max ,quan_max,scale
"""
"""
		print v[0].data.astype('int')
		c = v[0].data.astype('int')
		c = c.astype('string')
		print c.tostring()
		lc = open('lc.txt','w')
		lc.write (c.tostring())
		#np.savetxt("lc.txt",c)
print  net.params.items()
"""

