import os
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def main(args):

    SolverParameter = caffe_pb2.SolverParameter()
    f = open(args.solver, 'r')
    SolverParameter = text_format.Merge(str(f.read()), SolverParameter)
    SolverParameter.net = args.model
    SolverParameter.snapshot = SolverParameter.max_iter
    f.close()
    f = open(args.solver+'.temp.pt', 'w')
    f.write(text_format.MessageToString(SolverParameter))
    f.close()

    os_system = "./build/tools/inq quantize \\\n" 
    os_system = os_system + "--model=" + args.model + ' \\\n'
    os_system = os_system + "--weights=" + args.weights + ' \\\n'
    os_system = os_system + "--model_quantized=" + args.output_model + ' \\\n'
    os_system = os_system + "--solver=" + args.solver + '.temp.pt' + ' \\\n'
    os_system = os_system + "--trimming_mode=" + args.mode + ' \\\n'
    os_system = os_system + "--gpu=" + args.gpu + ' \\\n'  
    os_system = os_system + "--iterations=10 " 

    print os_system

    os.system(os_system)
    
    output_weight = SolverParameter.snapshot_prefix + '_iter_9.caffemodel'
    args.output_model = args.model #there is a big problem here
    print output_weight

    try:
        os.rename(output_weight, args.output_weights)
    except:
        return
    

