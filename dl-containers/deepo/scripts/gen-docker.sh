python ../generator/generate.py ../docker/Dockerfile.tensorflow-py36-cpu tensorflow python==3.6
python ../generator/generate.py ../docker/Dockerfile.sonnet-py36-cpu sonnet python==3.6
python ../generator/generate.py ../docker/Dockerfile.mxnet-py36-cpu mxnet python==3.6
python ../generator/generate.py ../docker/Dockerfile.cntk-py36-cpu cntk python==3.6
python ../generator/generate.py ../docker/Dockerfile.keras-py36-cpu keras python==3.6
python ../generator/generate.py ../docker/Dockerfile.pytorch-py36-cpu pytorch python==3.6
python ../generator/generate.py ../docker/Dockerfile.chainer-py36-cpu chainer python==3.6
python ../generator/generate.py ../docker/Dockerfile.theano-py36-cpu theano python==3.6
python ../generator/generate.py ../docker/Dockerfile.lasagne-py36-cpu lasagne python==3.6
python ../generator/generate.py ../docker/Dockerfile.caffe-py36-cpu caffe python==3.6
python ../generator/generate.py ../docker/Dockerfile.torch-cpu torch
python ../generator/generate.py ../docker/Dockerfile.darknet-cpu darknet
python ../generator/generate.py ../docker/Dockerfile.paddle-py36-cpu paddle python==3.6
python ../generator/generate.py ../docker/Dockerfile.all-py36-cpu tensorflow sonnet mxnet cntk keras pytorch chainer theano lasagne caffe torch darknet paddle python==3.6 onnx jupyterlab
python ../generator/generate.py ../docker/Dockerfile.tensorflow-py36-cu102 tensorflow python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.sonnet-py36-cu102 sonnet python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.mxnet-py36-cu102 mxnet python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.cntk-py36-cu102 cntk python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.keras-py36-cu102 keras python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.pytorch-py36-cu102 pytorch python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.chainer-py36-cu102 chainer python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.theano-py36-cu102 theano python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.lasagne-py36-cu102 lasagne python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.caffe-py36-cu102 caffe python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.torch-cu102 torch --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.darknet-cu102 darknet --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.paddle-py36-cu102 paddle python==3.6 --cuda-ver 10.2 --cudnn-ver 7
python ../generator/generate.py ../docker/Dockerfile.all-py36-cu102 tensorflow sonnet mxnet cntk keras pytorch chainer theano lasagne caffe torch darknet paddle python==3.6 onnx jupyterlab --cuda-ver 10.2 --cudnn-ver 7
