# This is docker template for deep learning project

## System Requirements

* [Install docker-compose](https://docs.docker.com/compose/install/)
* [Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (only for gpu usage)

## Containers


### flask_keras_2_2_4_cpu

flask api container for Keras 2.2.4

```
$ docker-compose up flask_keras_2_2_4_cpu
```

In different shell window, 

```
$ curl -F "img=@./data/sample/gu/gu1.jpg" http://localhost:5000/images
{"result":1.0,"success":true}
```

### jupyter_keras_2_2_4_cpu

jupyter container for Keras 2.2.4

```
$ docker-compose up jupyter_keras_2_2_4_cpu
```

Access http://localhost:8888/

See keras_2_2_4_cpu_example.ipynb to see an example

### jupyter_pytorch_1_0_0_gpu

jupyter container for PyTorch 1.0.0

```
$ docker-compose up jupyter_pytorch_1_0_0_gpu
```

Access http://localhost:28888/

See pytorch_1_0_0_gpu_example.ipynb to see an example

### train_keras_2_2_4_cpu

training container example for Keras 2.2.4

```
$ docker-compose up train_keras_2_2_4_cpu
```

### train_keras_2_2_4_gpu

training gpu container example for Keras 2.2.4

```
$ docker-compose up train_keras_2_2_4_gpu
```
