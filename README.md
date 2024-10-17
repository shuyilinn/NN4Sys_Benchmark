# NN4SysBench: Characterizing Neural Network Verification for Computer Systems
## Introduction
We propose a benchmark suite for neural network verification for systems 
(NN4Sys) in this repository. This suite includes verification 
benchmark for learned index, learned cardinality, learned Internet congestion
control, learned adaptive bitrate, learned distributed system scheduler, 
which are five tasks that apply neural networks to solve traditional tasks for systems. 


## Quick Start

### 1. Train the Model

Pre-trained models are available, so you may skip this step if you wish to use them directly.  
However, if you prefer to train the models yourself, the training code is located in the `/Models` directory.  
We provide scripts for training the following models:

- **Learned Internet Congestion Control**
- **Learned Adaptive Bitrate**
- **Learned Distributed System Scheduler**

You can find the training and testing instructions and pre-trained models for each model in the table below:

| Model                                  | Training and Testing Instructions                                                                 | Pre-trained Model |
|----------------------------------------|----------------------------------------------------------------------------------------------------|------------------|
| Learned Internet Congestion Control     | [Instruction](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Models/Aurora)               | [Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Models/Aurora/gym/results) |
| Learned Adaptive Bitrate               | [Instruction](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Models/Aurora)               | [Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Models/Pensieve/results) |
| Learned Distributed System Scheduler   | [Instruction](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Models/Aurora)               | [Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Models/Decima/best_models) |






### 2. Generate specifications and onnx models
#### 2.1 Install dependencies
```
conda create -n myenv python=3.9 --yes
conda env create -f environment.yml
```
Then activate the environment
```
conda activate myenv
```



#### 2.2 generate instance pools
The second step is to generate the "instance pool" for each model, which include thousands of instances which be the resources when generating the benchmark instances. 
To generate fixed input files for models, run
```
cd Models
python gen_upper.py --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```
Note: current we only provide the script for 
- **Learned Internet Congestion Control**
- **Learned Adaptive Bitrate**
- **Learned Distributed System Scheduler**
You can skip this step as these files are provided, you can refer to this [table]

### 2.3 Create onnx models
We use onnx format model for verifying. To create onnx models from trained models, run
```
cd Models
python export.py --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```
You can skip this step as onnx models are provided, you can refer to this [table](#onnx-and-specifications-table).

### 2.4 Generate specifications
Then we can create specifications, run

```
cd Benchmarks
python generate_properties.py {--seed 2024}
cd ..
```
random seed default is 2024. 
In this step, we generate 10 instances for each specification seperately. If you want to generate more instances

You can skip this step as instances are provided, you can refer to the [table] below.(#onnx-and-specifications-table).

<a name="onnx-and-specifications-table"></a>

| Model                                  | Training and Testing Instructions                                                                 | ONNX Model                                                                                     |  Instance VNNLIB                                                                                     |  Instance txt                                                                                     |
|----------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Learned Internet Congestion Control     | [Instance Pool](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/src/aurora/aurora_resources) | [ONNX Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/onnx)           | [VNNLIB](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/vnnlib)                       | [TXT](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/marabou_txt)                   |
| Learned Adaptive Bitrate               | [Instance Pool](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/src/pensieve/pensieve_resources) | [ONNX Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/onnx)           | [VNNLIB](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/vnnlib)                       | [TXT](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/marabou_txt)                   |
| Learned Distributed System Scheduler   | [Instance Pool](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/src/decima/decima_resources) | [ONNX Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/onnx)           | [VNNLIB](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/vnnlib)                       | [TXT](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/marabou_txt)                   |
| Database Learned Index                 | [Instance Pool](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/src)            | [ONNX Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/onnx)           | [VNNLIB](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/vnnlib)                       | [TXT](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/marabou_txt)                   |
| Learned Bloom Filter                   | [Instance Pool](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/src/bloom_filter/bloom_filter_resources) | [ONNX Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/onnx)           | [VNNLIB](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/vnnlib)                       | [TXT](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/marabou_txt)                   |
| Learned Cardinalities                  | [Instance Pool](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/src/mscn/mscn_resources) | [ONNX Model](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/onnx)           | [VNNLIB](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/vnnlib)                       | [TXT](https://github.com/shuyilinn/NN4Sys_Benchmark/tree/main/Benchmarks/marabou_txt)                   |









### 3. Verify
Currently we provide script for alpha-beta crown and marabou. We are actively developing this project, more verifiers will be supported in the future.
### 3.1 Verify with alpha-beta-crown
install abcrown https://github.com/Verified-Intelligence/alpha-beta-CROWN
run
```
cd Verification
python abcrown_run.py --path {abcrown.py path} --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```

### 3.2 Verify with marabou
Install marabou https://github.com/NeuralNetworkVerification/Marabou/tree/master
run
```
cd Verification
python marabou_run.py --path {runMarabou.py path} --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```

### 4. Results visualization
run
```
cd Verification/figures
python create_json.py
python draw.py
cd ../..
```