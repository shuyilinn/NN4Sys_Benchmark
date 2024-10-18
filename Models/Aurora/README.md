# Aurora: Reinforcement Learning Resources for Performance-Oriented Congestion Control

Aurora is a reinforcement learning-based project focused on optimizing congestion control for network performance. This repository contains resources for training and evaluating models within this domain.

## Training Instructions

To train the Aurora model, follow these steps:

1. **Navigate to the Training Directory**  
   Open a terminal and navigate to the `/gym/` directory within the project:
   ```bash
   cd /gym/
   ```

2. **Install Required Dependencies**  
   Ensure all necessary dependencies are installed. You can use `pip` to install missing packages. If a `requirements.txt` file is provided, run the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Training Script**  
Once the dependencies are installed, initiate the training by running the `train.py` script. You can specify the model type (`small`, `mid`, `big`, or `all`) to train different versions of the Aurora model. Use the following command:
   ```bash 
   python train.py --model {small, mid, big, all}
   ```
- `small`: Trains the small version of the model.
- `mid`: Trains the mid-sized model.
- `big`: Trains the large version of the model.
- `all`: Trains all three model versions sequentially.



Once training is complete, you will see the following message:
```bash
[Done] Finished training {args.model} model
```

