# Advanced-computer-vision-group-E-coursework

**Repository Link:** https://github.com/aal57/Advanced-computer-vision-coursework-group-E

**General**
The environment should be ran from the folder that contains the subfolders C-tasks and D-tasks.

**C1 task**
1. Open your terminal (Command Prompt or PowerShell).
2. Navigate to the `C-tasks` directory inside the project folder:
   ```bash
   cd C-tasks
3. Run the following command: py main.py --Task1Dataset Task1Dataset
4. To test different dataset, add dataset to C-tasks folder and change the dataset name after --Task1Dataset.

**C2 & C3 tasks**
1. Open your terminal (Command Prompt or PowerShell).
2. Navigate to the `C-tasks` directory inside the project folder:
   ```bash
   cd C-tasks
3. Run the following command: py main.py --IconDataset IconDataset --Task2Dataset Task2Dataset --Task3Dataset Task3Dataset
4. To test different dataset, add dataset to C-tasks folder and change the dataset name after --IconDataset, --Task2Dataset and --Task3Dataset.
5. To run only one or the other, do not add the other dataset names to the command.

**D1 task**
Running main within D1 will train a new model for 250 epochs and store it as d1.pth in the models folder.

**D2 task**
Running main within D2 will train a new model for 250 epochs and store it as d2.pth in the models folder.

**D3 task**
Running main within D3 will train a new model for 250 epochs and store it as d3.pth in the models folder.

**D4 task**
Running main will train a new model, adjust for x in [True] (trains fine labels) to x in [False] for coarse or [True, False] for both.
Set the margin in the train_model function, and the batch_size can be adjusted in the for batch_size in [...]

Test_checks.py has been set so d4_checks() is run twice, once with the correct margin for coarse, then with the correct margin for fine.

**D5 task**
Running main within D3 will train a new model for 200 epochs and store it as d5.pth in the models folder.
If a model has already been generated, then the training line can be commented out to skip training a new model.
It can be found by this comment "# Comment out training if model has already been trained"

**D6 task**
D6 is ready to run if all dependencies are installed

**D7 task**
Running D7 assumes a model from D1 has already been generated, otherwise D1 task needs to be ran first to get a model.
