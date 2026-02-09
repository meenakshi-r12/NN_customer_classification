# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model



## DESIGN STEPS

### STEP 1:
Loading the dataset

### STEP 2:
Split the dataset into training and testing

### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:
Build the Neural Network Model and compile the model.

### STEP 5:
Train the model with the training data.

### STEP 6:
Plot the performance plot

### STEP 7:
Evaluate the model with the testing data.


## PROGRAM

### Name: MEENAKSHI.R
### Register Number:212224220062

```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x)) 
    x=self.relu(self.fc2(x))
    x=self.fc3(x)  
    return x


# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```


## Dataset Information

<img width="206" height="526" alt="418444087-2b45a519-d54f-410a-9b12-6c909e64d249" src="https://github.com/user-attachments/assets/051cee6d-5952-4ce8-80da-dc99aa5e70bd" />

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="736" height="564" alt="418445523-6a8454da-97d8-4522-99a1-fddc33a19d50" src="https://github.com/user-attachments/assets/e5ffc443-2dc5-4307-9bf7-e4691d053959" />


### New Sample Data Prediction

<img width="1007" height="146" alt="418445934-ea925829-8559-4613-98ff-ba6d34eb8152" src="https://github.com/user-attachments/assets/1d1000c1-d38b-492d-b1b8-5d58a67dc30d" />

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
