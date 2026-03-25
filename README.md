# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="951" height="956" alt="551934476-b5def89f-2cc1-466d-8eef-59d38f49bca2" src="https://github.com/user-attachments/assets/781b851b-0de8-4fbb-9979-d5f53ab782ee" />



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
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)


    def forward(self, x):
         x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train_model(model, train_loader, criterion, optimizer, epochs):
     for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name: MEENAKSHI R")
print("Register No: 212224220062")    
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)

# Select the prediction for the sample (first element)
predicted_class_index = torch.argmax(output[0]).item()
predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: MEENAKSHI R")
print("Register No: 212224220062")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')

```


## Dataset Information

<img width="1333" height="255" alt="548162868-1d1268c4-0700-42bc-b0ac-0724e54801b4" src="https://github.com/user-attachments/assets/51e39ab0-a3fe-4d65-96d6-b10a38d382e5" />

## OUTPUT

<img width="665" height="560" alt="548162568-2bcdd70f-e048-425f-885d-bc0c6fac1b27" src="https://github.com/user-attachments/assets/a33c7c13-5683-4bbc-b4ed-a0532736dbdc" />

## Confusion Matrix

```
Epoch [100/100], Loss: 1.0770
Name: MEENAKSHI R
Register No: 212224220062
Test Accuracy: 0.46%
Confusion Matrix:
 [[210 115 116 143]
 [120 151 152  67]
 [ 73  91 252  56]
 [129  42  46 376]]
```
## Classification Report
```
Classification Report:
               precision    recall  f1-score   support

           A       0.39      0.36      0.38       584
           B       0.38      0.31      0.34       490
           C       0.45      0.53      0.49       472
           D       0.59      0.63      0.61       593

    accuracy                           0.46      2139
   macro avg       0.45      0.46      0.45      2139
weighted avg       0.46      0.46      0.46      2139
```
## New Sample Data Prediction
```
Name: MEENAKSHI R 
Register No: 212224220062
Predicted class for sample input: D
Actual class for sample input: D

```



## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
