# Egyptian-paper-currency-detection
Transform Egyptian paper currency image into sound by using Deep learning and transfer learning concept 
# Introduction
Modern automation systems in the real-world require a system for currency recognition. 
It has various potential applications including banknote counting machines, money exchange machines, electronic banking, currency monitoring systems, assist blind persons, etc. The recognition of currency is a very important Need for Blind and visually impaired people. 
They are not being able to differentiate between currencies correctly.
It is very easy for them to be cheated by others. 
Therefore, there is an urgent need to design a system to recognize the value of currencies easily regardless of rotation, illumination, scaling and other Factors that may reduce the quality of the currency such as noisy, wrinkled and striped currencies
# Generating the currency detection model
In this section you will learn how to build, train and deploy a currency detection model to Azure and the intelligent edge.
# Dataset preparation and pre-processing
In this section, we will share how you can create the dataset needed for training the model.
In our case, the dataset consists of 14 classes. These  14 classes denoting the different denominations (inclusive both the front and back of the currency note). Each class has around 7500 images (with notes placed in various places and in different angles, see below), You can easily create the dataset needed for training in half an hour with your phone.
Steps:
1. Record a vedio for each class
2. Extract frames 
3. Save each class in seprate folder

![Screenshot (104)](https://user-images.githubusercontent.com/45432562/92722223-8c584e80-f367-11ea-9ac0-330247b84c57.png)


