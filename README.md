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

Dataset uploaded into google drive:
https://drive.google.com/drive/folders/1ZOAiFvJbrR6K4Gu7MzR4ybBk9Bu1EHzt?usp=sharing

# Choosing the right model
Tuning deep neural architectures to strike an optimal balance between accuracy and performance has been an area of active research for the last few years. This becomes even more challenging when you need to deploy the model to web services and still ensure it is high-performing.
In our project, we choose to use VGG16 because it provides decent performance based on our empirical experiments.

# Build and train the model
Doing transfer learning with pre-trained models on large datasets

# Transfer Learning
Transfer learning is a machine learning method where you start off using a pre-trained model, adapt and fine-tune it for other domains. For use cases such as Seeing AI, because the dataset is not large enough, starting with a pre-trained model, and further fine-tuning the model can reduce training time, and alleviate possible overfitting.

In practice, using transfer learning often requires you to "freeze" a few top layers' weights of a pre-trained model, then let the rest of the layers be trained normally (so back-propagation process can change their weights). Using Keras for transfer learning is quite easy – just set the trainable parameter of the layers which you want to freeze to False, and Keras will stop updating the parameters of those layers, while still back propagate the weights of the rest of the layers:

# Deploy the model
Deploy the model to web service.
For API, we want to run the models locally to handle requests, Exporting a Keras model to H5 file and then import it in server:
keras.models.load_model("Egyptian_Paper_Currency_Detector_Model.h5")
trained model : https://drive.google.com/file/d/1FUBtkGevccPRh2f5416abcRp1lZ9DFXH/view?usp=sharing 

# Deploy the model as a REST API
Export model as H5 file and create API using flask framework to recieve images and return model putput as JSON response.
therefore it will help developers to use the model in hardware or software
![Screenshot (105)](https://user-images.githubusercontent.com/45432562/92724195-8152ed80-f36a-11ea-90e1-8df1ac894a12.png)

# Further discussions

# Even faster models
Recently, a newer version of VGG16 was released, called VGG19.

# focus on evaluating the proposed system to recognize banknotes of different countries
Adding dollars and euros in our system

# Conclusion
Our proposed system is based on a modern solution type of deep learning CNN and that insure Performing the process as fast and robust as possible. 
The basic techniques utilized in our proposed system include image dataset generator, Convolution operations, features extraction, and finally classify the image based on Egyptian Paper currencies. The experimental results demonstrate that the proposed method can Recognize Egyptian paper money with high quality reaches 92% and in a short time.

# Further inquiries
The code is open-source on GitHub. If you have any questions, feel free to reach out to us at moneyinterperter2020@gmail.com , amr.nayer@gmail.com.
