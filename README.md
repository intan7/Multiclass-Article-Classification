# Multiclass Article Classification
> This is the deep learning model to categorize the ***unseen 
articles into 5 categories namely Sport, Tech, Business, Entertainment and 
Politics***.

# Descriptions
>Text documents are essential as they are one of the richest sources of data for 
businesses. Text documents often contain crucial information which might shape 
the market trends or influence the investment flows. Therefore, companies often 
hire analysts to monitor the trend via articles posted online, tweets on social media 
platforms such as Twitter or articles from newspaper. 

>However, some companies may wish to only focus on articles related to technologies and politics. Thus, 
filtering of the articles into different categories is required. Often the categorization of the articles is conduced manually and retrospectively; 
which causing the waste of time and resources due to this arduous task. To solve this issue, a deep learning model was created and able to generate the category in short time.

# 📙 Requirement
>> ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
>>
>>  - To run tensorboard, you need to copy your logs path and then go to cmd/terminal.
>>  - Then, type tensorboard --logdir "LOGS_PATH".(Please make sure you're in the right environment)
>>  - You may go to http://localhost:6006/ to access yout tensorboard.
![alt text](https://github.com/intan7/Multiclass-Article-Classification/blob/main/static/run_tensor_cmd.png)

# TensorBoard
![alt text](https://github.com/intan7/Multiclass-Article-Classification/blob/main/static/tensorboard_ss.png)

# Results
The model accuracy is 0.94 and this accuracy can be increased with more data collected, increase the epochs numbers and increase the layers.

![alt text](https://github.com/intan7/Multiclass-Article-Classification/blob/main/static/CR.png)

![alt text](https://github.com/intan7/Multiclass-Article-Classification/blob/main/static/CM.png)



## Powered by
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

## This project is able to successfully run thanks to
 >https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
