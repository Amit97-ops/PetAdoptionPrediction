# Pet Adoption Rate Classifier

## Project Intro

This was a group project built within my Machine Learning class at UC Davis. Project goals included developing models and algorithms to predict the adoptibility of pets in an animal shelter given their online pet profile. The aim of the final AI tool is to aid shelters and rescuers on improving the pet's online profile to decrease euthanization rates.   

One of the main challenges we faced was choosing which variables to use with specific models. We conducted thorough research of the variables we were provided with to test the dependence of the output and a single variable. We were able to reduce the number of variables we provided the model and optimize its performance.

To fully implement the project, the technologies we used were Python, Numpy, Pandas, Jupyter Notebook, and multiple graphics libraries to display the data. 

## Data Visualization and Modelling

![image](https://github.com/Amit97-ops/PetAdoptionPrediction/blob/master/t-sne_final.png)

t-SNE was used to map the multi-dimensional data into 2 dimensions. The output was supposed to cluster outputs based on the similarity of data points with multiple features. As seen above, the outputs did not cluster as well as we'd hoped. This led us to believe that a high prediction rate would be difficult to achieve.

![image](https://github.com/Amit97-ops/PetAdoptionPrediction/blob/master/Final%20pr%20curve.png)

The Figure shows that the AUC of the LightGBM classifiers is in most cases, larger than the other two models. Class 4 (purple curve) shows the most significant difference in AUC between models. This is because Class 4 has the largest amount of samples among the five adoption time categories.

## Setup Instructions

Note: Do not forget to unzip sentiment data

Code explanation:
	If you want to create a new model, use training_model.py
	If you want to continue training an existing model, use refine_model.py. Be sure to save the model with a different name.


