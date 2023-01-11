# Heart Disease Patients Classification
###### METIS Data Science and Machine Learning Bootcamp 2022 by Krystian Krystkowiak
###### project/month(4/7) focus: CLASSIFICATION
#### Code [GitHub](https://public.tableau.com/views/Metis-Project-3-Business-on-BGG/Dashboard4_1?:language=en-GB&:display_count=n&:origin=viz_share_link)
#### Presentation [GitHub](https://github.com/Krystkowiakk/Board-Games-Design-Accelerator/blob/f732eab3f9e0b51cd780f49bed0d67d01a78e181/Project%20Presentation/Krystkowiak_Krystian_Project_3_Business_on_BGG.pdf)
#### Web App [streamlit](https://krystkowiakk-metis-project-7--streamlit-appstreamlit-app-xuz5iy.streamlit.app)

ABSTRACT

- To predict heart disease patients, implemented and fine-tuned various classification models (KNN, Logistic Regression, Random Forest, XGBoost, Naive Bayes), utilizing techniques such as regularization, ensembling, and addressing class imbalance. Achieved Recall score of 86% with XGBoost.
- The goal of this project was to establish a meaningfully predictive classification model for identifying high-risk patients. A model with acceptable recall (85%), precision, accuracy and F1 that was utilized for more informed outreach. The project included trying, testing, and tuning multiple different classification models and class imbalance techniques. The findings were visualized and communicated during a presentation using tools such as Keynote and Tableau.

DESIGN

The project was designed to have potential clients including doctors and medical institutions, as well as a wide range of businesses where heart disease is a concern such as insurances, medical apps, fitness or nutritionist and could also be used by individuals conscious about their health.

The data set used for the project was sourced from the Behavioral Risk Factor Surveillance System (BRFSS), an annual telephone survey conducted by the CDC to gather data on the health status of US residents. BRFSS completes more than 400,000 adult interviews each year, and the most recent data (as of February 15, 2022) used in this project was from 2020.

The model developed in this project can help detect and prevent factors that have the greatest impact on heart disease, using machine learning methods to discover data regularities, which can help predict a patient's condition and raise red flags during an initial questionnaire.

DATA

The data used for this project was sourced from Kaggle, from a dataset that was initially cleaned by Kamil Pytlak. The dataset can be found at the following link: www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease.

The dataset includes over 319,000 observations, with each individual row having 19 columns. The columns include various indicators related to heart disease and health.

The target variable for the model is HeartDisease and other relevant data columns are BMI, Smoking, AlcoholDrinking, Stroke, AlcoholDrinking, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer.

ALGORITHMS

- Data preprocessing: Filtering and cleaning data, converting categorical features to binary dummy variables, and creating new features (not implemented in the final models).
- Validation and testing: The entire training dataset of 319795 records was split into train(231051)/val(40774)/test(47970). All scores were calculated on validation set and trained on the training set only. Predictions on the test set were limited to the very end, so this split was only used and scores seen just once.
- To improve the performance of the model, various techniques were applied such as regularization, ensembling, and handling class imbalance. As the goal of the model is sensitive, recall rate was used as the target classification rate (85%). To handle the class imbalance (8% of positive samples), undersampling, class weights and tresholds were used. Classification models like KNN, Logistic Regression, Random Forest, XGBoost, Naive Bayes were used.
- The final model chosen for the project was XGBoost, which achieved a recall of 0.846 and accuracy of 0.855 on the test set.

TOOLS

- Python Pandas and Numpy for data preprocessing and manipulation.
- Python packages for classification models such as scikit-learn (sklearn) and XGBoost.
- Python libraries such as Seaborn and Plotly for data visualization.
- Tableau and Keynote for creating interactive visualizations and communicating results to stakeholders.

COMMUNICATION

A 5-minute video presentation that provides an overview of the project, including key findings and visualizations.
A streamlit web app built in the next project to further communicate the results and allow user-interaction.

###### project/month(7/7) focus: DATA ENGINEERING
#### Heart Disease Patients Classication a Web App [GitHub](https://github.com/Krystkowiakk/Metis-Project-7-Engineering_on-Heart_Disease_Indicators)

![Heart Disease Patients Classification](files/cover.jpg)


