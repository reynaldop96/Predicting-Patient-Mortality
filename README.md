# Predicting-Patient-Mortality

In this project, I use ICU clinical data to predict patient mortality one month after discharge from the hospital. The available data has the following features:
- **patient_id**: unique patient identification number
- **event_id**: Encodes clinical events the patient has had. There are 3 overarching types of events: when the patient recieves a drug, when the patient recieved a diagnosis, and when the patient recieves a laboratory test result. There are further subdivded depending on the specific diagnosis or test result. 
- **event_description**: Describes the type of diagnosis or drug from event_id
- **timestamp**: Date when the event happened. 
- **value**: value associated with the event. Only meaningful value is for laboratory test events, which records the value for the lab result. 

There are 2 data files: events.csv details the events for each patient in the format (patient id, event id, event description, timestamp, value) and mortality_events.csv details the date on which the patient died (patient id, timestamp, label).

General procedure: 
1- Use HIVE to compute descriptive statistics of the data. 
2- Use Pig to transform the data into SVM Light format for more efficient data storage. This was ideal because there are hundreds of possible event_id types, and each patient only recieves a very small subset of all possible event_id's, resulting in a very sparse data set. 
3- Finally, I create a logistic regression classifier in Python and an ensemble logistic regression classifier using Hadoop. 

**Result**: AUC of 0.65 using logistic regression and AUC of 0.72 using the ensemble model.
