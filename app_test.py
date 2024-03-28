import pickle


import pandas as pd
import numpy as np
import warnings
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict_ADHD','Predict_OCD','Predict_SAD','Predict_Depression'])

if app_mode=='Home':
    st.write("Based on the current published journals in Plos One and HISS journals, this web application has been built to predict depression, OCD, SAD and ADHD with machine learning algortihms..")

    st.write("This app is suitable for 4-17 years children and adolescent. For the parents, caregivers or teachers who are unsure whether their child or adolescent is affected with these mental illnesses or not or whether they should consult regarding this issue with any mental health professional, they can use this app in order to get a primary idea about depression,ADHD,OCD,SAD detection.") 

elif app_mode=='Predict_ADHD':
    
# loading the model
    #path = '/Users/marzia'
    #modelname = path + '/adhdmodel.pkl'
    #modelname = 'https://github.com/UmmeMarziaHaque/MH-Streamlit-App/blob/master/adhdmodel.pkl'
    loaded_model = pickle.load(open('adhdmodel.pkl', 'rb'))

    st.write("If you want to check if your child or adolescent has ADHD based on this prediction model, please select the answer for the following symptoms.")

    st.write("Is your child or adolescent doing any of the following?")
        



    pada1ay = st.selectbox('Senses a lack of close parental attention',("Yes","No"))
    pada1y = st.selectbox('Faces trouble paying attention or following instructions or to stay focused',("Yes","No")) 
    pada2by = st.selectbox('Leaves seat while wishing to sit',("Yes","No"))
    pada2cy = st.selectbox('Runs about or climbs excessively',("Yes","No"))
    pada2ey = st.selectbox('Often on the go/driven by a motor',("Yes","No"))
    pada2y = st.selectbox('Behaves impulsive or shows aggressiveness or similar behaviour or does any constant activity like fidgeting, talking, tapping or constant moving',("Yes","No"))
    paday = st.selectbox('Does not appear to be paying attention even when addressed directly',("Yes","No"))
    if pada2ey == "Yes":
        pada2em = "Yes"
    else:
        pada2em = "No"


    if pada2y == "Yes":
        pada2m = "Yes"
    else:
        pada2m = "No"
    
    if paday == "Yes":
        padam = "Yes"
    else:
        padam = "No"


 # Pre-processing user input    
    if pada1ay == "Yes":
        pada1ay = 1
    else:
        pada1ay = 0
    
    if pada1y == "Yes":
        pada1y = 1
    else:
        pada1y = 0
    
    if pada2by == "Yes":
        pada2by = 1
    else:
        pada2by = 0  
    
    if pada2cy == "Yes":
        pada2cy = 1
    else:
        pada2cy = 0  
        
    if pada2ey == "Yes":
        pada2ey = 1
    else:
        pada2ey = 0 
    
    if pada2y == "Yes":
        pada2y = 1
    else:
        pada2y = 0 
    
    
    if paday == "Yes":
        paday = 1
    else:
        paday = 0
    
    if pada2em == "Yes":
        pada2em = 1
    else:
        pada2em = 0
    
    if pada2m == "Yes":
        pada2m = 1
    else:
        pada2m = 0
    
    if padam == "Yes":
        padam = 1
    else:
        padam = 0

    features = ([pada1ay, pada1y, pada2by,pada2cy,pada2ey,pada2y,paday,pada2em,pada2m,padam])

    features = np.array(features)
#label encode your categorical columns
#le = preprocessing.LabelEncoder()
#for i in range(len(categorical)):
    #features[:, i] = le.fit_transform(features[:, i])

#features = np.array(features)

#features = np.fromstring(features, dtype=float, sep=' ')
#features = np.fromstring(features, dtype=int, sep=' ')
    results = features.reshape(1, -1)




 
# when 'Predict' is clicked, make the prediction and store it 
    if st.button("Get Your Prediction"):
    #X = pd.DataFrame({'poco1a2y':[poco1a2y],'poco3a3y':[poco3a3y],'pocc1a2y':[pocc1a2y],'poco2cy':[poco2cy],'pocc1cy':[pocc1cy],'pocc2cy':[pocc2cy],'pocc3cy':[pocc3cy],'pocc4cy':[pocc4cy],'pocc4cm':[pocc4cm]}
    #prediction = (loaded_model.predict_proba(X)[:,1] >= 0.6).astype(bool)
        print(pada1ay)
        print(pada1y)
        print(pada2by)
        print(pada2cy)
        print(pada2ey)
        print(pada2y)
        print(paday)
        print(pada2em)
        print(pada2m)
        print(padam)
    #T_filtered = scaler.transform(T)
    
    # Making predictions            
        prediction = loaded_model.predict(results)
    #st.success('Your Target is {}'.format(prediction))
    

    #prediction = loaded_model.predict(results)
        if prediction[0] == 1:
            st.success('ADHD exists')
        else:
            st.error( 'ADHD does not exist')

elif app_mode=='Predict_OCD':
# loading the model
    #path = '/Users/marzia'
    #modelname = path + '/ocdmodel.pkl'
    #modelname = 'https://github.com/UmmeMarziaHaque/MH-Streamlit-App/blob/master/ocdmodel.pkl'
    loaded_model = pickle.load(open('ocdmodel.pkl', 'rb'))
 


    st.write("If you want to check if your child or adolescent has OCD based on this prediction model, please select the answer for the following symptoms.")

    st.write("Is your child or adolescent doing any of the following?")
        



    poco1a2y = st.selectbox('Not simply worry about real-life problems',("Yes","No"))
    poco3a3y = st.selectbox('Attempts to ignore or suppress thoughts',("Yes","No")) 
    pocc1a2y = st.selectbox('Cleaning: handwashing or cleaning something repeatedly',("Yes","No"))
    poco2cy = st.selectbox('Doing something bad or have some negative feelings: showing rudeness, aggressive behaviour or cannot control behaviour or feelngs of being different from friends and family',("Yes","No"))
    if pocc1a2y == "Yes":
        pocc1cy = "Yes"
    else:
        pocc1cy = "No"

    
    pocc2cy = st.selectbox('Checking: repeatedly checking any door lock, switches of appliances or something like this',("Yes","No"))
    pocc3cy = st.selectbox('Counting random things like numbers, steps taken,road signs,floor/ceiling tiles or counting words or letters in any sentence/page',("Yes","No"))
    
    if pocc2cy == "Yes":
        pocc4cy = "Yes"
    else:
        pocc4cy = "No"
    
    if pocc3cy == "Yes":
        pocc4cy = "Yes"
    else:
        pocc4cy = "No"
    if pocc2cy == "Yes":
        pocc4cm = "Yes"
    else:
        pocc4cm = "No"
    if pocc3cy == "Yes":
        pocc4cm = "Yes"
    else:
        pocc4cm = "No"

 # Pre-processing user input    
    if poco1a2y == "Yes":
        poco1a2y = 1
    else:
        poco1a2y = 0
    if poco3a3y == "Yes":
        poco3a3y = 1
    else:
        poco3a3y = 0
    if pocc1a2y == "Yes":
        pocc1a2y = 1
    else:
        pocc1a2y = 0  
    if poco2cy == "Yes":
        poco2cy = 1
    else:
        poco2cy = 0  
        
    if pocc1cy == "Yes":
        pocc1cy = 1
    else:
        pocc1cy = 0 
    if pocc2cy == "Yes":
        pocc2cy = 1
    else:
        pocc2cy = 0 
    if pocc3cy == "Yes":
        pocc3cy = 1
    else:
        pocc3cy = 0
    if pocc4cy == "Yes":
        pocc4cy = 1
    else:
        pocc4cy = 0
    if pocc4cm == "Yes":
        pocc4cm = 1
    else:
        pocc4cm = 0
    


    features = ([poco1a2y, poco3a3y, pocc1a2y,poco2cy,pocc1cy,pocc2cy,pocc3cy,pocc4cy,pocc4cm])

    features = np.array(features)
#label encode your categorical columns
#le = preprocessing.LabelEncoder()
#for i in range(len(categorical)):
    #features[:, i] = le.fit_transform(features[:, i])

#features = np.array(features)

#features = np.fromstring(features, dtype=float, sep=' ')
#features = np.fromstring(features, dtype=int, sep=' ')
    results = features.reshape(1, -1)




 
# when 'Predict' is clicked, make the prediction and store it 
    if st.button("Get Your Prediction"):
    #X = pd.DataFrame({'poco1a2y':[poco1a2y],'poco3a3y':[poco3a3y],'pocc1a2y':[pocc1a2y],'poco2cy':[poco2cy],'pocc1cy':[pocc1cy],'pocc2cy':[pocc2cy],'pocc3cy':[pocc3cy],'pocc4cy':[pocc4cy],'pocc4cm':[pocc4cm]}
    #prediction = (loaded_model.predict_proba(X)[:,1] >= 0.6).astype(bool)
        print(poco1a2y)
        print(poco3a3y)
        print(pocc1a2y)
        print(poco2cy)
        print(pocc1cy)
        print(pocc2cy)
        print(pocc3cy)
        print(pocc4cy)
        print(pocc4cm)
  
    #T_filtered = scaler.transform(T)
    
    # Making predictions            
        prediction = loaded_model.predict(results)
    #st.success('Your Target is {}'.format(prediction))
    

    #prediction = loaded_model.predict(results)
        if prediction[0] == 1:
            st.success('OCD exists')
        else:
            st.error( 'OCD does not exist')
        
elif app_mode=='Predict_SAD':        
# loading the model
    #path = '/Users/marzia'
    #modelname = path + '/sadmodel.pkl'
    #modelname = 'https://github.com/UmmeMarziaHaque/MH-Streamlit-App/blob/master/sadmodel.pkl'
    loaded_model = pickle.load(open('sadmodel.pkl', 'rb'))
 
 

    st.write("If you want to check if your child or adolescent has SAD based on this prediction model, please select the answer for the following symptoms.")

    st.write("Is your child or adolescent doing any of the following?")
        



    PSA001 = st.selectbox('Wants to stay at home and not go places without ATTACHMENT FIGURE:a person with a strong emotional connection',("Yes","No"))
    psaa4y = st.selectbox('Shows unwillingness to go to school/elsewhere because of fear of separation',("Yes","No")) 
    psaa6y = st.selectbox('Refuse to go to sleep away from attachment figure',("Yes","No"))
    psaay = st.selectbox('Anxiety about separation from home/attachment figures',("Yes","No"))

    if (psaa4y == "Yes") or (psaay == "Yes") :
        health_anyprofessional = "Yes"
    else:
        health_anyprofessional = "No"

    

    
    if PSA001 == "Yes":
        PSA001 = "Yes"
    else:
        PSA001 = "No"
    
    if psaa4y == "Yes":
        psaa4y = "Yes"
    else:
        psaa4y = "No"
    if psaa6y == "Yes":
        psaa6y = "Yes"
    else:
        psaa6y = "No"
    if psaay == "Yes":
        psaay = "Yes"
    else:
        psaay = "No"
    
    if health_anyprofessional == "Yes":
        health_anyprofessional = "Yes"
    else:
        health_anyprofessional = "No"

 # Pre-processing user input    
    if PSA001 == "Yes":
        PSA001 = 1
    else:
        PSA001 = 0
    
    if psaa4y == "Yes":
        psaa4y = 1
    else:
        psaa4y = 0
    
    if psaa6y == "Yes":
        psaa6y = 1
    else:
        psaa6y = 0  
    
    if psaay == "Yes":
        psaay = 1
    else:
        psaay = 0  
        
    if health_anyprofessional == "Yes":
        health_anyprofessional = 1
    else:
        health_anyprofessional = 0 

    


    features = ([PSA001, psaa4y, psaa6y,psaay,health_anyprofessional])

    features = np.array(features)
#label encode your categorical columns
#le = preprocessing.LabelEncoder()
#for i in range(len(categorical)):
    #features[:, i] = le.fit_transform(features[:, i])

#features = np.array(features)

#features = np.fromstring(features, dtype=float, sep=' ')
#features = np.fromstring(features, dtype=int, sep=' ')
    results = features.reshape(1, -1)




 
# when 'Predict' is clicked, make the prediction and store it 
    if st.button("Get Your Prediction"):
    #X = pd.DataFrame({'poco1a2y':[poco1a2y],'poco3a3y':[poco3a3y],'pocc1a2y':[pocc1a2y],'poco2cy':[poco2cy],'pocc1cy':[pocc1cy],'pocc2cy':[pocc2cy],'pocc3cy':[pocc3cy],'pocc4cy':[pocc4cy],'pocc4cm':[pocc4cm]}
    #prediction = (loaded_model.predict_proba(X)[:,1] >= 0.6).astype(bool)
        print(PSA001)
        print(psaa4y)
        print(psaa6y)
        print(psaay)
        print(health_anyprofessional)
   
  
    #T_filtered = scaler.transform(T)
    
    # Making predictions            
        prediction = loaded_model.predict(results)
    #st.success('Your Target is {}'.format(prediction))
    

    #prediction = loaded_model.predict(results)
        if prediction[0] == 1:
            st.success('SAD exists')
        else:
            st.error( 'SAD does not exist')
            
elif app_mode=='Predict_Depression':        
# loading the model
    #path = '/Users/marzia'
    #modelname = path + '/depressionmodel.pkl'
    #modelname = 'https://github.com/UmmeMarziaHaque/MH-Streamlit-App/blob/master/depressionmodel.pkl'
    loaded_model = pickle.load(open('depressionmodel.pkl', 'rb'))
 
 

    st.write("If you want to check if your child or adolescent has SAD based on this prediction model, please select the answer for the following symptoms.")

    st.write("Is your child or adolescent doing any of the following?")
        



    psaa8y = st.selectbox('Appears sad or complain of any medical symptoms (insomnia,hypersomnia,sickness, headaches,muscle pains)  as a result of recent issues such as family composition/parental separation/lack of adult supervision/bullying/ racism/school violence, or something similar?',("Yes","No"))
    pmda7y = st.selectbox('Feels as though nothing was enjoyable, or worthlessness or guilt',("Yes","No"))
    pmda1y = st.selectbox('is in a bad temper or irritable mood',("Yes","No")) 
    pmda2y = st.selectbox('shows lessened/diminished enjoyment or lack of interest in anything',("Yes","No"))
    pmda3y = st.selectbox('Weight loss/gain/change of appetite',("Yes","No"))
    pmda4y = st.selectbox('Insomnia/hypersomnia',("Yes","No"))
    pmda5y = st.selectbox('psychomotor agitation/retardation',("Yes","No"))
    pmda6y = st.selectbox('Fatigue/energy loss',("Yes","No"))
   
    pmda8y = st.selectbox('Trouble focusing or deciding',("Yes","No"))
    pmda9y = st.selectbox('Considering or attempting suicide; having suicidal thoughts',("Yes","No"))
    pmday = st.selectbox('Occurrence of any one of these five symptoms within a two-week time frame',("Yes","No"))
   

    

    
    if psaa8y == "Yes":
        psaa8y = "Yes"
    else:
        psaa8y = "No"
    
    if pmda1y == "Yes":
        pmda1y = "Yes"
    else:
        pmda1y = "No"
    if pmda2y == "Yes":
        pmda2y = "Yes"
    else:
        pmda2y = "No"
    if pmda3y == "Yes":
        pmda3y = "Yes"
    else:
        pmda3y = "No"
    if pmda4y == "Yes":
        pmda4y = "Yes"
    else:
        pmda4y = "No"
    if pmda5y == "Yes":
        pmda5y = "Yes"
    else:
        pmda5y = "No"
    if pmda6y == "Yes":
        pmda6y = "Yes"
    else:
        pmda6y = "No"
    if pmda7y == "Yes":
        pmda7y = "Yes"
    else:
        pmda7y = "No"

    if pmda8y == "Yes":
        pmda8y = "Yes"
    else:
        pmda8y = "No"
    if pmda9y == "Yes":
        pmda9y = "Yes"
    else:
        pmda9y = "No"
        
    if pmday == "Yes":
        pmday = "Yes"
    else:
        pmday = "No"
 # Pre-processing user input    
    if psaa8y == "Yes":
        psaa8y = 1
    else:
        psaa8y = 0
    
    if pmda1y == "Yes":
        pmda1y = 1
    else:
        pmda1y = 0
    if pmda2y == "Yes":
        pmda2y = 1
    else:
        pmda2y = 0
    if pmda3y == "Yes":
        pmda3y = 1
    else:
        pmda3y = 0
    if pmda4y == "Yes":
        pmda4y = 1
    else:
        pmda4y = 0
    if pmda5y == "Yes":
        pmda5y = 1
    else:
        pmda5y = 0
    if pmda6y == "Yes":
        pmda6y = 1
    else:
        pmda6y = 0
    if pmda7y == "Yes":
        pmda7y = 1
    else:
        pmda7y = 0
    if pmda8y == "Yes":
        pmda8y = 1
    else:
        pmda8y = 0
    if pmda9y == "Yes":
        pmda9y = 1
    else:
        pmda9y = 0
        
    if pmday == "Yes":
        pmday = 1
    else:
        pmday = 0

    


    features = ([psaa8y, pmda1y, pmda2y,pmda3y,pmda4y,pmda5y,pmda6y,pmda7y,pmda8y,pmda9y,pmday])

    features = np.array(features)
#label encode your categorical columns
#le = preprocessing.LabelEncoder()
#for i in range(len(categorical)):
    #features[:, i] = le.fit_transform(features[:, i])

#features = np.array(features)

#features = np.fromstring(features, dtype=float, sep=' ')
#features = np.fromstring(features, dtype=int, sep=' ')
    results = features.reshape(1, -1)




 
# when 'Predict' is clicked, make the prediction and store it 
    if st.button("Get Your Prediction"):
    #X = pd.DataFrame({'poco1a2y':[poco1a2y],'poco3a3y':[poco3a3y],'pocc1a2y':[pocc1a2y],'poco2cy':[poco2cy],'pocc1cy':[pocc1cy],'pocc2cy':[pocc2cy],'pocc3cy':[pocc3cy],'pocc4cy':[pocc4cy],'pocc4cm':[pocc4cm]}
    #prediction = (loaded_model.predict_proba(X)[:,1] >= 0.6).astype(bool)
        print(psaa8y)
        print(pmda1y)
        print(pmda2y)
        print(pmda3y)
        print(pmda4y)
        print(pmda5y)
        print(pmda6y)
        print(pmda7y)
        print(pmda8y)
        print(pmda9y)
        print(pmday)
  
    #T_filtered = scaler.transform(T)
    
    # Making predictions            
        prediction = loaded_model.predict(results)
    #st.success('Your Target is {}'.format(prediction))
    

    #prediction = loaded_model.predict(results)
        if prediction[0] == 1:
            st.success('Depression exists')
           
        else:
            st.error( 'Depression does not exist')
