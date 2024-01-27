import pandas as pd
from tkinter import ttk
data = pd.read_excel(r'C:\Users\rgukt\Desktop\New folder\Placement.xlsx')
import warnings
warnings.filterwarnings('ignore')
data.head()
data.tail()
data.shape
data.info()
data.isnull().sum()
data.describe()
data.columns
data['status'].unique()
data['status'].value_counts()
data.columns
data[(data['degree_t']=="Sci&Tech") & (data['status']=="Placed")].sort_values(by="salary",ascending=False).head()
data = data.drop(['sl_no','salary'],axis=1)
data['ssc_b'] = data['ssc_b'].map({'Central':1,'Others':0})
data['hsc_b'] = data['hsc_b'].map({'Central':1,'Others':0})
data['hsc_s'] = data['hsc_s'].map({'Science':2,'Commerce':1,'Arts':0})
data['degree_t'] = data['degree_t'].map({'Sci&Tech':2,'Comm&Mgmt':1,'Others':0})
data['specialisation'] =data['specialisation'].map({'Mkt&HR':1,'Mkt&Fin':0})
data['workex'] = data['workex'].map({'Yes':1,'No':0})
data['status'] = data['status'].map({'Placed':1,'Not Placed':0})
X = data.drop('status',axis=1)
y= data['status']
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
lr = LogisticRegression()
lr.fit(X_train,y_train)

svm = svm.SVC()
svm.fit(X_train,y_train)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rf.predict(X_test)
y_pred6 = gb.predict(X_test)
from sklearn.metrics import accuracy_score
score1=accuracy_score(y_test,y_pred1)
score2=accuracy_score(y_test,y_pred2)
score3=accuracy_score(y_test,y_pred3)
score4=accuracy_score(y_test,y_pred4)
score5=accuracy_score(y_test,y_pred5)
score6=accuracy_score(y_test,y_pred6)
print(score1,score2,score3,score4,score5,score6)
final_data = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
            'ACC':[score1*100,
                  score2*100,
                  score3*100,
                  score4*100,
                  score5*100,score6*100]})
final_data
import seaborn as sns
sns.barplot(x=final_data['Models'], y=final_data['ACC'])

new_data = pd.DataFrame({
    'gender':0,
    'ssc_p':67.0,
    'ssc_b':0,
    'hsc_p':91.0,
    'hsc_b':0,
    'hsc_s':1,
    'degree_p':58.0,
    'degree_t':2,
    'workex':0,
    'etest_p':55.0,
     'specialisation':1,
    'mba_p':58.8,   
},index=[0])
lr= LogisticRegression()
lr.fit(X,y)
p=lr.predict(new_data)
prob=lr.predict_proba(new_data)
if p==1:
    print('Placed')
    print(f"You will be placed with probability of {prob[0][1]:.2f}")
else:
    print("Not-placed")
prob
import joblib
joblib.dump(lr,'model_campus_placement')
model = joblib.load('model_campus_placement')
model.predict(new_data)
from tkinter import *
import joblib
import pandas as pd
import tkinter.font as font

def show_entry_fields():
    text = clicked.get()
    if text == "Male":
        p1 = 1
    else:
        p1 = 0
    p2 = float(e2.get())
    text = clicked1.get()
    if text == "Central":
        p3 = 1
    else:
        p3 = 0
    p4 = float(e4.get())
    text = clicked6.get()
    if text == "Central":
        p5 = 1
    else:
        p5 = 0
    text = clicked2.get()
    if text == "Science":
        p6 = 2
    elif text == "Commerce":
        p6 = 1
    else:
        p6 = 0
    p7 = float(e7.get())
    text = clicked3.get()
    if text == "Sci&Tech":
        p8 = 2
    elif text == "Comm&Mgmt":
        p8 = 1
    else:
        p8 = 0
    text = clicked4.get()
    if text == "Yes":
        p9 = 1
    else:
        p9 = 0
    p10 = float(e10.get())
    text = clicked5.get()
    if text == "Mkt&HR":
        p11 = 1
    else:
        p11 = 0
    p12 = float(e12.get())

    model = joblib.load('model_campus_placement')
    new_data = pd.DataFrame({
        'gender': p1,
        'ssc_p': p2,
        'ssc_b': p3,
        'hsc_p': p4,
        'hsc_b': p5,
        'hsc_s': p6,
        'degree_p': p7,
        'degree_t': p8,
        'workex': p9,
        'etest_p': p10,
        'specialisation': p11,
        'mba_p': p12,
    }, index=[0])
    
    result = model.predict(new_data)
    result_prob = model.predict_proba(new_data)

    if result[0] == 0:
        Label(master, text="Can't Placed - Work hard to get placed", bg="red", fg="white", font=("Arial", 20)).grid(
            row=31, columnspan=2)
    else:
        placement_probability = result_prob[0][1] * 100

        if placement_probability < 30:
            status = "Student won't get placed"
        elif placement_probability < 40:
            status = f"Student will get placed with probability of {placement_probability:.2f}%"
        elif placement_probability < 50:
            status = f"Student will get placed with probability of {placement_probability:.2f}%"
        elif placement_probability < 60:
            status = f"Student will get placed with probability of {placement_probability:.2f}%"
        elif placement_probability < 70:
            status = f"Student will get placed with probability of {placement_probability:.2f}%"
        elif placement_probability < 80:
            status = f"Student will get placed with probability of {placement_probability:.2f}%"
        elif placement_probability < 90:
            status = f"Student will get placed with probability of {placement_probability:.2f}%"
        else:
            status = "Student will get placed with probability of 100%"

        Label(master, text=f"Student Status: {status}", bg="green", fg="white", font=("Arial", 15)).grid(row=31,
                                                                                                          columnspan=2)
        Label(master, text=f"Placement Probability: {placement_probability:.2f}%", font=("Arial", 15)).grid(row=33)
        Label(master, text="Percent", font=("Arial", 15)).grid(row=34)

# ... rest of the code remains unchanged

# ... rest of the code remains unchanged
master = Tk()
master.title("Campus Placement Prediction System")


label = Label(master, text = "Campus Placement Prediction System"
                          , bg = "green", fg = "white",font=("Arial", 20)) \
                               .grid(row=0,columnspan=2)


Label(master, text="Gender",font=("Arial", 15)).grid(row=1)
Label(master, text="Secondary Education percentage- 10th Grade",font=("Arial", 15)).grid(row=2)
Label(master, text="Board of Education",font=("Arial", 15)).grid(row=3)
Label(master, text="Higher Secondary Education percentage- 12th Grade",font=("Arial", 15)).grid(row=4)
Label(master, text="Board of Education",font=("Arial", 15)).grid(row=5)
Label(master, text="Specialization in Higher Secondary Education",font=("Arial", 15)).grid(row=6)
Label(master, text="Degree Percentage",font=("Arial", 15)).grid(row=7)
Label(master, text="Under Graduation(Degree type)- Field of degree education",font=("Arial", 15)).grid(row=8)
Label(master, text="Work Experience",font=("Arial", 15)).grid(row=9)
Label(master, text="Enter test percentage",font=("Arial", 15)).grid(row=10)
Label(master, text="branch specialization",font=("Arial", 15)).grid(row=11)
Label(master, text="Masters percentage",font=("Arial", 15)).grid(row=12)
clicked = StringVar()
options = ["Male","Female"]

clicked1 = StringVar()
options1 = ["Central","STATE","Others"]

clicked2 = StringVar()
options2 = ["Science","Commerce","Arts"]

clicked3 = StringVar()
options3 = ["Sci&Tech","Comm&Mgmt","Others"]

clicked4 = StringVar()
options4 = ["Yes","No"]

clicked5 = StringVar()
options5 = ["ECE","CSE","CIVIL","MECH","Mky&Fin"]

clicked6 = StringVar()
options6 = ["Central","STATE","Others"]
e1 = ttk.Combobox(master, textvariable=clicked)
e1['values'] = options
e1.config(style="TCombobox", font=("Arial", 12))
e1.set(options[0])
e1['state'] = 'readonly'
e2 = Entry(master)
e3 = ttk.Combobox(master, textvariable=clicked1)
e3['values'] = options1
e3.config(style="TCombobox", font=("Arial", 12))
e3.set(options1[0])
e3['state'] = 'readonly'
e4 = Entry(master)
e5 = ttk.Combobox(master, textvariable=clicked6)
e5['values'] = options6
e5.config(style="TCombobox", font=("Arial", 12))
e5.set(options6[0])
e5['state'] = 'readonly'

e6 = ttk.Combobox(master, textvariable=clicked2)
e6['values'] = options2
e6.config(style="TCombobox", font=("Arial", 12))
e6.set(options2[0])
e6['state'] = 'readonly'
e7 = Entry(master)
e8 = ttk.Combobox(master, textvariable=clicked3)
e8['values'] = options3
e8.config(style="TCombobox", font=("Arial", 12))
e8.set(options3[0])
e8['state'] = 'readonly'

e9 = ttk.Combobox(master, textvariable=clicked4)
e9['values'] = options4
e9.config(style="TCombobox", font=("Arial", 12))
e9.set(options4[0])
e9['state'] = 'readonly'
e10 = Entry(master)
e11 = ttk.Combobox(master, textvariable=clicked5)
e11['values'] = options5
e11.config(style="TCombobox", font=("Arial", 12))
e11.set(options5[0])
e11['state'] = 'readonly'
e12 = Entry(master)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
buttonFont = font.Font(family='Helvetica', size=16, weight='bold')
Button(master, text='Predict',height= 1, width=8,activebackground='#00ff00',font=buttonFont,bg='black', fg='white',command=show_entry_fields).grid()

mainloop()

