# Projek Akhir Project-Based Virtual Intern : Data Scientist ID/X Partners x Rakamin Academy
# Nama: Imam Waliyuddin Rabbani

# Import Library

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import joblib
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Business Understanding

'''Dalam kasus prediksi risiko kredit, melibatkan identifikasi tujuan bisnis, yang dalam hal ini adalah untuk mengurangi risiko kredit yang tidak terbayar, 
meningkatkan pengambilan keputusan kredit, dan meminimalkan kerugian perusahaan. Selanjutnya, definisi masalah, yaitu bagaimana memprediksi risiko kredit 
dari pelanggan yang mengajukan pinjaman. Untuk memahami kebutuhan dan perspektif pemangku kepentingan, diperlukan komunikasi dengan tim manajemen perusahaan, 
tim risiko, dan tim keuangan. Kriteria keberhasilan dapat didefinisikan sebagai peningkatan akurasi prediksi risiko kredit dan pengurangan risiko kredit yang 
tidak terbayar.'''

# Analytic Approach

'''Analytic approach yang digunakan adalah deskriptif, diagnosis/statistical, dan prediktif. Untuk deskriptif, akan dilihat bagaimana pesebaran data yang ada, 
nilai max dan min, dsb. Lalu untuk statistical, akan dilihat bagaimana hubungan korelasi semua variabel dengan variabel target.
Lalu akan dibuat model menggunakan deep learning jika data yang ada cukup besar dan akan dilakukan prediksi untuk credit risk itu sendiri.'''

# Data Requirements
'''
 - id = A unique LC assigned ID for the loan listing.
 - member_id = A unique LC assigned Id for the borrower member.
 - loan_amnt = Last month payment was received.
 - funded_amnt = The total amount committed to that loan at that point in time.
 - funded_amnt_inv = ?.
 - term = The number of payments on the loan. Values are in months and can be either 36 or 60.
 - int_rate = Indicates if income was verified by LC, not verified, or if the income source was verified.
 - installment = The monthly payment owed by the borrower if the loan originates.
 - grade = LC assigned loan grade.
 - sub_grade = LC assigned loan subgrade.
 - emp_title = The job title supplied by the Borrower when applying for the loan.*.
 - emp_length = Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. .
 - home_ownership = The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
 - annual_inc = The self-reported annual income provided by the borrower during registration.
 - verification_status = ?.
 - issue_d = The month which the loan was funded.
 - loan_status = Current status of the loan.
 - pymnt_plan = ?.
 - url = URL for the LC page with listing data.
 - desc = Loan description provided by the borrower.
 - purpose = A category provided by the borrower for the loan request. .
 - title = The loan title provided by the borrower.
 - zip_code = The first 3 numbers of the zip code provided by the borrower in the loan application.
 - addr_state = The state provided by the borrower in the loan application.
 - dti = ?.
 - delinq_2yrs = The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.
 - earliest_cr_line = The month the borrower's earliest reported credit line was opened.
 - inq_last_6mths = The number of inquiries in past 6 months (excluding auto and mortgage inquiries).
 - mths_since_last_delinq = The number of months since the borrower's last delinquency.
 - mths_since_last_record = The number of months since the last public record.
 - open_acc = The number of open credit lines in the borrower's credit file.
 - pub_rec = Number of derogatory public records.
 - revol_bal = Total credit revolving balance.
 - revol_util = Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.        .
 - total_acc = The total number of credit lines currently in the borrower's credit file.
 - initial_list_status = The initial listing status of the loan. Possible values are – Whole, Fractional.
 - out_prncp = Remaining outstanding principal for total amount funded.
 - out_prncp_inv = Remaining outstanding principal for portion of total amount funded by investors.
 - total_pymnt = Payments received to date for total amount funded.
 - total_pymnt_inv = Payments received to date for portion of total amount funded by investors.
 - total_rec_prncp = Principal received to date.
 - total_rec_int = Interest received to date.
 - total_rec_late_fee = Late fees received to date.
 - recoveries = Indicates if a payment plan has been put in place for the loan.
 - collection_recovery_fee = post charge off collection fee.
 - last_pymnt_d = Last month payment was received.
 - last_pymnt_amnt = Last total payment amount received.
 - next_pymnt_d = Next scheduled payment date.
 - last_credit_pull_d = ?.
 - collections_12_mths_ex_med = Number of collections in 12 months excluding medical collections.
 - mths_since_last_major_derog = Months since most recent 90-day or worse rating.
 - policy_code = publicly available policy_code=1,new products not publicly available policy_code=2.
 - application_type = Indicates whether the loan is an individual application or a joint application with two co-borrowers.
 - annual_inc_joint = The combined self-reported annual income provided by the co-borrowers during registration.
 - dti_joint = A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income.
 - verification_status_joint = ?.
 - acc_now_delinq = The number of accounts on which the borrower is now delinquent.
 - tot_coll_amt = Total collection amounts ever owed.
 - tot_cur_bal = Total current balance of all accounts.
 - open_acc_6m = Number of open trades in last 6 months.
 - open_il_6m = Number of installment accounts opened in past 12 months.
 - open_il_12m = ?.
 - open_il_24m = Number of installment accounts opened in past 24 months.
 - mths_since_rcnt_il = Months since most recent installment accounts opened.
 - total_bal_il = Total current balance of all installment accounts.
 - il_util = Ratio of total current balance to high credit/credit limit on all install acct.
 - open_rv_12m = Number of revolving trades opened in past 12 months.
 - open_rv_24m = Number of revolving trades opened in past 24 months.
 - max_bal_bc = Maximum current balance owed on all revolving accounts.
 - all_util = Balance to credit limit on all trades.
 - total_rev_hi_lim = ?.
 - inq_fi = Number of personal finance inquiries.
 - total_cu_tl = Number of finance trades.
 - inq_last_12m = Number of credit inquiries in past 12 months.
'''

# Data Collection
data = pd.read_csv("loan_data_2007_2014.csv")
data = data.drop('no', axis=1)
data = data.dropna(axis=1, how='all')
data.head()


# # Data Understanding
data.info()

data.shape

data.describe()

data.dtypes

data.columns

# Data Preparation
column_counts = data.count()
selected_columns = column_counts[column_counts > 400000].index
new_data = data[selected_columns]

data_cleaned = new_data.dropna()
data_cleaned.reset_index(drop=True, inplace=True)
data_cleaned.head()

data_cleaned = data_cleaned.drop(columns=['id','member_id','emp_title','issue_d','url','title','zip_code','addr_state','earliest_cr_line','last_pymnt_d','last_credit_pull_d','policy_code','application_type'])

loan_status_mapping = {
    'Fully Paid': 0, 
    'Does not meet the credit policy. Status:Fully Paid': 0, 
    'Current': 1, 
    'Charged Off': 2, 
    'Does not meet the credit policy. Status:Charged Off':2, 
    'Default': 3, 
    'Late (16-30 days)': 4, 
    'Late (31-120 days)': 4,
    'In Grace Period': 5 
}
data_cleaned['loan_status'] = data_cleaned['loan_status'].replace(loan_status_mapping)

object_columns = data_cleaned.select_dtypes(include=['object'])

for col in object_columns.columns:
    unique_values = data_cleaned[col].unique()
    text_to_numeric = {value: index for index, value in enumerate(unique_values)}
    data_cleaned[col] = data_cleaned[col].replace(text_to_numeric)

data_cleaned.info()

data_cleaned.head()

# Exploratory Data Analysis
df_hist =data_cleaned.copy()
df_hist.hist(figsize=(30,30))
plt.show()

data_cleaned = data_cleaned.drop(columns=['annual_inc','pymnt_plan','pub_rec','revol_bal','total_rec_late_fee','recoveries','collection_recovery_fee','collections_12_mths_ex_med','acc_now_delinq','delinq_2yrs','inq_last_6mths'])

correlation = data_cleaned.corrwith(data_cleaned['loan_status'])

plt.figure(figsize=(20, 16))
plt.style.use('ggplot')

ax = correlation.plot(kind='bar', color='skyblue')
plt.title("Korelasi antara kolom '{}' dengan kolom lain:".format('jii'))
plt.xlabel('Kolom')
plt.ylabel('Korelasi')

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(correlation):
    ax.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom', fontsize=9)

plt.show()

plt.figure(figsize = (30,20))
sns.heatmap(data_cleaned.corr(), annot=True, square=True, fmt='.2f')
plt.title('Correlation', fontsize = 20)
plt.show()

fig = px.histogram(data_cleaned, x="loan_status")
fig.update_layout(bargap=0.7)
fig.show()

# Model Building
columns_to_normalize = [col for col in data_cleaned.columns if col != 'loan_status']
scaler = MinMaxScaler(feature_range=(-1, 1))
data_cleaned[columns_to_normalize] = scaler.fit_transform(data_cleaned[columns_to_normalize])

joblib.dump(scaler, "scaler.pkl")

x= data_cleaned.copy()
x.drop('loan_status',axis = 1,inplace = True)
x = x.to_numpy()
y = data_cleaned['loan_status']
y=y.to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)

X_train.shape, X_val.shape, X_test.shape

y_train = keras.utils.to_categorical(y_train, 6)
y_val = keras.utils.to_categorical(y_val, 6)

# Training
model = keras.Sequential([
    layers.Input(shape=(24,)), 
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2,verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()

train_accuracy = history.history['accuracy']  
val_accuracy = history.history['val_accuracy']
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)

plt.tight_layout()
plt.show()

## Testing
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}')

# Model Evaluation
y_pred = model.predict(X_test)

y_predict = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_predict)

cm = confusion_matrix(y_test, y_predict)

report = classification_report(y_test, y_predict, zero_division=1)
print(report)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (6 Classes)")
plt.show()

# Model Deployment
model.save("my_model.h5")

# Github Repo: https://github.com/ImamWaliyuddin/Data-Scientist-ID-X-Partners
# Dashboard Link: https://data-scientist-id-x-partners-imam.streamlit.app/

