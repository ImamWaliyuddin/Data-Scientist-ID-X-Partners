import streamlit as st
import tensorflow as tf 
import numpy as np
import joblib

# Muat model yang telah Anda latih
loaded_model = tf.keras.models.load_model("/my_model.h5")
scaler = joblib.load("/scaler.pkl")

# Fungsi untuk melakukan prediksi
def predict(model, input_data):
    input_data = np.array(input_data).reshape(1, -1) 
    input_data = scaler.transform(input_data)
    predictions = model.predict(input_data)
    return predictions

def classes(predict):
    if predict == 0:
        return "Fully Paid"
    elif predict == 1:
        return "Current"
    elif predict == 2:
        return "Charged Off"
    elif predict == 3:
        return "Default"
    elif predict == 4:
        return "Late"
    elif predict == 5:
        return "In Grace Period"

# Judul dashboard
st.title("Prediksi Credit Risk")

# Input dari pengguna
st.header("Input Nilai")
# Misalnya, Anda dapat membuat input untuk beberapa fitur
loan_amnt = st.number_input("Loan Amount", min_value=None, max_value=None, step=None)
funded_amnt = st.number_input("Funded Amount", min_value=None, max_value=None, step=None)
funded_amnt_inv = st.number_input("Funded Amount Investor", min_value=None, max_value=None, step=None)
term = st.number_input("Term (60 months = 0, 36 months = 1)", min_value=None, max_value=None, step=None)
int_rate = st.number_input("Interest Rate", min_value=None, max_value=None, step=None)
installment = st.number_input("Installment", min_value=None, max_value=None, step=None)
grade = st.number_input("Grade (C = 0, B = 1, A = 2, E = 3, D = 4, F = 5, G = 6)", min_value=None, max_value=None, step=None)
sub_grade = st.number_input("Sub Grade (C4 = 0, C1 = 1, B5 = 2, A4 = 3, C5 = 4, E1 = 5, C3 = 6, B1 = 7, B2 = 8, D1 = 9, A1 = 10, B3 = 11, B4 = 12, D2 = 13, A5 = 14, A3 = 15, A2 = 16, E4 = 17, D3 = 18, C2 = 19, F2 = 20, D4 = 21, F3 = 22, E3 = 23, F4 = 24, F1 = 25, D5 = 26, E5 = 27, G4 = 28, E2 = 29, G3 = 30, G2 = 31, G1 = 32, F5 = 33, G5 = 34)", min_value=None, max_value=None, step=None)
emp_length = st.number_input("Employment Length (< 1 year = 0, 10+ years = 1, 1 year = 2, 3 years = 3, 8 years = 4, 9 years = 5, 5 years = 6, 4 years = 7, 6 years = 8, 2 years = 9, 7 years = 10)", min_value=None, max_value=None, step=None)
home_ownership = st.number_input("Home Ownership (RENT = 0, OWN = 1, MORTGAGE = 2, OTHER = 3, NONE = 4, ANY = 5)", min_value=None, max_value=None, step=None)
verification_status = st.number_input("Verification Status (Source Verified = 0, Not Verified = 1, Verified = 2)", min_value=None, max_value=None, step=None)
purpose = st.number_input("Purpose (car = 0, other = 1, wedding = 2, debt_consolidation = 3, credit_card = 4, home_improvement = 5, major_purchase = 6, medical = 7, moving = 8, small_business = 9, vacation = 10, house = 11, renewable_energy = 12, educational = 13)", min_value=None, max_value=None, step=None)
dti = st.number_input("dti", min_value=None, max_value=None, step=None)
open_acc = st.number_input("Open Acc", min_value=None, max_value=None, step=None)
revol_util = st.number_input("Revolving line utilization rate", min_value=None, max_value=None, step=None)
total_acc = st.number_input("Total Acc", min_value=None, max_value=None, step=None)
initial_list_status = st.number_input("Initial List Status (f = 0, w = 1)", min_value=None, max_value=None, step=None)
out_prncp = st.number_input("Outstanding Principal", min_value=None, max_value=None, step=None)
out_prncp_inv  = st.number_input("Outstanding Principal Investor", min_value=None, max_value=None, step=None)
total_pymnt = st.number_input("Total Payment", min_value=None, max_value=None, step=None)
total_pymnt_inv = st.number_input("Total Payment Investor", min_value=None, max_value=None, step=None)
total_rec_prncp = st.number_input("Principal Received", min_value=None, max_value=None, step=None)
total_rec_int = st.number_input("Interest Received", min_value=None, max_value=None, step=None)
last_pymnt_amnt = st.number_input("Last Total Payment Amount", min_value=None, max_value=None, step=None)


# Tombol prediksi
if st.button("Prediksi"):
    
    input_data = [loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate,
       installment, grade, sub_grade, emp_length, home_ownership,
       verification_status, purpose, dti, open_acc,
       revol_util, total_acc, initial_list_status, out_prncp,
       out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp,
       total_rec_int, last_pymnt_amnt] 
    
    predictions = predict(loaded_model, input_data)
    predicted_class = predictions.argmax()
    result = classes(predicted_class)
    st.header("Hasil Prediksi")
    st.write("Prediksi:", result)
