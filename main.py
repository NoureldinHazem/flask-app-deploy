from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    tree_model = pickle.load(open('Decision_trees.sav', 'rb'))
    min_max_scaler = pickle.load(open('scaling_tree.sav', 'rb'))

    if request.method == 'POST':

        arr = np.array([])
        arr = np.append(arr, request.form['borrowerAPR'])
        arr = np.append(arr, request.form['borrowerRate'])
        arr = np.append(arr, request.form['lenderYield'])
        arr = np.append(arr, request.form['estimatedReturn'])
        arr = np.append(arr, request.form['prosperRating'])
        arr = np.append(arr, request.form['prosperScore'])
        arr = np.append(arr, request.form['creditScoreRangeLower'])
        arr = np.append(arr, request.form['creditScoreRangeUpper'])
        arr = np.append(arr, request.form['bankcardUtilization'])
        arr = np.append(arr, request.form['availableBankcardCredit'])
        arr = np.append(arr, request.form['statedMonthlyIncome'])
        arr = np.append(arr, request.form['prosperPaymentsOneMonthPlusLate'])
        arr = np.append(arr, request.form['loanMonthsSinceOrigination'])
        arr = np.append(arr, request.form['loanOriginalAmount'])
        arr = np.append(arr, request.form['monthlyLoanPayment'])
        arr = np.append(arr, request.form['lP_CustomerPrincipalPayments'])
        arr = np.append(arr, request.form['lP_ServiceFees'])

        arr = np.reshape(arr, (1, -1))

        normalized_arr = min_max_scaler.fit_transform(arr)
        my_prediction = tree_model.predict(normalized_arr)

    return render_template('results.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
