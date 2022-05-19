from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

model = pickle.load(open('finalized_model.sav', 'rb'))




app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', pagetitle="Home page")

@app.route('/blog')
def blog():
    return render_template('blog.html', pagetitle="Blog")


@app.route('/treatment')
def treatment():
    return render_template('treatment.html', pagetitle="treatment")


@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html', pagetitle="Diagnosis")


@app.route('/detect')
def detect():
    return render_template('detect.html', pagetitle="Detect")


@app.route('/contact')
def contact():
    return render_template('contact.html', pagetitle="Contact")


@app.route('/result',  methods=['POST'])
def result():
    data1 = request.form['radius_mean']
    data2 = request.form['texture_mean']
    data3 = request.form['perimeter_mean']
    data4 = request.form['area_mean']
    data5 = request.form['smoothness_mean']
    data6 = request.form['compactness_mean']
    data7 = request.form['concavity_mean']
    data8 = request.form['concave points_mean']
    data9 = request.form['symmetry_mean']
    data10 = request.form['fractal_dimension_mean']
    data11 = request.form['radius_se']
    data12 = request.form['texture_se']
    data13 = request.form['perimeter_se']
    data14 = request.form['area_se']
    data15 = request.form['smoothness_se']
    data16 = request.form['compactness_se']
    data17 = request.form['concavity_se']
    data18 = request.form['concave points_se']
    data19 = request.form['symmetry_se']
    data20 = request.form['fractal_dimension_se']
    data21 = request.form['radius_worst']
    data22 = request.form['texture_worst']
    data23 = request.form['perimeter_worst']
    data24 = request.form['area_worst']
    data25 = request.form['smoothness_worst']
    data26 = request.form['compactness_worst']
    data27 = request.form['concavity_worst']
    data28 = request.form['concave points_worst']
    data29 = request.form['symmetry_worst']
    data30 = request.form['fractal_dimension_worst']
    


   
    arr1 = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12,
     data13, data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25,
      data26, data27, data28, data29, data30]])
    


    arr = arr1.astype(np.float)



     


       
    pred = model.predict(arr)




    return render_template('result.html',data=pred, pagetitle="result")








if __name__ =="__main__":
    app.run(debug=True)
