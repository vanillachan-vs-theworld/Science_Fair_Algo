from flask import Flask,request,render_template,redirect
import os
app = Flask(__name__)
import cv2
import numpy as np
import glob
ts = 0
found = None
app.config["IMAGE_UPLOADS"] = "static\Image_Upload"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename


@app.route('/home',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']

		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)


		filename = secure_filename(image.filename)

		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))

		return render_template("main.html",filename=filename)



	return render_template('main.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename = "static\Image_Upload" + filename), code=301)

for file_name in glob.glob('static\\Image_Upload\\*'):
    ts = 0
    found = None
    fts = os.path.getmtime(file_name)
    print(fts,ts)
    
    if fts > ts:
        ts = fts
        found = file_name
        print(found)
        img1 = cv2.imread(found)
    b, g, r = cv2.split(img1)

    ttl = img1.size / 3 
    B1 = float(np.sum(b)) / ttl 
    G1 = float(np.sum(g)) / ttl
    R1 = float(np.sum(r)) / ttl
    total = R1+G1+B1
    B = float(np.sum(b)) / ttl / total
    G = float(np.sum(g)) / ttl / total
    R = float(np.sum(r)) / ttl / total
    
    B_mean1 = list()
    G_mean1 = list()
    R_mean1 = list()

    B_mean1.append(B)
    G_mean1.append(G)
    R_mean1.append(R)

    print(B_mean1)
    print(G_mean1)
    print(R_mean1)
#images = ("/Image_Upload" + filename)
#print(images)
app.run(debug=True,port=2000)



#images = glob.glob(r"C:\Users\jun63\Downloads\Cropped Science Fair Folder\Hand Soap\25.0_ Hand Soap/*.jpg")
#img1 = cv2.imread(r"C:\Users\jun63\Downloads\Cropped Science Fair Folder\Distilled Water (Control)/IMG_3819")



#create array and feed into the models; make one site for each one????