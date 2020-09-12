from flask import Flask,render_template,jsonify,request,url_for,redirect
from werkzeug.utils import secure_filename      #secure_filename()函数获取filename属性的安全版本。
import sys
import json

sys.path.append(r'F:\Python\rjb_yfzd\rjb_mb')
import count_total,z_json
import os

base_dir=os.path.dirname(__file__)

STATICFILES_DIRS = (os.path.join(base_dir, "static"))

app=Flask(__name__)

@app.route('/')
def index():
    fina=count_total.count()
    # for (key,values) in  fina.items():
    #     print (key,values)
    return render_template('index.html',result=fina)
@app.route('/getdata2',methods=["GET"])
def getdata2():
    if request.method == "GET":
        fina=count_total.count()
        # json_str = json.dumps(customers, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        json1=jsonify(fina)
        print(json1)
    return json1

@app.route('/getdata', methods=["GET"])
def get_data():
    customers =[]
    if request.method == "GET":
        fina=count_total.count()
        for (key,values) in  fina.items():
            customers.append(z_json.total(key,values),)
        json_str = json.dumps(customers, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        # json1=jsonify(customers)
        # print(json_str)
    return json_str, 200, {"Content-Type":"application/json"}

@app.route('/uploader',methods = ['POST', 'GET'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      basepath = os.path.dirname(__file__)  # __file__当前py文件所在路径, 然后os.path.dirname是返回当前文件的文件夹
      dir_name = 'testvideo'
      # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
      if not os.path.isdir(dir_name):
         os.makedirs(dir_name)
      upload_path = os.path.join(basepath,dir_name,secure_filename(f.filename))#路径拼接
      upload_path = os.path.abspath(upload_path) # 将路径转换为绝对路径
      f.save(upload_path)
      return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug='True')