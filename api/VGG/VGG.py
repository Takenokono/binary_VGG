from flask import Blueprint, request, jsonify, abort ,render_template

module = Blueprint('comixify.comixify', __name__, url_prefix='/')

import os 
from logic.VGG import flickr_api
import werkzeug


@module.route('/check', methods=['GET','POST'])
def helth_check():

    return jsonify({}), '200'


@module.route('/work',methods=['POST'])
def work():
    tmp_file = request.files['uploadFile']
    filename = tmp_file.filename
    saveFileName = werkzeug.utils.secure_filename(filename)
    path ='./tmp'
    tmp_file.save(os.path.join(path, saveFileName))
    #保存
    movie_path = os.path.join(path, saveFileName)

    #tmp削除
    os.remove(movie_path)
    
    return jsonify({}), '200'

@module.route('/front_page', methods=['GET','POST'])
def return_params():
    return render_template('front_page.html')
