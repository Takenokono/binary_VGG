from flask import Blueprint, request, jsonify, abort ,render_template

module = Blueprint('comixify.comixify', __name__, url_prefix='/')

import os 
#ディレクトリ操作の為の標準モジュール候補
import shutil #「shutil.rmtree()」がファイルを持つディレクトリの削除を行うメソッド？
from logic.VGG import flickr_api , fit, predict
import werkzeug


@module.route('/check', methods=['GET','POST'])
def helth_check():

    return jsonify({}), '200'


@module.route('/work',methods=['POST'])
def work():
    '''
    #ミッケ画像の受け取り
    tmp_file = request.files['uploadFile']
    filename = tmp_file.filename
    saveFileName = werkzeug.utils.secure_filename(filename)
    path ='./images/last_check'
    tmp_file.save(os.path.join(path, saveFileName))
    #保存
    movie_path = os.path.join(path, saveFileName)
    '''
    #対象の名前
    target_name = request.values['name']

    #### 処理開始
    #flickrから画像をダウンロード
    flickr_api(target_name)

    #ダウンロードした画像を学習
    fit()

    #学習データから推論
    predict()

    #tmp削除
    ## os.remove(movie_path)
    '''
    os.remove('./images/test/target')
    os.remove('./images/train/target')

    '''

    

    return jsonify({}), '200'



@module.route('/front_page', methods=['GET','POST'])
def return_params():
    return render_template('front_page.html')
