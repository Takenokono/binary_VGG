from flask import Blueprint, request, jsonify, abort ,render_template

module = Blueprint('comixify.comixify', __name__, url_prefix='/')


from logic.VGG import flickr_api


@module.route('/check', methods=['GET','POST'])
def helth_check():
    return jsonify({}), '200'


@module.route('/work',methods=['POST'])
def work():
    return jsonify({}), '200'

@module.route('/front_page', methods=['GET','POST'])
def return_params():
    return render_template('front_page.html')
