from flask import Flask
app = Flask(__name__)


def enableDebug():
    app.debug = True


if __name__ == '__main__':
    enableDebug()

    from api.VGG import VGG
    
    modules = [ VGG.module ]

    for module in modules:
        app.register_blueprint(module)

    app.run(threaded=False, host='0.0.0.0', port=80)
