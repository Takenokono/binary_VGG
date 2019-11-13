import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import numpy as np
import time ,sys
import cv2


##flickrからダウンロードした画像をモデルに学習させる
def fit():
    #画像の大きさを設定
    img_width, img_height = 150, 150

    train_img_path = './images/train/'
    test_img_path = './images/test/'
    result_path = './result'

    batch_size = 100

    input_tensor = Input(shape=(img_width,img_height,3))
    model = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor,input_shape=None)

    #VGG16の全結合層の部分を再定義
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation='sigmoid'))

    full_model = Model(input=model.input, output=top_model(model.output))

    #15層目までの重みを凍結。(構成については調べてください。))
    for layer in full_model.layers[:15]:
        layer.trainable = False

    full_model.compile(loss='binary_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])

    #画像の量まし
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,zoom_range=0.2,horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255,zoom_range=0.2,horizontal_flip=True)

    #学習用画像生成器
    train_generator = train_datagen.flow_from_directory(
        train_img_path,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False)
    print(train_generator.class_indices)

    #検証用画像生成器
    validation_generator = validation_datagen.flow_from_directory(
        test_img_path,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False)
    print(validation_generator.class_indices)
    
    
    history = full_model.fit_generator(
        train_generator,
        samples_per_epoch=1000,
        nb_epoch=5,
        validation_data=validation_generator,
        nb_val_samples=50)
    
    full_model.save_weights(os.path.join(result_path,'Fintuning.h5'))



##TODO:ミッケ画像を切り抜いて閾値(0.85)を超えたものを枠取りする機能の追加
##学習した重みデータを読み込んで推論を行う
def predict():
    #画像の大きさを設定
    img_width, img_height = 150, 150

    #tmpからミッケ画像を持ってくる。
    '''test_img_path = './images/last_check' '''
    test_img_path = './tmp/mikke8.jpg' #ミッケ画像の読み込み
    result_path = './result'

    input_tensor = Input(shape=(img_width,img_height,3))
    model = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor,input_shape=None)

    #VGG16の全結合層の部分を再定義
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation='sigmoid'))

    full_model = Model(input=model.input, output=top_model(model.output))

    # 学習済みの重みをロード
    full_model.load_weights(os.path.join(result_path, 'Fintuning.h5'))

    full_model.compile(loss='binary_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])

    ##TODO:ここにミッケ画像を分割して処理させる機能を追加する(?)
    mikke = cv2.imread(test_img_path)
    height , width = mikke.shape[:2]  #ミッケの全体画像の縦幅・横幅を取得
    width = width - 660 #下記の説明文の領域を削除

    for i in [10,9,8,7,6,5,4]:
        now_h = 0
        before_h = 0
        now_w = 0
        before_w = 0
        tmp_h = int(height / i)
        tmp_w = int(width / i)
        for j in range(i):
            now_h = now_h + tmp_h
            for k in range(i):
                now_w = now_w + tmp_w
                target_img = mikke[before_h:now_h, before_w:now_w]
                
                #tmpファイルの作成
                tmp_path = './tmp/tmp.jpg'
                cv2.imwrite(tmp_path,target_img)
                
                #tmpファイルの推論
                img = image.load_img(filename, target_size=(img_width, img_height))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)   ##学習時の正規化に合わせて、推論時も正規化
                x = x / 255.0
                pred = full_model.predict(x)[0]
                if pred[0]>0.85:
                    mikke = cv2.rectangle(mikke,(before_h,before_w),(now_h,now_w),(255,0,0),3)
                before_w = now_w
            before_h = before_h + tmp_h
    
    a_img_path = './tmp/predict.jpg'
    cv2.imwrite(a_img_path,mikke)
    

    '''
    # テスト用画像を取得して変数に入れる
    test_imagelist = os.listdir(test_img_path)
    test_imagelist.sort()

    # それぞれの画像について推論する
    for test_image in test_imagelist:
        if test_image == '.DS_Store':
            continue
        filename = os.path.join(test_img_path, test_image)
        img = image.load_img(filename, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 学習時の正規化に合わせて、推論時も正規化
        x = x / 255.0
        # 推論を行う
        # キャラクターごとに確率を出力
        pred = full_model.predict(x)[0]
        
        return pred
    '''


##flickrから学習画像を学習用・検証用に分けてimageディレクトリに保存
def flickr_api(img_name):
    #APIキーの情報
    key = "9bda2e46c5427ab8142328717894c178"
    secret = "de0dabbbc19eb7bb"
    wait_time = 1
 
    #保存フォルダの指定
    savedir = "./images"
    
    #画像の数
    number_img = 200

    flickr = FlickrAPI(key, secret, format='parsed-json') #Flickrにアクセス
    result = flickr.photos.search( #Flickrから探してくる。
        text = img_name,  
        per_page = number_img, #外れデータを考慮して多め
        media = 'photos',   #mediaは写真
        sort = 'relevance', #relevanceは関連順
        safe_search = 1,  #UIコンテンツは取得しない
        extras = 'url_q, licence' #所得したいデータ(画像アドレス、ライセンス)
    )

    photos = result['photos']

    for i , photo in enumerate(photos['photo']): #enumerateは、添字と値を返す。つまりここでは、iに番号。photoに値がくる。
        url_q = photo['url_q'] #持ってきた写真のurl
        
        #dir作成
        filepath = savedir + '/train/' + 'target'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        filepath = savedir + '/test/' + 'target'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)

        
        #学習用画像
        if i<int(number_img*0.9):
            filepath = savedir + '/train/' + 'target' + '/' + photo['id'] + '.jpg'
            if os.path.exists(filepath): continue #重複確認
            urlretrieve(url_q,filepath) #url_qのURLの先を、filepathに保存
            
        #検証用画像  
        else:
            filepath = savedir + '/test/' + 'target' + '/' + photo['id'] + '.jpg' 
            if os.path.exists(filepath): continue #重複確認
            urlretrieve(url_q,filepath)
            

#検証用コマンド
if __name__ =="__main__":
    predict()