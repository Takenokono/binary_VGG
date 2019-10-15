import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import numpy as np
import time ,sys

def fit():
    #画像の大きさを設定
    img_width, img_height = 150, 150

    train_img_path = './images/train/'
    test_img_path = './images/test/'

    batch_size = 32

    input_tensor = Input(shape=(img_width,img_height,3))
    model = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor,input_shape=None)

    #VGG16の全結合層の部分を再定義
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation='sigmoid'))

    full_model = Model(input=model.input, output=top_model(vgg16.output))

    #15層目までの重みを凍結。(構成については調べてください。))
    for layer in full_model.layer[:15]:
        layer.trainable = False

    #画像の量まし
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,zoom_range=0.2,horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255,zoom_range=0.2,horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_img_path,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        test_img_path,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True)
    
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=800)



def flickr_api(img_name):
    #APIキーの情報
    key = ""
    secret = ""
    wait_time = 1
 
    #保存フォルダの指定
    savedir = "../images"
    
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
        filepath = savedir + '/train/' + img_name
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        filepath = savedir + '/test/' + img_name
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)

        
        
        if i<=int(number_img*0.9):
            filepath = savedir + '/train/' + img_name + '/' + photo['id'] + '.jpg'
            if os.path.exists(filepath): continue #重複確認
            urlretrieve(url_q,filepath) #url_qのURLの先を、filepathに保存
            
            
        else:
            filepath = savedir + '/test/' + img_name + '/' + photo['id'] + '.jpg' 
            if os.path.exists(filepath): continue #重複確認
            urlretrieve(url_q,filepath)
            


if __name__ =="__main__":
    flickr_api('Dog')