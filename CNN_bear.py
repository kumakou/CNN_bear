# 必要なライブラリをインポート
import torch
from torchvision import models, transforms
import warnings

from PIL import Image, ImageTk
import json

import time

import random

#ファイルダイアログ用
from tkinter import CENTER, NW, Tk, Label,Button,Canvas

# 警告メッセージを非表示にする
warnings.filterwarnings("ignore")

# Tkinterウィンドウの作成
window = Tk()

# ウィンドウを閉じる関数
def close_window():
    window.quit()

def main(window):
    photo_images = []  # 画像の参照を保持するリスト
    AI_point = 0
    human_point = 0
    label_text = {
        0: "ヒグマ",
        1: "ホッキョクグマ",
        2: "ツキノワグマ",
        3: "マレーグマ",
        4: "ナマケグマ",
        5: "メガネグマ",
        6: "アメリカクロクマ",
        7: "パンダ",
        8: "コアラ",
        9: "熊じゃないです",
    }

    photo ={
        0:"./photo/brown_bear.jpg",
        1:"./photo/polar_bear.jpg",
        2:"./photo/asian_black_brar.jpg",
        3:"./photo/sun_bear.jpg",
        4:"./photo/melursus_ursinus.jpg",
        5:"./photo/spectacled_bear.jpg",
        6:"./photo/amerikan_black_bear.jpeg" ,
        7:"./photo/panda.jpg",
        8:"./photo/koara.jpg",
    }

    ## torchvision.modelsからImageNet学習済みモデルを複数ロード
    model_vgg16 = models.vgg16(pretrained=True)

    ##すべてのモデルを評価モードに
    model_vgg16.eval()

    canvas = Canvas(window)
    canvas.pack()
    print("AIと対決しよう！")
    print("熊の画像が表示されるので，どの熊かを当ててね．")
    print("次へボタンを押すと、解答をすることができるよ．")
    
    for i in range(3):
        ##ImageNetの画像に合わせて入力画像を変形させる
        image_converter = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        random_number = random.randint(1, 8)

        # 画像の読み込みとリサイズ
        original_image = Image.open(photo[random_number])
        # 画像のサイズをウィンドウの大きさに合わせてリサイズ
        window.geometry("{}x{}".format(original_image.width, original_image.height))

        time.sleep(1)
        print("第"+str(i+1)+"問目")
        #windwのタイトルを変更
        window.title("第"+str(i+1)+"問目")
        

         # 前回の画像を消去
        canvas.delete('all')

        # キャンバスのサイズを変更
        canvas.config(width=original_image.width, height=original_image.height)

        # 画像の表示

        # 画像の表示
        photo_image = ImageTk.PhotoImage(original_image)
        
        canvas.create_image(0, 0, anchor=NW, image=photo_image)

        # ウィンドウの表示
        window.update()

        ##入力画像を指定されたフォーマットに変形，バッチデータ化
        target_image = Image.open(photo[random_number])
        converted_image = image_converter(target_image)
        batch = converted_image[None]

        ##入力を促して，入力内容を取得
        print("あなたの予想は？次の中から数字で選んでね")
        print("0:ヒグマ, 1:ホッキョクグマ, 2:ツキノワグマ, 3:マレーグマ, 4:ナマケグマ, 5:メガネグマ, 6:アメリカクロクマ, 7:パンダ, 8:コアラ, 9:熊じゃないです")
        input_str = int(input())
        print("あなたの予想：" + label_text[input_str]+"ですね．")
        #1秒待つ
        time.sleep(1)

        ##ImageNetの推論カテゴリ番号と対応する名前のデータをロード
        with open('./imagenet_class_index.json') as file:
            labels = json.load(file)

            ##各モデルごとに画像データを推論
            result = model_vgg16(batch)
            
            possibility_result = result[0].softmax(dim=0)##結果をソフトマックス関数で確率に変換

            result_index = torch.argmax(possibility_result)##最も高い確率のインデックスを取得
            answer = labels[str(result_index.item())]##カテゴリの名前を取得

            ##結果を表示
            print("AIの予想は：" +answer[1] +"です．")

            time.sleep(1)
            print("正解は")
            time.sleep(1)
            print(".")
            time.sleep(1)
            print("..")
            time.sleep(1)
            print("...")
            print(label_text[random_number] + "でした．") 

            if input_str == random_number:
                print("あなたに１ポイント入ります")
                human_point = human_point +1

            if label_text[random_number]==answer[1]:
                print("AIに１ポイント入ります")
                AI_point = AI_point +1

            print("現在の得点は　あなたが" + str(human_point) + "ポイント　AIが" + str(AI_point) + "ポイントです．")
            
            if i < 2:
                print("------------------------")
                print("\n")

            time.sleep(4)
    # ウィンドウを閉じるボタンの処理
    if human_point > AI_point:
        print("あなたの勝ちです")
    elif human_point < AI_point:
        print("AIの勝ちです")
    else:
        print("引き分けです")

    print("また遊んでね！")
    close_window()

if __name__ == '__main__':
    main(window)