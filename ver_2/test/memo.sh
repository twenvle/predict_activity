#!/bin/bash

# ファイルのベース名（拡張子なし）を指定
filename="input"

# 1. Gaussianを実行 (環境に合わせて g16 や g09 に変更してください)
g16 ${filename}.gjf

# 2. chkファイルからfchkファイルを生成
formchk ${filename}.chk ${filename}.fchk

# 3. logファイル内の "Frequencies" の行に "-" (マイナス) があるかチェック
# grep -q は何も出力せずに終了ステータスだけを返すオプションです
if grep "Frequencies -- " ${filename}.log | grep -q "\-[0-9]"; then
    
    echo "警告: 虚振動(Imaginary Frequency)が検出されました。"
    
    # logファイルとfchkファイルの名前を変更
    mv ${filename}.log ${filename}_imag.log
    mv ${filename}.fchk ${filename}_imag.fchk
    
    echo "ファイル名を変更しました: ${filename}_imag.log"

else
    echo "虚振動は検出されませんでした（または正常終了）。"
fi