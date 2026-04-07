## 教師データの追加

1. ver_4/dataに新たにexcelファイルを作り、データを追加する
2. data/in/knownに追加したサンプルのDFT計算の結果(logファイル)を入れる(smilesは1で追加したものと同じにするように)
3. save_known_data.ipynbで、ver_4/out/csvfile/knownに記述子を含めたデータをcsvファイルで作成する
4. どの記述子の組み合わせが一番予測値が高いか全通り計算する(時間がかかるので共用パソコン上で行う)
5. relative.ipynbで、記述子同士が強い相関を有していないか確認
6. make_pkl.ipynbで、4で一番予測値が高かった記述子の組み合わせを用いて、ver_4/out/pklfileにpklファイルを作成する
7. data/in/unknownに未知のサンプルのDFT計算の結果(logファイル)を入れる
8. add_unknown_data.ipynbで、ver_4/out/csvfile/unknownに記述子を含めたデータをcsvファイルで作成する
9. predict.ipynbで、未知のデータを予測する
