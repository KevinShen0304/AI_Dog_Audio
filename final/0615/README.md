資料放置:
.\train\train : 訓練集音頻
.\train\noise_foraug : 背景音頻
(刪除音頻不足5s之音檔與資料)


程式執行順序:

1-1.gene_img
產生頻譜圖

1-2.aug_maker_backgroundnoise
產生增加背景音的頻譜圖(資料增強用)


2-1.Model
模型建立

2-2.pre_train
模型訓練(尚未使用資料增強)

3-1.MyTrainGenerator
建立datagenerator函數

3-2.generator_train
再次模型訓練(使用資料增強)


4.predict_With_private
預測結果(包含private_test&public_test)

