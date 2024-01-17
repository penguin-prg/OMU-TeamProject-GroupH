import csv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle


class PenguinModel:
    def __init__(self, model_path: str) -> None:
        """
        コンストラクタ
        :param model_path: 保存されたリニアモデルのファイルパス
        """
        self.model_lightGBM = pickle.load(open(model_path, "rb"))
        self.features_columns = pd.read_csv(
            "lib/inference/features.csv"
        ).columns.tolist()

    def _process_data(self, data_meta: dict) -> pd.DataFrame:
        """
        データの前処理を行うメソッド
        :param data_meta: 予測に必要なメタデータを含む辞書
        :return: 前処理済みのデータが格納されたDataFrame
        """
        user_data = data_meta.get("user", {})
        tags_dict = data_meta.get("tags", [])
        tags = [t["name"] for t in tags_dict]

        data = {
            "user.items_count": [user_data.get("items_count", 0)],
            "user.followers_count": [user_data.get("followers_count", 0)],
            "body_len": [len(data_meta.get("body", ""))],
            "title_len": [len(data_meta.get("title", ""))],
        }
        print(f"tags: {tags}")
        sentence = ["body", "title"]
        for c in sentence:
            data[f"{c}_rows"] = [len(data_meta.get(c, "").split("\n"))]
        use_tags = []
        for c in self.features_columns:
            if c.replace("has_", "") in tags:
                data[c] = [1]
                print(f"{c} is used")
                use_tags.append(c.replace("has_", ""))
            else:
                data[c] = [0]
        # タイトル、文章中のタグ, 記号等の出現回数
        marks = [".", ",", "!", "?", "(", ")", "[", " "]
        marks += ["]", "{", "}", "'", '"', ":", ";", "-", "/", "&", "#", "@", "%"]
        for c in sentence:
            for t in use_tags + marks:
                data[f"{c}_has_{t}"] = [data_meta.get(c, "").count(t)]
                print(f"{c}_has_{t} is used {data_meta.get(c, '').count(t)} times")
        print(data)
        return pd.DataFrame(data)

    def predict(self, df_meta: pd.DataFrame) -> np.ndarray:
        """
        予測を行うメソッド
        :param df_meta: 前処理済みのデータが格納されたDataFrame
        :return: 予測結果の配列
        """
        preprocess_data = self._process_data(df_meta)
        prediction = self.model_lightGBM.predict(
            preprocess_data, num_iteration=self.model_lightGBM.best_iteration
        )
        return prediction[0]
