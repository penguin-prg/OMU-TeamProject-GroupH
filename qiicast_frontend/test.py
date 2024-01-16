import requests

# FastAPIサーバーのエンドポイント
api_endpoint = "http://127.0.0.1:8000/"  # FastAPIサーバーの実際のエンドポイントに変更してください


# Streamlitアプリケーションの本体
def main():
    response = send_request(api_endpoint)
    print(response)


# FastAPIサーバーにGETリクエストを送信する関数
def send_request(endpoint):
    try:
        # GETリクエストを送信
        response = requests.get(endpoint)

        # レスポンスのJSONを取得
        response_json = response.json()
        return response_json
    except Exception as e:
        print(f"Error sending request: {str(e)}")


# Streamlitアプリケーションを実行
if __name__ == "__main__":
    main()
