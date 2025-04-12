from pyngrok import ngrok

# Проброс порта 8000 (тот же, на котором работает uvicorn)
public_url = ngrok.connect(8000)
print("🚀 FastAPI доступно по адресу:", public_url)