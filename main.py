import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from textblob import TextBlob

# Función para construir el nombre del archivo JSON
def get_data_file_name(crypto):
    return f'{crypto}_data.json'

# Función para cargar datos desde un archivo JSON
def load_data(crypto):
    data_file = get_data_file_name(crypto)
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    return None

# Función para guardar datos en un archivo JSON
def save_data(crypto, precio, tendencia, dias, contador_tendencia, fecha_creacion):
    data_file = get_data_file_name(crypto)
    data = {
        "crypto": crypto,
        "precio": str(precio),
        "tendencia": tendencia,
        "dias": dias,
        "contador_tendencia": contador_tendencia,
        "fecha_creacion": fecha_creacion
    }
    with open(data_file, 'w') as f:
        json.dump(data, f)

# Función para obtener datos de precios de la criptomoneda
def get_crypto_data(period='1M', crypto="btc"):
    url = f'https://bitso.com/api/v3/currency_graph_public?currencyCode={crypto}&period={period}&preferredCurrency=mxn&language=es-mx'
    response = requests.get(url)

    if response.status_code != 200:
        print("Error al obtener datos de la API, código de estado:", response.status_code)
        return None, None

    data = response.json()

    if 'payload' in data and 'graph_data' in data['payload']:
        points = data['payload']['graph_data']['points']
        df = pd.DataFrame(points)

        df['number'] = df['number'].astype(float)
        df['date'] = pd.to_datetime(df['date'], unit='ms')

        df.rename(columns={'date': 'timestamp', 'number': 'close'}, inplace=True)
        df.set_index('timestamp', inplace=True)

        # Obtener el last_price, eliminando la coma
        last_price = float(data['payload']['last_price']['price'].replace(',', ''))
        return df, last_price
    else:
        print("La estructura de la respuesta no contiene 'payload' o 'graph_data'")
        return None, None

# Función para obtener noticias sobre la criptomoneda en un rango de fechas
def get_crypto_news(crypto_name, api_key, from_date, to_date):
    url = (f'https://newsapi.org/v2/everything?q={crypto_name}&from={from_date}&to={to_date}'
           f'&sortBy=publishedAt&language=es&apiKey={api_key}')
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print("Error al obtener noticias:", response.status_code)
        return []

# Función para analizar el sentimiento de una noticia
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Devuelve un valor de -1 (negativo) a 1 (positivo)

# Función para calcular el sentimiento promedio de varias noticias
def get_average_sentiment(articles):
    sentiments = []
    for article in articles:
        title = article.get('title') or ''
        description = article.get('description') or ''
        text = title + ". " + description
        sentiment = analyze_sentiment(text)
        sentiments.append(sentiment)
    return sum(sentiments) / len(sentiments) if sentiments else 0

# Ajustar la tendencia según el sentimiento de las noticias
def adjust_trend_with_news(trend, sentiment_score):
    print(f"Sentimiento promedio de noticias (score): {sentiment_score}")

    # Aumentamos los umbrales para asegurarnos de que solo un sentimiento fuerte cambie la tendencia
    if sentiment_score < -0.3 and trend == "sube":
        print("Ajuste: Cambiando de 'sube' a 'estable' por sentimiento negativo")
        return "estable"
    elif sentiment_score > 0.3 and trend == "baja":
        print("Ajuste: Cambiando de 'baja' a 'estable' por sentimiento positivo")
        return "estable"

    # Si el sentimiento no es suficientemente fuerte, dejamos la tendencia tal como está
    return trend

# Función para crear datasets con ventanas de tiempo para múltiples días
def create_multi_step_dataset(df, window_size=20, forecast_days=7):
    data = df['close'].values
    X, y = [], []
    for i in range(len(data) - window_size - forecast_days + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + forecast_days])
    return np.array(X), np.array(y)

# Entrenamiento del modelo con una ventana de tiempo específica
def train_lstm_model(X, y, window_size, forecast_days):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_days))

    model.compile(optimizer='adam', loss='mean_squared_error')
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

    # Aumentar el número de épocas
    history = model.fit(X_reshaped, y, epochs=150, batch_size=32, verbose=1)
    return model, history

# Predicción de la tendencia (sube o baja)
def predict_trend(model, X, forecast_days):
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
    predictions = model.predict(X_reshaped)

    trend = []
    for i in range(forecast_days - 1):
        if predictions[-1][i + 1] > predictions[-1][i]:
            trend.append(1)  # 1 indica que sube
        else:
            trend.append(-1)  # -1 indica que baja

    days_up = sum(1 for x in trend if x == 1)
    days_down = sum(1 for x in trend if x == -1)

    if days_up > days_down:
        return "sube", days_up
    elif days_down > days_up:
        return "baja", days_down
    else:
        return "estable", 0

if __name__ == '__main__':
    # Parámetros
    forecast_days = 7
    window_size = 20
    crypto = "eth"
    api_key = 'API_key'  # Reemplaza con tu clave de API

    # Cargar datos anteriores
    previous_data = load_data(crypto)

    # Obtener los datos de cierre de la criptomoneda para un mes
    df, last_price = get_crypto_data(period='1M', crypto=crypto)

    if df is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['close'] = scaler.fit_transform(df[['close']])

        X, y = create_multi_step_dataset(df, window_size, forecast_days)
        if len(X) == 0 or len(y) == 0:
            print("No hay suficientes datos para crear conjuntos de entrenamiento. Por favor, ajusta 'window_size' y 'forecast_days'.")
            exit()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # Entrenar el modelo y predecir la tendencia
        model, history = train_lstm_model(X_train, y_train, window_size, forecast_days)
        trend, days_predicted = predict_trend(model, X_test, forecast_days)

        # Definir las fechas para obtener noticias del último mes
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        articles = get_crypto_news(crypto, api_key, from_date, to_date)
        sentiment_score = get_average_sentiment(articles)

        # Imprimir valores de diagnóstico
        print(f"Tendencia predicha por el modelo (antes del ajuste): {trend}")
        print(f"Sentimiento promedio de noticias: {sentiment_score} "
              f"({'positivo' if sentiment_score > 0 else 'negativo' if sentiment_score < 0 else 'neutral'})")

        # Ajustar la tendencia según el sentimiento de las noticias
        adjusted_trend = adjust_trend_with_news(trend, sentiment_score)

        print(f"Tendencia final ajustada en {forecast_days} días: {adjusted_trend}")
        print(f"Precio actual: {last_price} MXN")

        # Comparar la tendencia actual con la almacenada en el JSON
        if previous_data:
            previous_trend = previous_data['tendencia']
            contador_tendencia = previous_data.get('contador_tendencia', 0)
            fecha_creacion_str = previous_data.get('fecha_creacion', None)
            if fecha_creacion_str:
                fecha_creacion = datetime.fromisoformat(fecha_creacion_str)
            else:
                fecha_creacion = datetime.now()

            dias_transcurridos = max((datetime.now() - fecha_creacion).days, 1)

            if adjusted_trend == previous_trend:
                # Incrementar el contador por los días transcurridos
                contador_tendencia += dias_transcurridos

                # Verificar si el contador alcanza la mitad de los días pronosticados
                if contador_tendencia >= forecast_days / 2:
                    print(f"Tendencia confirmada en {contador_tendencia} días.")
                    # Aquí puedes agregar código para enviar una alerta
                else:
                    print(f"La tendencia se mantiene por {contador_tendencia} días.")
            else:
                print("La tendencia ha cambiado. Actualizando el archivo JSON.")
                # Reiniciar el contador y actualizar la tendencia
                contador_tendencia = 0
                previous_trend = adjusted_trend

            # Actualizar la fecha de creación al día de hoy
            fecha_creacion = datetime.now()

            # Guardar los datos actualizados
            save_data(crypto, last_price, adjusted_trend, forecast_days, contador_tendencia, fecha_creacion.isoformat())
        else:
            print("No hay datos previos. Guardando la tendencia actual en el archivo JSON.")
            # Guardar los datos por primera vez
            contador_tendencia = 0
            fecha_creacion = datetime.now()
            save_data(crypto, last_price, adjusted_trend, forecast_days, contador_tendencia, fecha_creacion.isoformat())

        print("El programa ha finalizado correctamente.")

    else:
        print("No se pudo obtener datos históricos para el entrenamiento.")
