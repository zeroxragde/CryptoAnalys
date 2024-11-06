CryptoAnaly
CryptoAnaly es una aplicación que predice la tendencia del precio de una criptomoneda utilizando un modelo LSTM (Long Short-Term Memory) y el análisis de noticias relacionadas. La aplicación considera tanto patrones históricos de precios como el análisis de sentimientos de noticias recientes para ofrecer una predicción ajustada de la posible tendencia en el mercado.

Funcionalidades
Predicción de Tendencia Basada en Patrones Históricos:

Utiliza un modelo LSTM entrenado con datos históricos de precios de una criptomoneda (por ejemplo, Bitcoin, Ethereum) para predecir si el precio subirá o bajará en los próximos días.
Analiza una ventana de datos de precios (por ejemplo, los últimos 10 días) para predecir la tendencia de los próximos 5 días.
Análisis de Sentimientos de Noticias:

Obtiene noticias relevantes sobre la criptomoneda utilizando la API de NewsAPI.
Analiza el sentimiento de los títulos y descripciones de las noticias utilizando TextBlob, que devuelve una puntuación de sentimiento positiva, negativa o neutral.
Ajuste de Tendencia Basado en Sentimientos:

Compara la tendencia predicha por el modelo con el sentimiento de las noticias.
Si el modelo predice una tendencia de "sube" o "baja", pero el sentimiento de las noticias es significativamente contrario, la aplicación ajusta la predicción a "estable".
Si el sentimiento y la predicción están alineados, se respeta la predicción original del modelo.
Registro de Predicciones en un Archivo JSON:

Guarda la predicción de tendencia, el precio actual y la fecha en un archivo JSON específico para cada criptomoneda.
Permite comparar predicciones anteriores con el precio actual y ajustar las predicciones futuras.
Requisitos
Python 3.x

Bibliotecas de Python:

numpy
pandas
tensorflow
scikit-learn
matplotlib
requests
textblob
nltk
Clave de API de NewsAPI:

Necesitas una clave de API de NewsAPI para obtener noticias sobre la criptomoneda.
Instalación
Clona el repositorio:

bash
Copiar código
git clone https://github.com/tu_usuario/CryptoAnaly.git
cd CryptoAnaly
Crea un entorno virtual (opcional pero recomendado):

bash
Copiar código
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
Instala las dependencias:

bash
Copiar código
pip install -r requirements.txt
Si no tienes un archivo requirements.txt, puedes instalar las bibliotecas manualmente:

bash
Copiar código
pip install numpy pandas tensorflow scikit-learn matplotlib requests textblob nltk
Descarga los recursos de nltk necesarios para TextBlob:

python
Copiar código
python -m textblob.download_corpora
Configuración
Clave de API de NewsAPI:

Regístrate en NewsAPI para obtener una clave de API.
En el archivo principal de la aplicación (por ejemplo, crypto_analy.py), reemplaza 'YOUR_NEWS_API_KEY' con tu clave de API.
python
Copiar código
MY_NEWS_API_KEY = 'TU_CLAVE_DE_API'
Seleccionar la Criptomoneda:

Modifica la variable crypto en el script principal para analizar la criptomoneda deseada.
python
Copiar código
crypto = "eth"  # Cambia "eth" por el símbolo de la criptomoneda que quieras analizar
Uso
Ejecuta el script principal:

bash
Copiar código
python crypto_analy.py
La aplicación realizará los siguientes pasos:

Obtiene datos históricos de precios de la criptomoneda seleccionada.
Entrena un modelo LSTM utilizando los datos históricos.
Obtiene noticias recientes sobre la criptomoneda.
Analiza el sentimiento de las noticias obtenidas.
Predice la tendencia y ajusta la predicción según el análisis de sentimiento.
Guarda los resultados en un archivo JSON específico de la criptomoneda.
Interpretar los Resultados
Tendencia predicha: Indica si se espera que el precio suba, baje o permanezca estable en los próximos días.
Precio actual: Muestra el precio actual de la criptomoneda en MXN.
Sentimiento promedio de noticias: Indica si las noticias recientes tienen un sentimiento positivo, negativo o neutral.
Contribuciones
Las contribuciones son bienvenidas. Si deseas mejorar la aplicación, por favor, realiza un fork del repositorio y envía un pull request con tus cambios.

Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para obtener más detalles.

Notas
Uso de Datos y APIs:

Asegúrate de cumplir con los términos de uso de las APIs utilizadas, como Bitso y NewsAPI.
Las claves de API deben mantenerse seguras y no compartirse públicamente.
Advertencia:

Esta aplicación es una herramienta de análisis y predicción que combina modelos de aprendizaje automático con análisis de sentimientos.
Las predicciones no son garantizadas y no deben ser consideradas como asesoramiento financiero.
Siempre realiza tu propia investigación antes de tomar decisiones de inversión.

GitHub: ZeroXRagde

