- Análisis de Sentimientos de Reseñas: Las opiniones son el alma del negocio. Los algoritmos deben ser entrenados para entender el sentimiento detrás de las palabras. Pista: Experimenta con diferentes modelos de NLP y elige el que mejor se adapte.
- Recomendación Basada en Contenido: La personalización es clave. Desarrolla un sistema de recomendación que no solo observe el comportamiento del usuario, sino también el contenido intrínseco de las series y películas. Pista: Considera usar un modelo híbrido que combine características del contenido y datos del usuario.
- Predicción de Abandono: En el mundo del streaming, cada usuario cuenta. Utiliza Random Forest para identificar patrones y predecir qué usuarios podrían no ver la película entera. Pista: La selección de características y la ingeniería de las mismas puede ser crucial.
- Mejora de la API de Recomendación con Aprendizaje por Refuerzo: Una vez establecida la base del sistema de recomendación, es hora de optimizar. Utiliza un agente de aprendizaje por refuerzo en el entorno simulado proporcionado. Pista: Los modelos basados en Q-learning o DQN pueden ser un buen punto de partida.


python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt