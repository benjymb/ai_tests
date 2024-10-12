import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import gym  # You can use gym to simulate environments

class RecommendationEnv(gym.Env):
    """Simulación de un entorno de recomendación para el aprendizaje por refuerzo"""
    def __init__(self, user_profiles, item_features, max_steps=10):
        super(RecommendationEnv, self).__init__()
        self.user_profiles = user_profiles  # Matriz de características de usuarios
        self.item_features = item_features  # Matriz de características de ítems (películas/series)
        self.max_steps = max_steps
        self.current_step = 0
        self.current_user = None
        
        # Espacios de acción y observación
        self.action_space = gym.spaces.Discrete(len(self.item_features))  # Número de ítems para recomendar
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self.user_profiles[0]),), dtype=np.float32)
    
    def reset(self):
        """Resetea el entorno para un nuevo episodio (nueva interacción de usuario)"""
        self.current_step = 0
        self.current_user = random.choice(self.user_profiles)  # Selecciona aleatoriamente un perfil de usuario
        return self.current_user
    
    def step(self, action):
        """Ejecuta una acción (recomendación) y calcula la recompensa"""
        self.current_step += 1
        recommended_item = self.item_features[action]
        
        # Calcula la recompensa (ej. producto interno entre perfil de usuario e ítem)
        reward = np.dot(self.current_user, recommended_item)
        
        # Determina si el episodio ha terminado
        done = self.current_step >= self.max_steps
        
        # Retorna la observación, recompensa, si ha terminado el episodio, y info adicional
        return self.current_user, reward, done, {}
    
    def render(self, mode='human'):
        pass

class DQNAgent:
    """Agente DQN que aprende a recomendar ítems a los usuarios"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Factor de descuento
        self.epsilon = 1.0   # Tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Construye la red neuronal que predice los valores Q"""
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Guarda la experiencia en la memoria"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Selecciona una acción (exploración vs explotación)"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """Entrena el modelo usando experiencias pasadas"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


