from refuerzo.agent import RecommendationEnv, DQNAgent
import numpy as np

# Simulación del entorno de recomendación
if __name__ == "__main__":
    # Simula perfiles de usuarios y características de ítems
    num_users = 100
    num_items = 20
    num_features = 10  # Características por usuario/ítem
    user_profiles = np.random.rand(num_users, num_features)
    item_features = np.random.rand(num_items, num_features)

    # Inicializa el entorno y el agente DQN
    env = RecommendationEnv(user_profiles, item_features)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    # Entrenamiento del agente en el entorno simulado
    for e in range(1000):  # Número de episodios
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{1000}, Score: {time}, Epsilon: {agent.epsilon}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)