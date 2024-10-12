import numpy as np
import pandas as pd

from contenido.agent import SimulationEnvironment
from helpers.file_helper import get_filepath

movies_df = pd.read_csv(get_filepath('metadatos_peliculas.csv'))
user_interactions_df = pd.read_csv(get_filepath('interacciones.csv'))

env = SimulationEnvironment(movies_df, user_interactions_df)
env.reset()
env.current_user['Age_Group']
env.step(0)
