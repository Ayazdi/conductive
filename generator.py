from tensorflow.keras.models import model_from_json

import pandas as pd
import numpy as np

import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import pickle

# Load scaler models
with open("x_scaler.pkl", 'rb') as file:
    scaler_x = pickle.load(file)

with open("y_scaler.pkl", 'rb') as file:
    scaler_y_2 = pickle.load(file)

# read coordinates
site = pd.read_csv('site_coordinates.csv')


def load_generator_model(model):
    """
    load the saved trained generator model
    """
    json_file = open(f'{model}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{model}.h5")
    return loaded_model


g_model = load_generator_model('generator_model')


def visualize(thickness):
    """
    3D interpolation of the thickness profile.
    """
    site['SITE_Z'] = thickness
    points2D = np.vstack([site['SITE_X'], site['SITE_Y']]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices

    fig = ff.create_trisurf(site['SITE_X'], site['SITE_Y'], site['SITE_Z'],
                            simplices=simplices,
                            title="wafare", aspectratio=dict(x=1, y=1, z=0.5))
    fig.show()


def create_process_vector(flow, space, dept, tool):
    """
    Creat a procces vector of the parameters for the model.
    """
    features = np.array([flow, space, dept])
    if tool == 1:
        encode = np.array([1, 0, 0, 0])
        features = np.concatenate([features, encode])
    elif tool == 2:
        encode = np.array([0, 1, 0, 0])
        features = np.concatenate([features, encode])
    elif tool == 3:
        encode = np.array([0, 0, 1, 0])
        features = np.concatenate([features, encode])
    elif tool == 4:
        encode = np.array([0, 0, 0, 1])
        features = np.concatenate([features, encode])
    features = np.array(features).reshape(1, -1)
    print(features)
    process_vector = scaler_y_2.transform(features)
    return process_vector


def generate_latent_points(latent_dim=32, n_samples=1):
    """
    generate points in latent space as input for the generator
    """
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input


def generate_thickness(latent_points, process_vector, g_model):
    """
    Genrerate thickness profile form the generator model with given parameters
    """
    X_test = g_model.predict([latent_points, process_vector])
    X_test = X_test.reshape(49,)
    X_test = scaler_x.inverse_transform(X_test.reshape(1, -1))
    return X_test[0]


if __name__ == '__main__':
    latent_points = generate_latent_points()
    print("Please enter flow rate value:\n")
    flow = 0.916 #input()
    print("Please enter spacing value:\n")
    space = 0.355 #input()
    print("Please enter deposition time value:\n")
    dept = 68.5 #input()
    print("Please enter tool number from 1, 2, 3 and 4:\n")
    tool = 4 #input()

    process_vector = create_process_vector(flow, space, dept, tool)
    thickness = generate_thickness(latent_points, process_vector, g_model)
    visualize(thickness)
