import cv2


def rgb_to_gray(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    return obs


def resize(obs, shape):
    obs = cv2.resize(obs, shape)
    return obs

def gray_resize(obs, shape):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, shape)
    return obs

