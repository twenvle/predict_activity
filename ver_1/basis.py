import pandas as pd
import numpy as np


df = pd.read_csv("sample/basis.txt", sep="\\s+", names=["atom", "x", "y", "z"])
# df_ion = pd.read_csv("sample/basis_ion.txt", sep="\\s+", names=["atom", "x", "y", "z"])

coordinates = df[["x", "y", "z"]].to_numpy()
# ion_coordinates = df_ion[["x", "y", "z"]].to_numpy()


def if_ion(x):
    if x == "ab":
        a = [coordinates[6] - coordinates[0], coordinates[7] - coordinates[1]]
        b = [coordinates[1] - coordinates[0], coordinates[0] - coordinates[1]]
    elif x == "ab_ion":
        a = []
        b = []
    return a, b


def if_ion2(x):
    cooh_h2o = []
    if x == "ab":
        co_double1 = coordinates[10] - coordinates[6]
        co_single1 = coordinates[11] - coordinates[6]
        ch1 = coordinates[17] - coordinates[6]
        co_water1 = coordinates[20] - coordinates[6]
        ch1_water1 = coordinates[21] - coordinates[6]
        ch2_water1 = coordinates[22] - coordinates[6]
        co_double2 = coordinates[9] - coordinates[7]
        co_single2 = coordinates[8] - coordinates[7]
        ch2 = coordinates[16] - coordinates[7]
        co_water2 = coordinates[18] - coordinates[7]
        ch1_water2 = coordinates[19] - coordinates[7]
        ch2_water2 = coordinates[20] - coordinates[7]
        cooh_h2o = [
            [co_double1, co_single1, ch1, co_water1, ch1_water1, ch2_water1],
            [co_double2, co_single2, ch2, co_water2, ch1_water2, ch2_water2],
        ]
    elif x == "ab_ion":
        cooh_h2o = []
    return cooh_h2o


def rotation_matrix(axis, theta):
    axis = np.array(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def align_plane(aprime, bprime, ab="ab"):
    if ab == "ab":
        a, b = if_ion("ab")
    elif ab == "ab_ion":
        a, b = if_ion("ab_ion")

    R = []
    for i in range(2):
        a_i, b_i = np.array(a[i], float), np.array(b[i], float)
        aprime_i, bprime_i = np.array(aprime[i], float), np.array(bprime[i], float)

        n = np.cross(a_i, b_i)
        n /= np.linalg.norm(n)
        nprime = np.cross(aprime_i, bprime_i)
        nprime /= np.linalg.norm(nprime)
        if np.dot(n, nprime) < 0:  # 法線方向が反転していたら修正
            nprime = -nprime

        cos_theta1 = np.clip(np.dot(n, nprime), -1, 1)
        theta1 = np.arccos(cos_theta1)
        axis1 = np.cross(n, nprime)
        if np.linalg.norm(axis1) < 1e-8:
            axis1 = np.array([1, 0, 0])
        R1 = rotation_matrix(axis1, theta1)

        a_rotated = R1 @ a_i
        a_rotated /= np.linalg.norm(a_rotated)
        aprime_i /= np.linalg.norm(aprime_i)
        cos_theta2 = np.clip(np.dot(a_rotated, aprime_i), -1, 1)
        theta2 = np.arccos(cos_theta2)
        axis2 = nprime
        if np.dot(np.cross(a_rotated, aprime_i), axis2) < 0:
            theta2 = -theta2
        R2 = rotation_matrix(axis2, theta2)
        R.append(R2 @ R1)
        print(theta1, theta2)
    return R


def change_coordinates(R, coords="ab"):
    if coords == "ab":
        coords = if_ion2("ab")
    elif coords == "ab_ion":
        coords = if_ion2("ab_ion")
    coords_new = []
    for i in range(2):
        coords_box = []
        R_i = R[i]
        for coord in coords[i]:
            coord = np.array(coord, float)
            coord_new = R_i @ coord
            coords_box.append(coord_new)
        coords_new.append(coords_box)
    return coords_new
