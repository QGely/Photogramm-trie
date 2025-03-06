import cv2
import numpy as np
from scipy.optimize import leastsq
from scipy.spatial import Delaunay
import glob

import tkinter as tk
from tkinter import filedialog
import os

# Ouvrir une boîte de dialogue pour sélectionner le dossier
root = tk.Tk()
root.withdraw()  # Masquer la fenêtre principale

# Demander à l'utilisateur de sélectionner un dossier
dossier = filedialog.askdirectory(title="Sélectionnez le dossier contenant les images de calibration")

# Construire le chemin avec le motif pour les fichiers PNG
chemin_images_calibration = os.path.join(dossier, "*.PNG")
print("Chemin d'accès des images de calibration :", chemin_images_calibration)

# Charger les images de calibration
images = [cv2.imread(fichier) for fichier in glob.glob(chemin_images_calibration)]


class Camera:
    def __init__(self, focal_length, principal_point_offset, rotation, translation):
        self.f = focal_length
        self.c = principal_point_offset
        self.R = rotation
        self.t = translation

    def project(self, pts3):
        pcam = self.R.T @ (pts3 - self.t)
        p = self.f * (pcam / pcam[2:])
        pts2 = p[:2, :] + self.c
        return pts2

    def update_extrinsics(self, params):
        self.t = params[:3]
        self.R = make_rotation(params[3], params[4], params[5])

def make_rotation(rx, ry, rz):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def residuals(pts3, pts2, cam, params):
    cam.update_extrinsics(params)
    projected_pts2 = cam.project(pts3)
    return (projected_pts2 - pts2).ravel()

def calibrate_camera(images, checkerboard_size):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan image

    gray = None

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if gray is None:
        raise ValueError("Aucune image valide n'a été fournie pour la calibration.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs

def decode(image_paths, start, threshold, cthreshold=0.6):
    codes = []
    for i in range(start, start + len(image_paths), 2):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i+1], cv2.IMREAD_GRAYSCALE)
        difference = cv2.absdiff(img1, img2)
        _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        codes.append(mask)
    box_code = np.stack(codes, axis=0)
    return box_code

def triangulate(pts2L, camL, pts2R, camR):
    pts3 = []
    for i in range(pts2L.shape[1]):
        qL = np.array([camL.f * pts2L[0, i], camL.f * pts2L[1, i], camL.f])
        qR = np.array([camR.f * pts2R[0, i], camR.f * pts2R[1, i], camR.f])
        zL, zR = np.linalg.lstsq(np.array([qL, -qR]), camR.t - camL.t, rcond=None)[0]
        pt3D_L = camL.R @ (qL * zL) + camL.t
        pt3D_R = camR.R @ (qR * zR) + camR.t
        pts3.append((pt3D_L + pt3D_R) / 2)
    return np.array(pts3).T

def reconstruct(imprefix, threshold, cthreshold, camL, camR):
    decode_masks = decode([f"{imprefix}{i:02}.png" for i in range(20)], 0, threshold, cthreshold)
    mask_indices = np.intersect1d(np.nonzero(decode_masks[0]), np.nonzero(decode_masks[1]))
    pts2L = np.array([[i % decode_masks.shape[2], i // decode_masks.shape[2]] for i in mask_indices])
    pts2R = np.array([[i % decode_masks.shape[2], i // decode_masks.shape[2]] for i in mask_indices])
    pts3 = triangulate(pts2L.T, camL, pts2R.T, camR)
    return pts3

def make_mesh(pts3, threshold, boxlimits):
    tri = Delaunay(pts3[:2].T)
    triangles = []
    for simplex in tri.simplices:
        if np.all(np.linalg.norm(pts3[:, simplex[0]] - pts3[:, simplex[1:]], axis=1) < threshold):
            triangles.append(simplex)
    triangles = np.array(triangles)
    return triangles

def write_ply(filename, vertices, colors, faces):
    with open(filename, 'w') as ply_file:
        ply_file.write("ply\nformat ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_index\nend_header\n")
        for i, v in enumerate(vertices):
            ply_file.write(f"{v[0]} {v[1]} {v[2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
        for face in faces:
            ply_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

# Charger les images de calibration et calculer les matrices de caméra
cam_mtx, cam_dist, cam_rvecs, cam_tvecs = calibrate_camera(images, (9, 6))

# Indices des caméras gauche et droite
indice_camL = 0
indice_camR = 1

# Extraire les paramètres intrinsèques
focal_length = cam_mtx[0, 0]
principal_point_offset = (cam_mtx[0, 2], cam_mtx[1, 2])

# Extraire les paramètres extrinsèques pour chaque caméra
rotation_vector_L = cam_rvecs[indice_camL]
translation_vector_L = cam_tvecs[indice_camL]
rotation_matrix_L, _ = cv2.Rodrigues(rotation_vector_L)

rotation_vector_R = cam_rvecs[indice_camR]
translation_vector_R = cam_tvecs[indice_camR]
rotation_matrix_R, _ = cv2.Rodrigues(rotation_vector_R)

# Créer les instances de Camera
camL = Camera(focal_length, principal_point_offset, rotation_matrix_L, translation_vector_L)
camR = Camera(focal_length, principal_point_offset, rotation_matrix_R, translation_vector_R)

# Charger les images gauche et droite pour reconstruire la couleur
imageL = cv2.imread("chemin/vers/image_gauche.png")
imageR = cv2.imread("chemin/vers/image_droite.png")

# Exemple de pts2L et pts2R (doivent être calculés en fonction de vos correspondances)
# pts2L et pts2R peuvent être obtenus lors de la reconstruction
pts2L = np.array([[100, 200], [150, 250]]).T  # Exemple de points
pts2R = np.array([[102, 202], [152, 252]]).T  # Exemple de points

# Calcul des couleurs moyennes
colors = []
for (xL, yL), (xR, yR) in zip(pts2L.T, pts2R.T):
    couleurL = imageL[int(yL), int(xL)]
    couleurR = imageR[int(yR), int(xR)]
    couleur_moyenne = (couleurL.astype(np.float32) + couleurR.astype(np.float32)) / 2
    colors.append(couleur_moyenne)

colors = np.array(colors, dtype=np.uint8)

# Processus de reconstruction et génération du maillage
pts3 = reconstruct("image_path_prefix_", 50, 0.6, camL, camR)

# Définir les limites pour le maillage et appeler la fonction pour générer le maillage
triangles = make_mesh(pts3, 2.0, [-1, 1, -1, 1, 0, 2])

# Sauvegarder le maillage au format .ply
write_ply("output_mesh.ply", pts3.T, colors, triangles)

