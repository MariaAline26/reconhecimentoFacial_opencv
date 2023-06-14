import cv2
import os
import numpy as np

eingenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

# Percorre a pasta das imagens e retorna a lista dos ID's e as imagens das pessoas dos respectivos ID's
def getImageComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
#    print(caminhos)
    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem) [-1].split('.')[1])
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids), faces
ids, faces = getImageComId()
#print(faces)

print("Treinando")

# Aprendizagem supervisionada
eingenface.train(faces, ids)

# Arquivo que é responsável por classificar se o registro é da pessoa 1, 2, 3 ...
eingenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado")