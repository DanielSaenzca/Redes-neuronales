import mnist_loader
import network
import pickle #Libreria para guardar cualquier objeto serializable de python en disco duro
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) #Convertimos los datos de entrenamiento en una lista
#Un iterador es un objeto que me permite generar los elementos de una lista
test_data = list(test_data)
net=network.Network([784,30,10])
net.SGD( training_data, 30, 10, 5.0, test_data=test_data) #Entrenamos a la red con SGD
#(training_data, epochs, mini-batch_size, learning rate (eta), test_data=test_data)
archivo = open("Datosred1.pkl",'wb') #archivo donde se guarda la red en disco, w=write, b=vamos a escribir algo que no es ascii, (como bits)
pickle.dump(net,archivo)
archivo.close()
exit()
#Va a guardar estas variables:
#        self.num_layers = len(sizes)
#        self.sizes = sizes
#       self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
#        self.weights = [np.random.randn(y, x)
#                        for x, y in zip(sizes[:-1], sizes[1:])]
#leer el archivo
archivo_lectura = open("Datosred1.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()
net.SGD( training_data, 10, 50, 0.5, test_data=test_data)
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#Tenemos que convertir la imagen a escala de grises
#esquema de como usar la red : 
imagen = leer_imagen("disco.jpg")
print(net.feedforward(imagen))