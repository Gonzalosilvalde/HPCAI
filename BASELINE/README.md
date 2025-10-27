# Links del lab
https://awesome-archduke-bec.notion.site/Lab-AI-HPC-Tools-e647da3f04dc4e66a40692da0d5f9c27

# GUIA BASICA PARA UNA IA en python

## La Red

Primero antes que nada como es obvio, los imports necesarios en la cabecera del programa

Por ejemplo sin saber aún que vamos a necesitar:

```python
import torch
import torch.nn as nn
...

```

Después, tenemos que crear el esqueleto del modelo. Como no se aún la pinta que va a tener el modelo que vamos a usar que nos dio el profesor, un ejemplo básico:

```python
class Red(nn.Module):
    def __init__(self, ...):
        # Iniciamos las partes de la red, seguramente todo esto ya esté en el ejemplo de hugging face, pero, al momento de escribir esto ni lo mire, asi que lo pongo por sea caso. Esto seria un ejemplo tonto de una UNeT.
        self.conv1 = ConvBlock(...)
        self.conv2 = ConvBlock(...)
        self.conv3 = ConvBlock(...)
        self.conv4 = ConvBlock(...)
        self.conv5 = ConvBlock(...)

        self.upconv1 = UpConv(...)
        self.conv6 = ConvBloc(...)
        self.upconv2 = UpConv(...)
        self.conv7 = ConvBloc(...)
        self.upconv3 = UpConv(...)
        self.conv8 = ConvBloc(...)
        self.upconv4 = UpConv(...)
        self.conv9 = ConvBloc(...)

        self.outconv = nn.Conv2d(...)
    #Después de iniciar el esqueleto de la red neuronal, toca hacer el forward pass.
    def forward(self, ...):
        x1 = self.conv1(...)
        x = F.max_pool2d(...)

        x2 = self.conv2(...)
        x = F.max_pool2d(...)
        ... # Resto de los pasos del forward pass, depende de la red, además, puede que ni tengamos que hacerlo.
    # Muchas veces, tenemos que iniciar los pesos de la red de alguna manera, suele depender de la red, así que simplemente pondŕe la función :)
    def weight_init(self, ...):
        ...
    # Y por último, a mi me gusta poner una función para inicializarla
    def initialize(self):
        self.apply(self.weight_init(..))
    # Conste que, además se pueden necesitar más funciones o hacer más cosas al inicializar... esperemos que, o no sea necesario, o que ya nos lo dea masticado el código de huggingfaces.
```

## Entrenamiento

Esto, se puede hacer en el mismo documento *.py* sin ningún problema, a mi por orden me gusta hacerlo en otro, ya se verá como se hace.

```python
# Los imports y estas cosas

# De lo primero (de manera global, dentro de una clase, lo que se vea en verdad), se inicializan los optimizadores, la red, lo que sea necesario

red = RED() # La de antes
funcion_loss = nn.MSELoss(...) # MSELoss porque si, puede ser otra dependiendo del problema, BCE, cross entropy... depende mucho del problema
optimizador = torch.optim.Adam(...) # Lo mismo que con el loss, aquí pueden ser muchas funciones, AdamL, SGD, RMSProp... No depende tanto del proyecto pero depende del proyecto.
... # Aquí se pueden hacer otras cosas, como comprobar si ya hay algo entrenado de antes, mirar si ya hay algun optimizador... muchas cosas, de base diria que no hace falta nada mas.

```



Aquí, se pueden hacer muchas cosas... depende mucho del objetivo del código y estas cosas, pero lo más básico es tener una función train:

```python
def train(...):
    loss = 0.0

    for i in range ...: # Como siempre, depende mucho de las iteraciones que necesitemos por época, puede ser un número estático, el número de elementos en el conjunto de entrenamiento, etc
        data = ... # Cargamos un elemento del dataset de entrenamiento
        optimizer.zero_grad()
        prediccion = red(data)
        loss = funcion_loss(predicción, gt) # gt es groundtruth, es el resultado óptimo que tendría que tener la red neuronal. Tiene que estar en el dataset.
        loss.backward()
        optimizer.step()
```
Despues, en un caso real se hace una función de test, es igual a la de train, pero sin los pasos del optimizador y del loss, y se utilizaría el conjunto de test en vez del de entrenamiento. Los pasos de optimizador y loss no se hacen porque no nos interesa que en test la red entrene.

Por último, tenemos el main:

```python
def main(): # O como hagamos el main, python y sus cosas, de normal se hace lo de if __main__, pero no me acuerda como era
    ... # se hace lo que sea antes de iniciar variables o cualquier clase o lo que sea
    for i in range numero_epocas:
        train(...)
        test(...) # Esto si hacemos test, lo dicho antes
```

Esto es de forma general como se entrena una red neuronal, pero faltan algunas cosillas, en el ejemplo completo que te enviare puedes revisar de forma mas completa. :)


## Uso para nuestro caso

Primero crear el entorno virtual e instalar los paquetes:

```sh
python -m venv hpcai
source hpcai/bin/activate
pip install --upgrade pip # Creo que no es necesario esto del todo, pero siempre lo hago porseaca
pip install -U transformers datasets accelerate evaluate torch torchvision
pip install huggingface_hub
```
