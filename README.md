# X-Ray Classifier: COVID19 

## Enunciado
El objetivo de esta tarea es disenar un reconocedor automatico que a partir de una radiografia del torax de una persona sea capaz de diagnosticar si la persona tiene COVID19, Neuomonia o se encuentra sana. Para realizar este sistema de reconocimiento se cuenta con las imagenes radiograficas provenientes de la Italian Society of Medical and Interventional Radiology (SIRM) COVID-19 DATABASE. 


## Descripcion
Se cuenta con una base de datos de 6.300 patches en blanco y negro (escala de grises) de 64x64 pixeles, correspondientes a porciones de radiografias del torax pertenecientes a tres clases: 0-Normal, 1-Neumonia y 2-COVID19. La base de datos que emplearemos se encuentra balanceada, es decir cada una de las clases cuenta con un tercio de las muestras. De la base de datos original, se ha extraido 210 radiografias por clase, y a cada radiografia se le extrajo 10 patches de 64x64 pixeles de las zonas mas oscuras del lado izquierdo de la radiografia.

[OPCIONAL] si te interesa el tema y quieres hacer una contribucion en el diagnostico del COVID19 usando esta metodologia se recomienda ver los [detalles de la base de datos](https://github.com/domingomery/patrones/blob/master/tareas/Tarea_03/data/detalles).

Algunos ejemplos de radiografias y sus correspondientes 10 patches extraidos para cada una de las clases se muestran a continuacion:

<img src="https://github.com/domingomery/patrones/blob/master/tareas/Tarea_03/data/example.jpg" width="600">


La base de datos utilizadas se encuentra disponible en los siguientes links:

* Training (1.680 patches por clase) [descargar](https://github.com/domingomery/patrones/blob/master/tareas/Tarea_03/data/train.zip)
* Testing (420 patches por clase) [descargar](https://github.com/domingomery/patrones/blob/master/tareas/Tarea_03/data/test.zip)

Los patches han sido guardados en archivos PNG con el siguiente nombre: Xmm_nnnn_ppp.png donde:

- 'X' es el caracter inicial de cada nombre de archivo
- mm: es 00, 01 o 02 segun la clase 0, 1, o 2 respectivamente
- nnnn: es el numero de la radiografia de la clase (0001, 0002,... 0168: para training, 0169, 0170, ... 0210 para testing)
- ppp: es el numero del patch extraido de la radiografia (001, 002, ... 010)  


El clasificador debera ser entrenado con las muestras de training y debera ser probado con las muestras de testing. 
Se debe disenar dos clasificadores:

* I) este clasificador considera la clasificacion de cada patch de forma individual. Es decir, la matriz de confusion del testing tendra 420 muestras por clase.

* II) este clasificador es el que le interesa a los laboratorios, ya que considera los grupos de 10 patches por radiografias como una unidad y serviria para determinar si un paciente tiene COVID19, Neumonia o esta sano. Es decir, en este clasificador la matriz de confusion de testing tendra 42 muestras por clase. Una forma simple de llevar a cabo el clasificador II es haciendo mayoria de votos en los grupos de 10 patches segun lo que determine el clasificador I.

ES NECESARIO:

- Se deben probar y analizar al menos 5 estrategias distintas.
- Informe con la extraccion de caracteristicas, la seleccion/transformacion y clasificacion. Se debera reportar tanto en el training como en el testing el accuracy y la matriz de confusion.
