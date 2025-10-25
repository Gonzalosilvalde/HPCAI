# BASELINE (Gonzalo Silvade and Unai Iborra)

## Profiling
Se han definido los siguientes tres parámetros para el método train():
    1. profiler=None,
    2. save_profiler_time_table: bool = False,
    3. save_tensorboard_metrics: bool = False,

El parámetro "profiler" permite definir el profiler deseado para el entrenamiento según se defina la clase torch.profiler.profile(). El parámetro save_profiler_time_table” permite al entrenamiento guardar en formato tabla los resultados del profiling, ordenados según el tiempo tardado en ejecutar CUDA por cada parte de la ejecución del entrenamiento.

El parámetro "save_tensorboard_metric" permite guardar para una futura visualización con tensorboa: 
    1. Los "losses", "step times" y "learning rates" de cada step en formato escalar
    2. El tiempo total, pérdida final, batch size y número de épocas.

Para ejecutar fácilmente el servidor de tensorboard se ha creado el script /scripts/start_profiling_srv.sh (se ha configurado para visualizar los resultados de todos los entrenamientos realizados)


Se ha preparado el código para correr con las siguientes configuraciones:
    1. Sin profiler: Ésta configuración no usa profiler pero sí guarda tanto los resultados del entrenamiento como las métricas para visualizar mediante tensorboard.
    2. Con profiler: Ésta configuración utiliza un profiler que sigue el siguiente "schedule":  schedule(wait=2, warmup=100, active=10, repeat=1). Esto indica al profiler que no utilice los primeros 2 steps, que haga profiling de los siguientes 100 pero no utilice ni guarde sus resultados (inicializar el profiler causa una bajada de rendimiento significativa según la documentación por lo que se recomienda utilizar warmup en el profiler para resultados más fiables). Finalmente se realiza el profiling de los 10 siguientes steps. Esta configuración guarda tanto los resultados del profiling en formato tabla, como en json para visualización mediante perfetto. La decisión de realizar profiling en únicamente 10 steps ha sido tomada debido a que en pruebas de profiling de todo el entrenamiento, el entrenamiento finalizaba por falta de memoria a la hora de guardar los datos, y en los casos de entrenamientos con menos epochs, el entrenamiento si acababa pero los resultados del profiling ocupaban decenas de gigabytes. Pese a que el profiling no se realice en todo el entrenamiento, los resultados aportan información relevante sobre dónde el entrenamiento toma más tiempo y recursos. Además debido a este schedule, los resultados de tiempo total de la tabla de métricas de profiling no coincidirá con los resultados de entrenamiento real (el resultado de la tabla es el tiempo total de entrenamiento con profiling aplicado). 

Se ha explorado la opción de visualizar los resultados del profiling mediante tensorboard pero no se ha podido realizar dado que esta funcionalidad ha sido deprecada según la documentación de pytorch (https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). La documentación indica que se debe utilizar perfetto para la visualización del profiling.


## Resultados de los entrenamientos

El modelo ha sido entrenado en diferentes hardwares con la finalidad de observar las diferencias de tiempo y recursos entre ellos:
    1. Nvidia A100 GPU
    2. Nvida Tesla 4 GPU
    3. Nvidia RTX5070 GPU
    4. Intel Xeon Ice Lake 8352Y CPU (with 64 threads)

Los resultados de dichos entrenamientos han sido los siguientes:
### Nvidia A100 GPU
### Nvida Tesla 4 GPU
### Nvidia RTX5070 GPU
### Intel Xeon Ice Lake 8352Y CPU (with 64 threads)