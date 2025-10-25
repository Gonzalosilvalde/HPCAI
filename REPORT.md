# BASELINE (Gonzalo Silvade and Unai Iborra)

Three parameters have been set for the train() method:
profiler=None,
save_profiler_time_table: bool = False,
save_tensorboard_metrics: bool = False,

The “profiler” parameter allows the training to save profiling metrics as specified in the torch.profiler.profile() class. The “save_profiler_time_table” allows the training to save a table in text format, where the data from the profiler is shown sorted by time taken in the training.
The save_tensorboard_metric, saves the loss, step times and learning rates metrics in an scalar format, and total time, final loss, batch size and number of epochs for later graph visualization with tensorboard. To easily launch the tensorboard server, /scripts/start_profiling_srv.sh has been created.

The code has been configured to run the training two times with different configurations:
    1. No profiler: This configuration uses no profiler and saves the training metrics for later visualization with tensorboard.
    2. With profiler: This configuration uses a profiler with the following schedule: schedule(wait=2, warmup=100, active=10, repeat=1) which indicates the profiler to not profile the first two steps, profile without saving the next 100 steps and profile and save the following 10 steps. The decission to save 10 steps has been made because in the tests done, either the resulting profiling output had the size of tenths of gigabytes or the program ran out ofn memory. Even if the profiling results don’t take into account all the steps in the model, it helps to visualize where the model takes more resources and time. Because of the schedule, the output table times won´t match the time spent training the module.

Visualization of the profiling metrics data table in tensorboard has been explored but not been implemented due to being deprecated according to the pytorch documentation (https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

Training results:

The model has been tested in different hardwares:
    1. Nvidia A100 GPU
    2. Nvida Tesla 4 GPU
    3. Nvidia RTX5070 GPU
    4. Intel Xeon Ice Lake 8352Y (with 64 threads)

The training results from those different hardwares are the following: