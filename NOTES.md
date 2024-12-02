- Uncertainty depends on DT (sampling time)
  - makes sense, the computation of the discrete Q matrix depends on DT
- Real-time achievable in cartesian and joint space up to DT = 0.05s and pred_horizon = 0.5s
- To profile python code from CLI:
  - run the code with the profiler:
    ```OMP_NUM_THREADS=4 python3 -m cProfile -o ../../results/profiling/profile_output_<space_c>_<space_e>.prof evaluate_error_metrics.py```
  - analyze the results visually using:
    ```snakeviz ../../results/profiling/profile_output_<space_c>_<space_e>.prof```
- Metrics: Cartesian-Cartesian OK, Joint-Joint OK, Joint-Cartesian WRONG
  - Am I missing something to move from joint space to cartesian space?
    - -> Not necessarily, filtering may be wrong in the first place
- Required parameter retuning since dataset has changes
  - How to do it properly?
  - Once DT is fixed, just tune Q isotropically?
  - Tune the values of the M matrix for the IMM?
- SINDy super slow and wrong results -> singular matrix...