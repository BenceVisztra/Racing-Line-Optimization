# Racing Line Optimization and Experimental Data Comparison
This repository contains MATLAB code to calculate optimal lap times and optimize racing lines using a Frenet-Serret frame. It allows users to import RaceChrono telemetry data and benchmark it directly against the simulated optimal line.

The following vehicle parameters are adjustable:
- Lateral acceleration
- Longitudinal acceleration (forward)
- Longitudinal deceleration (braking)
- Lateral jerk
- Longitudinal jerk
- Minimum turning radius

The following picture shows the interactible output figure in matlab
![Figure](https://github.com/BenceVisztra/Racing-Line-Optimization/blob/main/t-race_opt.png)

As an example, the t-race track is included in the code:
https://t-raceglobal.com/
The current datalog from the T-race world record lap (as of 2026.09.26) of 12.09 seconds is included in the repository. Importing your own laps is a bit tedious for now, you need to manually align your lines with the track then trim the data so that the discontinuity is on the actual finish line. Aligning works by first adjusting the variable theta such that your T fits around the points laid out by the track, and if necessary, adding a small offset in the "translation" section, however if your lines are tight extra translation should not be necessary. Trimming works by adjusting start_idx and end_idx variables, so that the log starts being evaluated from the correct starting point, this is required for the time delta functionality to work well.

The optimal line is split into segments through N nodes for the discrete solver to work. The current solver has a limitation of having to solve a Jacobian operation which is essentially solving an O(N^2) matrix, meaning extra samples increase computational time exponentially. N=100 samples works well for relatively fast runtime, N=150 is roughly the limit of what can still be computed. Complex tracks will require more resolution, so for those implementing a more complex discrete solver will be necessary.

Track Example and Data Import
The repository includes the T-Race track geometry [t-raceglobal.com](https://t-raceglobal.com/) and the current T-Race world record datalog (12.09 sec, as of 2026-09-26).

Importing external telemetry requires manual alignment and trimming:
- Rotation: Adjust the theta variable to rotate the GPS projection until the track layout matches the simulated reference points.
- Translation: Apply an offset if the rotated telemetry is spatially shifted from the Cartesian origin. Tight lines typically require minimal to no translation.
- Trimming: Adjust start_idx and end_idx so the telemetry array begins exactly at the finish line. This synchronizes the distance parameter, which is strictly required for the time delta calculations to function correctly.

Solver Limitations
The reference path is discretized into $N$ nodes. The current formulation relies on MATLAB's fmincon function using the SQP algorithm. Because the constraint Jacobian is treated as a dense matrix, the linear system solve required at each iteration scales at $O(N^3)$ computational complexity.
- $N=100$ provides rapid convergence.
- $N=150$ represents the practical limit for this dense formulation.
Applying this codebase to larger, complex circuits where extra resolution would be required will require migrating to an interior-point solver with a sparse, analytically defined Jacobian, or utilizing a specialized optimal control framework.
