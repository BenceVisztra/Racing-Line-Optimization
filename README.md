# Racing Line Optimization and Experimental Data Comparison
This repository contains MATLAB code to calculate optimal lap times and optimize racing lines using a Frenet-Serret frame. It allows users to import RaceChrono telemetry data and benchmark it directly against the simulated optimal line.

## Adjustable Vehicle Parameters:
- Lateral acceleration
- Longitudinal acceleration (forward)
- Longitudinal deceleration (braking)
- Lateral jerk
- Longitudinal jerk
- Minimum turning radius

The following picture shows the interactible output figure in matlab:

![Figure](https://github.com/BenceVisztra/Racing-Line-Optimization/blob/main/t-race_opt.png)

## Track Example and Data Import

The repository includes the T-Race track geometry ([t-raceglobal.com](https://t-raceglobal.com/)) and the current T-Race world record datalog (12.09 sec, as of 2026-09-26).

### Importing external telemetry requires manual alignment and trimming:

- Rotation: Adjust the theta variable to rotate the GPS projection until the track layout matches the simulated reference points.
- Translation: Apply an offset if the rotated telemetry is spatially shifted from the Cartesian origin. Tight lines typically require minimal to no translation.
- Trimming: Adjust start_idx and end_idx so the telemetry array begins exactly at the finish line. This synchronizes the distance parameter, which is strictly required for the time delta calculations to function correctly.

## Solver Limitations

The reference path is discretized into $N$ nodes. The current formulation relies on MATLAB's fmincon function using the SQP algorithm. Because the constraint Jacobian is treated as a dense matrix, the linear system solve required at each iteration scales at $O(N^3)$ computational complexity.
- $N=100$ provides rapid convergence.
- $N=150$ represents the practical limit for this dense formulation.
  
Applying this codebase to larger, complex circuits where extra resolution would be required will require migrating to an interior-point solver with a sparse, analytically defined Jacobian, or utilizing a specialized optimal control framework.


## Custom Tracks

Simple custom track layouts can be implemented by defining the coordinates inside the track_pts matrix. The section between P1 and P2 represents the finish line, the rest of the points define the remaining apexes. For tracks where the corner has a nonzero radii, a keepout circle shall be defined around the center point of the apex.
