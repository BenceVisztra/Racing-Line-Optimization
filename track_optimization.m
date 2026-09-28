% Racing Line and Lap Time Optimization using CasADi
clear; clc; close all;
% Run this command first
% addpath('<yourpath>/casadi-3.8.1-windows64-matlab2018b')
% addpath('D:\Downloads/casadi-3.8.1-windows64-matlab2018b')

%% 1. Parameters and Constraints
g = 9.81; % m/s^2

% Vehicle Limits
v_max = 67.0 / 3.6; % Top speed limit
mass = 110.0; % kg
P_motor = 12000.0; % W (12 kW)
efficiency = 0.85; % Drivetrain efficiency multiplier
A_lat = 0.9 * g;
A_long_fwd = 0.55 * g;
A_long_brake = 0.55 * g;
J_long = 1.0 * g;
R_min = 3;

% Dynamic Lateral Jerk Parameters
lat_jerk_max = 4.0 * g;
lat_jerk_min = 1.5 * g;
speed_lat_jerk_min = 20.0 / 3.6; % km/h (Highest jerk allowed below this speed)
speed_lat_jerk_max = 60.0 / 3.6; % km/h (Lowest jerk allowed above this speed)

% Solver Nodes (recommended at least 1-5 node/m)
nodes = 1500;
% Optimizer filters
W_smooth = 0.001 * nodes; 
W_vel_smooth = 0.00005 * nodes; 


% Import Data
rc_data = readtable(['gps_kartplanet_51_87.csv']); %gps_kartplanet_david_47_56 %gps_t_race_ts_11_96 %gps_kistarcsa_mojo_28_33
% Telemetry Rotation
theta = 0; % radians %3.665 for 12.09; 3.697 for 11.96


% Track Definition
track_name = 'track_kartplanet.csv';
track_pts = track_selection(track_name);
% Options:
% T race CW
% T race CCW
% track_kartplanet.csv
% track_kistarcsa.csv

P_max = P_motor * efficiency; % Effective power applied to track

%% 2. Generate Reference Centerline with Periodic Boundaries
% Use the full continuous track directly
pts_base = track_pts; 

% Sanitize data: replace any NaNs caused by CSV formatting
pts_base(:,3) = fillmissing(pts_base(:,3), 'nearest');
pts_base(:,4) = fillmissing(pts_base(:,4), 'nearest');
% Fallback if the entire 4th column was empty/unparseable
if all(isnan(pts_base(:,4))), pts_base(:,4) = pts_base(:,3); end

num_base_pts = size(pts_base, 1);

% Replicate base track coordinates 3 times to ensure complete periodic symmetry
pts_ext = repmat(pts_base(:, 1:2), 3, 1);
pts_ext = [pts_ext; pts_base(1, 1:2)]; % Close trailing loop

% Calculate cumulative distance of the extended track
dx_ext = diff(pts_ext(:,1));
dy_ext = diff(pts_ext(:,2));
d_chord_ext = sqrt(dx_ext.^2 + dy_ext.^2);
s_ext = [0; cumsum(d_chord_ext)];

% Center lap starts at the beginning of Lap 2 and ends at the beginning of Lap 3
s_start = s_ext(num_base_pts + 1);
s_end   = s_ext(2 * num_base_pts + 1);

% Interpolate densely to ensure smooth derivatives
N_ext = num_base_pts * 3; 
s_interp_ext = linspace(s_ext(1), s_ext(end), N_ext);
ref_x_raw = makima(s_ext, pts_ext(:,1), s_interp_ext)';
ref_y_raw = makima(s_ext, pts_ext(:,2), s_interp_ext)';

% Smooth the centerline coordinates
ref_x_smooth = smoothdata(ref_x_raw, 'gaussian', 15);
ref_y_smooth = smoothdata(ref_y_raw, 'gaussian', 15);

% Calculate continuous heading (psi)
dx_ext_dense = gradient(ref_x_smooth);
dy_ext_dense = gradient(ref_y_smooth);
psi_ext_dense = unwrap(atan2(dy_ext_dense, dx_ext_dense));

% Extract exactly the middle lap using the requested number of solver nodes
N = nodes;
s_lap = linspace(s_start, s_end, N);
ref_path.x = interp1(s_interp_ext, ref_x_smooth, s_lap)';
ref_path.y = interp1(s_interp_ext, ref_y_smooth, s_lap)';
ref_path.psi = interp1(s_interp_ext, psi_ext_dense, s_lap)';

% Interpolate and smooth the Left/Right widths from columns 3 and 4
w_left_ext = repmat(pts_base(:,3), 3, 1);
w_left_ext = [w_left_ext; pts_base(1, 3)];
W_L_raw = makima(s_ext, w_left_ext, s_interp_ext)';
ref_path.W_left = interp1(s_interp_ext, smoothdata(W_L_raw, 'gaussian', 15), s_lap)';

w_right_ext = repmat(pts_base(:,4), 3, 1);
w_right_ext = [w_right_ext; pts_base(1, 4)];
W_R_raw = makima(s_ext, w_right_ext, s_interp_ext)';
ref_path.W_right = interp1(s_interp_ext, smoothdata(W_R_raw, 'gaussian', 15), s_lap)';

% Ensure exact machine-precision closure
ref_path.x(end)   = ref_path.x(1);
ref_path.y(end)   = ref_path.y(1);
ref_path.psi(end) = ref_path.psi(1);
ref_path.W_left(end) = ref_path.W_left(1);
ref_path.W_right(end) = ref_path.W_right(1);

%% 3. State Vector Initialization (CasADi)
N=nodes;

import casadi.*
opti = casadi.Opti();

% Declare optimization variables
n  = opti.variable(N, 1);       % Lateral deviation (m)
v  = opti.variable(N, 1);       % Velocity (m/s)
dt = opti.variable(N-1, 1);     % Time step (s)

%% 4. Bounds and Track Limits
% Subtract a 0.5m safety margin from both sides to keep the kart on the drivable racing surface
lb_n = -(ref_path.W_right - 0.5); 
ub_n =  (ref_path.W_left - 0.5);

opti.subject_to(lb_n <= n <= ub_n);
opti.subject_to(1.0 <= v <= v_max); % Applied top speed limit
opti.subject_to(dt >= 0.01);

opti.set_initial(n, zeros(N, 1));
opti.set_initial(v, 10 * ones(N, 1));
opti.set_initial(dt, 0.2 * ones(N-1, 1));

%% 5. Objective & Nonlinear Constraints (CasADi)

% Target the 2nd derivative (wiggliness/steering) instead of distance from center
costFunc = sum(dt) ...
    + W_smooth * sumsqr(diff(diff([n; n(1); n(2)]))) ...
    + W_vel_smooth * sumsqr(diff([v; v(1)]));

opti.minimize(costFunc);

% 1. Cartesian Coordinates
x = ref_path.x - n .* sin(ref_path.psi);
y = ref_path.y + n .* cos(ref_path.psi);

dx = diff(x); % Length N-1
dy = diff(y);
ds = sqrt(dx.^2 + dy.^2); 

% 2. Kinematics Equality (Distance vs Velocity)
v_avg = 0.5 * (v(1:N-1) + v(2:N));
opti.subject_to(ds == v_avg .* dt);

% 3. Flying Start Periodic Boundary Conditions
opti.subject_to(n(1) == n(N));
opti.subject_to(v(1) == v(N));

dn_start = (n(2) - n(1)) / dt(1);
dn_end   = (n(N) - n(N-1)) / dt(end);
opti.subject_to(dn_start == dn_end);

dv_start = (v(2) - v(1)) / dt(1);
dv_end   = (v(N) - v(N-1)) / dt(end);
opti.subject_to(dv_start == dv_end);

% 4. Periodic Curvature Calculation (Covers Node 1 / Node N)
dx_wrap = [dx(end); dx];
dy_wrap = [dy(end); dy];
ds_wrap = [ds(end); ds];

dx1 = dx_wrap(1:end-1); dy1 = dy_wrap(1:end-1);
dx2 = dx_wrap(2:end);   dy2 = dy_wrap(2:end);

cross_p = dx1 .* dy2 - dy1 .* dx2;
dot_p   = dx1 .* dx2 + dy1 .* dy2;
dHeading = atan2(cross_p, dot_p);

ds_nodes = 0.5 * (ds_wrap(1:end-1) + ds_wrap(2:end));
kappa = dHeading ./ ds_nodes; % Length N-1 (all unique nodes)

opti.subject_to(-1/R_min <= kappa <= 1/R_min);

% 5. Periodic Acceleration Constraints
a_x_seg = diff(v) ./ dt; % Length N-1
a_x_nodes = 0.5 * ([a_x_seg(end); a_x_seg(1:end-1)] + a_x_seg); % Length N-1
a_y_nodes = (v(1:N-1).^2) .* kappa; % Length N-1

% Traction limit at low speeds, power limit at high speeds
A_long_dyn = fmin(A_long_fwd, P_max ./ (mass .* v_avg));

opti.subject_to(a_x_nodes.^2 + a_y_nodes.^2 <= A_lat^2);
opti.subject_to(-A_long_brake <= a_x_nodes <= A_long_dyn);

% 6. Periodic Jerk Constraints
dt_nodes = 0.5 * ([dt(end); dt(1:end-1)] + dt);
j_x = diff([a_x_nodes; a_x_nodes(1)]) ./ dt_nodes;
j_y = diff([a_y_nodes; a_y_nodes(1)]) ./ dt_nodes;

opti.subject_to(-J_long <= j_x <= J_long);

% Interpolate velocity to the nodes to match j_y dimension (N-1)
v_nodes = 0.5 * ([v(end); v(1:end-1)] + v);
v_clamp = fmax(speed_lat_jerk_min, fmin(speed_lat_jerk_max, v_nodes(1:N-1)));

% Dynamic Lateral Jerk based on velocity
jerk_slope = (lat_jerk_min - lat_jerk_max) / (speed_lat_jerk_max - speed_lat_jerk_min);
J_lat_dynamic = lat_jerk_max + jerk_slope * (v_clamp - speed_lat_jerk_min);

opti.subject_to(-J_lat_dynamic <= j_y <= J_lat_dynamic);

% 7. Solver Options & Execution
p_opts = struct('expand', true);
s_opts = struct('max_iter', 2000, 'tol', 1e-6);
opti.solver('ipopt', p_opts, s_opts);

disp('Executing CasADi/IPOPT solver...');
sol = opti.solve();

true_lap_time = sum(sol.value(dt));
fprintf('True Lap Time: %.3f s\n', true_lap_time);




%% 6. Racebox data
% 1. Import Data
rc_data = rmmissing(rc_data);
lat = rc_data.latitude;
lon = rc_data.longitude;
v_rc_raw = rc_data.speed;

% Extract accelerations and convert from G to m/s^2 for the struct
ax_rc_raw = rc_data.longitudinal_acc * 9.81;
ay_rc_raw = rc_data.lateral_acc * 9.81;

% 2. Equirectangular Projection to Local Cartesian (Meters)
R_earth = 6371000;
lat0 = mean(lat);  
lon0 = mean(lon);  

x_proj = R_earth * cos(deg2rad(lat0)) .* (deg2rad(lon) - deg2rad(lon0));
y_proj = R_earth * (deg2rad(lat) - deg2rad(lat0));

% 3. Global Rotation & Auto-Translation
R_mat = [cos(theta), -sin(theta); sin(theta), cos(theta)];
coords_rot = R_mat * [x_proj, y_proj]';

% Initial guess using centroids
init_dx = mean(ref_path.x) - mean(coords_rot(1,:)');
init_dy = mean(ref_path.y) - mean(coords_rot(2,:)');

% Optimize translation to snap GPS onto the track centerline
ds_idx = 1:5:length(coords_rot(1,:)');
x_sub = coords_rot(1, ds_idx)';
y_sub = coords_rot(2, ds_idx)';

align_cost = @(T) sum(min((x_sub + T(1) - ref_path.x').^2 + (y_sub + T(2) - ref_path.y').^2, [], 2));
opts = optimset('Display', 'none');
T_opt = fminsearch(align_cost, [init_dx, init_dy], opts);
fprintf('Auto-translation applied: X shifted by %+.2fm, Y shifted by %+.2fm\n', T_opt(1), T_opt(2));

x_trans = coords_rot(1,:)' + T_opt(1);
y_trans = coords_rot(2,:)' + T_opt(2);

% 4. Automatic S/F Line Detection & Trimming
dist_to_sf = sqrt((x_trans - ref_path.x(1)).^2 + (y_trans - ref_path.y(1)).^2);

% Find local minima (crossings), enforcing at least 150 points between them to ignore jitter
[~, cross_idx] = findpeaks(-dist_to_sf, 'MinPeakDistance', 150);

% Ensure edge-cases (file starting exactly on the S/F line) are caught
if dist_to_sf(1) < 5 && (isempty(cross_idx) || cross_idx(1) > 50)
    cross_idx = [1; cross_idx];
end
if dist_to_sf(end) < 5 && (isempty(cross_idx) || cross_idx(end) < length(dist_to_sf)-50)
    cross_idx = [cross_idx; length(dist_to_sf)];
end

% Strict 5-meter radius filter to ignore adjacent straights
cross_idx = cross_idx(dist_to_sf(cross_idx) < 5); 

if length(cross_idx) >= 2
    % Case A: Multi-lap session, trim to the first full lap found
    rc_start = cross_idx(1);
    rc_end = cross_idx(2);
    
    x_rc = x_trans(rc_start:rc_end);
    y_rc = y_trans(rc_start:rc_end);
    v_rc = v_rc_raw(rc_start:rc_end);
    ax_rc = ax_rc_raw(rc_start:rc_end);
    ay_rc = ay_rc_raw(rc_start:rc_end);
    fprintf('Auto-trimmed GPS log to S/F line (Indices %d to %d).\n', rc_start, rc_end);
    
elseif length(cross_idx) == 1
    % Case B: Single-lap file from RaceChrono that starts on the wrong side of the track.
    shift = cross_idx(1);
    
    x_rc = [x_trans(shift:end); x_trans(1:shift-1)];
    y_rc = [y_trans(shift:end); y_trans(1:shift-1)];
    v_rc = [v_rc_raw(shift:end); v_rc_raw(1:shift-1)];
    ax_rc = [ax_rc_raw(shift:end); ax_rc_raw(1:shift-1)];
    ay_rc = [ay_rc_raw(shift:end); ay_rc_raw(1:shift-1)];
    fprintf('Single-lap file detected. Cyclically shifted data to align with S/F line.\n');
    
else
    % Case C: Fallback. Use as-is.
    x_rc = x_trans;
    y_rc = y_trans;
    v_rc = v_rc_raw;
    ax_rc = ax_rc_raw;
    ay_rc = ay_rc_raw;
    fprintf('S/F line not crossed within 5 meters. Check if GPS rotation (theta) is accurate.\n');
end

% 5. Calculate Telemetry Kinematics for Hover Display
dx_rc = diff(x_rc);
dy_rc = diff(y_rc);
ds_rc = sqrt(dx_rc.^2 + dy_rc.^2);

v_avg_rc = 0.5 * (v_rc(1:end-1) + v_rc(2:end));
v_avg_rc(v_avg_rc < 0.1) = 0.1; 
dt_rc = ds_rc ./ v_avg_rc;

t_rc = [0; cumsum(dt_rc)]; % Cumulative elapsed time

% Jerk (differentiating the raw CSV accelerations)
jx_rc = diff(ax_rc) ./ dt_rc;
jy_rc = diff(ay_rc) ./ dt_rc;

% Pad arrays with NaNs to exactly match the length of the coordinate arrays
rc_hover.t  = t_rc; 
rc_hover.v  = v_rc;
rc_hover.ax = ax_rc; 
rc_hover.ay = ay_rc;
rc_hover.jx = [NaN; jx_rc];
rc_hover.jy = [NaN; jy_rc];
rc_hover.s  = [0; cumsum(ds_rc)];

%% 7. Extract, Calculate Kinematics, Interpolate
n_opt  = sol.value(n);
v_opt  = sol.value(v);
dt_opt = sol.value(dt);
x_opt = ref_path.x - n_opt .* sin(ref_path.psi);
y_opt = ref_path.y + n_opt .* cos(ref_path.psi);

% Append first node to close the loop for plotting (N+1 points)
x_plot = [x_opt; x_opt(1)];
y_plot = [y_opt; y_opt(1)];

% Recalculate spatial derivatives to find kappa
dx_opt = diff(x_opt); 
dy_opt = diff(y_opt);
ds_opt = sqrt(dx_opt.^2 + dy_opt.^2); 
heading_opt = atan2(dy_opt, dx_opt); 
dHeading_opt = diff(heading_opt); 
dHeading_opt(dHeading_opt > pi) = dHeading_opt(dHeading_opt > pi) - 2*pi;
dHeading_opt(dHeading_opt < -pi) = dHeading_opt(dHeading_opt < -pi) + 2*pi;
kappa = dHeading_opt ./ ds_opt(1:N-2);

% Recalculate kinematics exactly as constrained by the solver
a_x_seg = diff(v_opt) ./ dt_opt; 
a_x_nodes = 0.5 * (a_x_seg(1:end-1) + a_x_seg(2:end)); 
a_y_nodes = (v_opt(2:N-1).^2) .* kappa; 
dt_nodes = 0.5 * (dt_opt(1:end-1) + dt_opt(2:end)); 
j_x = diff(a_x_nodes) ./ dt_nodes(1:end-1); 
j_y = diff(a_y_nodes) ./ dt_nodes(1:end-1); 

t_opt = [0; cumsum(dt_opt)]; % Calculate cumulative elapsed time

% Calculate cumulative distance of the sparse optimal line
s_opt_raw = [0; cumsum(sqrt(diff(x_plot).^2 + diff(y_plot).^2))];

% Strip duplicate spatial nodes caused by the cyclic boundary closure
[s_opt, unique_idx] = unique(s_opt_raw);

% Target interpolation density (matches GPS log length)
N_dense = length(x_rc);
s_dense = linspace(0, s_opt(end), N_dense)';

% Interpolate spatial coordinates
x_plot_dense = interp1(s_opt, x_plot(unique_idx), s_dense, 'makima');
y_plot_dense = interp1(s_opt, y_plot(unique_idx), s_dense, 'makima');

% Fill boundary NaNs with nearest valid values to prevent NaN propagation
t_clean     = fillmissing([t_opt; t_opt(end)], 'nearest');
v_clean     = fillmissing([v_opt; v_opt(1)], 'nearest');
ax_clean    = fillmissing([NaN; a_x_nodes; NaN; NaN], 'nearest');
ay_clean    = fillmissing([NaN; a_y_nodes; NaN; NaN], 'nearest');
jx_clean    = fillmissing([NaN; NaN; j_x; NaN; NaN], 'nearest');
jy_clean    = fillmissing([NaN; NaN; j_y; NaN; NaN], 'nearest');
kappa_clean = fillmissing([NaN; kappa; NaN; NaN], 'nearest');

% Apply unique index mask to kinematics before interpolating
t_clean     = t_clean(unique_idx);
v_clean     = v_clean(unique_idx);
ax_clean    = ax_clean(unique_idx);
ay_clean    = ay_clean(unique_idx);
jx_clean    = jx_clean(unique_idx);
jy_clean    = jy_clean(unique_idx);
kappa_clean = kappa_clean(unique_idx);

% Interpolate kinematics to the dense array
hover_data.t     = interp1(s_opt, t_clean, s_dense, 'makima');
hover_data.v     = interp1(s_opt, v_clean, s_dense, 'makima');
hover_data.ax    = interp1(s_opt, ax_clean, s_dense, 'makima');
hover_data.ay    = interp1(s_opt, ay_clean, s_dense, 'makima');
hover_data.jx    = interp1(s_opt, jx_clean, s_dense, 'makima');
hover_data.jy    = interp1(s_opt, jy_clean, s_dense, 'makima');
hover_data.kappa = interp1(s_opt, kappa_clean, s_dense, 'makima');
hover_data.s     = s_dense;


%% 8. Plot Results
% Create or update Figure 1, force docking, and clear previous run data
figure(1);
set(gcf, 'WindowStyle', 'docked');
clf;

% Calculate continuous track boundary coordinates
bound_L_x = ref_path.x - ref_path.W_left .* sin(ref_path.psi);
bound_L_y = ref_path.y + ref_path.W_left .* cos(ref_path.psi);
bound_R_x = ref_path.x - (-ref_path.W_right) .* sin(ref_path.psi);
bound_R_y = ref_path.y + (-ref_path.W_right) .* cos(ref_path.psi);

% Plot the continuous boundaries
plot(bound_L_x, bound_L_y, '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5, 'HandleVisibility', 'off'); hold on;
plot(bound_R_x, bound_R_y, '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Calculate track bounding box to make figure height 5% taller than drawing
x_all = [ref_path.x; bound_L_x; bound_R_x];
y_all = [ref_path.y; bound_L_y; bound_R_y];
x_min = min(x_all); x_max = max(x_all);
y_min = min(y_all); y_max = max(y_all);

x_span = x_max - x_min;
y_span = y_max - y_min;

% Add 5% extra height symmetrically to Y-limits
y_padding = (y_span * 1.05 - y_span) / 2;
xlim([x_min - 2, x_max + 2]);
ylim([(y_min - y_padding) - 2, (y_max + y_padding) + 2]);

% Setup axes, preserve true 1:1 scale, and enable grid
daspect([1 1 1]); 
grid on;

% Setup Centerline and Dummy Lines for Legend
h_center = plot(ref_path.x, ref_path.y, 'k--', 'DisplayName', 'Centerline'); hold on;
h_opt_dummy = plot(NaN, NaN, 'Color', [1, 1, 0], 'LineStyle', '--', 'LineWidth', 2);
h_gps_dummy = plot(NaN, NaN, 'Color', [1, 1, 0], 'LineStyle', '-', 'LineWidth', 2);

% Draw Start/Finish Line at the first node (where counting starts)
h_fin = plot([bound_L_x(1), bound_R_x(1)], [bound_L_y(1), bound_R_y(1)], ...
    'w-', 'LineWidth', 2, 'DisplayName', 'Start/Finish');


% --- Delta Time Track Ribbon ---
% Close the boundary loops to match the N+1 length of x_plot / y_plot
bLx_plot = [bound_L_x; bound_L_x(1)];
bLy_plot = [bound_L_y; bound_L_y(1)];
bRx_plot = [bound_R_x; bound_R_x(1)];
bRy_plot = [bound_R_y; bound_R_y(1)];

% Interpolate actual track limits to the dense array used for coloring
X_left  = interp1(s_opt, bLx_plot(unique_idx), s_dense, 'makima');
Y_left  = interp1(s_opt, bLy_plot(unique_idx), s_dense, 'makima');
X_right = interp1(s_opt, bRx_plot(unique_idx), s_dense, 'makima');
Y_right = interp1(s_opt, bRy_plot(unique_idx), s_dense, 'makima');

% Extract unique GPS spatial data for interpolation
s_rc_raw_temp = [0; cumsum(ds_rc)];
[s_rc_clean_temp, u_idx] = unique(s_rc_raw_temp);
v_gps_clean = v_rc(u_idx);

% Calculate spatial delta time rate (s/m)
v_gps_dense = interp1(s_rc_clean_temp, v_gps_clean, s_dense, 'linear', 'extrap');
v_gps_dense(v_gps_dense < 1) = 1; % Prevent division by zero at standstill
delta_rate = (1 ./ v_gps_dense) - (1 ./ hover_data.v);

% Map delta rate explicitly to RGB arrays
% Positive (losing time) -> Red, Negative (gaining/neutral) -> Green
rate_norm = max(min(delta_rate / 0.015, 1), -1); 
R_rib = 0.15 + 0.65 * (rate_norm > 0) .* rate_norm; 
G_rib = 0.15 + 0.65 * (rate_norm < 0) .* abs(rate_norm); 
B_rib = 0.15 * ones(size(rate_norm));

% Format as 2D patch geometry to preserve line anti-aliasing
N_pts = length(s_dense);
V = [X_left(:), Y_left(:); X_right(:), Y_right(:)];
F = [(1:N_pts-1)', (1:N_pts-1)'+N_pts, (2:N_pts)'+N_pts, (2:N_pts)'];
C = [R_rib(:), G_rib(:), B_rib(:); R_rib(:), G_rib(:), B_rib(:)];

% Plot the colored ribbon under the telemetry lines
patch('Vertices', V, 'Faces', F, 'FaceVertexCData', C, ...
    'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.4, ...
    'HandleVisibility', 'off');


% 1. Color Mapping
% Calculate longitudinal G-force for the color mapping
G_long = hover_data.ax / g;

% Create a continuous multi-colored line using patch
p_opt = patch([x_plot_dense; NaN], [y_plot_dense; NaN], [G_long; NaN], ...
    'FaceColor', 'none', ...
    'EdgeColor', 'interp', ...
    'LineStyle', '--', ...
    'LineWidth', 2, ...
    'DisplayName', 'Optimal Line', ...
    'UserData', hover_data);

% Define a Red (Braking) -> Yellow (Neutral) -> Green (Acceleration) colormap
cmap = [linspace(1, 1, 128)', linspace(0, 1, 128)', zeros(128, 1); ...
        linspace(1, 0, 128)', linspace(1, 1, 128)', zeros(128, 1)];
colormap(gca, cmap);

% Force symmetric color limits so 0 G is exactly yellow
max_G = max(A_long_fwd, A_long_brake) / g;
try
    clim(gca, [-max_G, max_G]); % R2022a and newer
catch
    caxis(gca, [-max_G, max_G]); % Older MATLAB versions
end

% Add colorbar scale
cb = colorbar;
cb.Label.String = 'Longitudinal Acceleration (G)';
cb.Color = [0.9 0.9 0.9];

% 2. Setup axes and labels
% Force axis ticks to 2m intervals
axis equal; grid on;

xticks(-100:2:100);
yticks(-100:2:100);

true_lap_time = sum(dt_opt);
total_distance = sum(sqrt(diff(x_plot).^2 + diff(y_plot).^2));
xlabel('X (m)'); ylabel('Y (m)');

% Calculate the actual minimum radius used on the optimal trajectory
R_actual_min = min(1 ./ abs(kappa));

% Construct the parameter string
param_str = sprintf('Vehicle Limits:\nMass: %.0f kg\nPower: %.1f kW\nEfficiency: %.0f%%\nV_{max}: %.0f km/h\nA_{lat}: %.2f G\nA_{long, fwd}: %.2f G\nA_{long, brake}: %.2f G\nJ_{lat}: %.1f-%.1f G/s\nV(J_{lat}): %.0f-%.0f km/h\nJ_{long}: %.1f G/s\nR_{min, actual}: %.2f m', ...
    mass, P_motor/1000, efficiency*100, v_max*3.6, A_lat/g, A_long_fwd/g, A_long_brake/g, lat_jerk_max/g, lat_jerk_min/g, speed_lat_jerk_min*3.6, speed_lat_jerk_max*3.6, J_long/g, R_actual_min);

% Create the text box fixed in the top-left corner using normalized units
text(0.02, 0.98, param_str, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.15 0.15 0.15], ... 
    'Color', [0.9 0.9 0.9], ...              
    'EdgeColor', [0.5 0.5 0.5], ...          
    'Margin', 5, ...
    'Interpreter', 'tex');

% GPS Telemetry Processing
s_rc_raw = [0; cumsum(ds_rc)];
[s_rc_clean, unique_idx] = unique(s_rc_raw);
v_rc_clean = v_rc(unique_idx);
v_rc_interp = interp1(s_rc_clean, v_rc_clean, s_lap, 'linear', 'extrap');

% Calculate longitudinal G-force for the color mapping
G_long_rc = rc_hover.ax / 9.81;

% Create a continuous colored solid line for telemetry
p_gps = patch([x_rc; NaN], [y_rc; NaN], [G_long_rc; NaN], ...
    'FaceColor', 'none', ...
    'EdgeColor', 'interp', ...
    'LineStyle', '-', ...
    'LineWidth', 2, ...
    'DisplayName', 'GPS', ...
    'UserData', rc_hover);


% Generate the corrected legend for the continuous track
lgd = legend([h_center, h_opt_dummy, h_gps_dummy, h_fin], ...
    {'Centerline', 'Optimal Line', 'GPS', 'Start/Finish'}, ...
    'Location', 'southeast');

%% 9. Speed Traps
% --- Speed Annotations (Min/Max/Trap) ---

% Require a 2 km/h (2/3.6 m/s) prominence to register a new speed extremum
prominence_threshold = 2 / 3.6;

% 1. Optimal Line Extrema & Top Speed Zones (Green - Pushed OUTSIDE the track)
[opt_max_v, opt_max_idx] = findpeaks(v_opt, 'MinPeakProminence', prominence_threshold);
[opt_min_v, opt_min_idx] = findpeaks(-v_opt, 'MinPeakProminence', prominence_threshold);
opt_min_v = -opt_min_v;

% Identify flat-out Top Speed zones (within 0.2 m/s of v_max)
at_vmax = (v_max - v_opt) < 0.2;
vmax_trans = diff([0; at_vmax; 0]);
vmax_starts = find(vmax_trans == 1);
vmax_ends = find(vmax_trans == -1) - 1;

% Filter for sustained top speed (more than 10 solver nodes)
sustained = (vmax_ends - vmax_starts) > 10;
vmax_starts = vmax_starts(sustained);
vmax_ends = vmax_ends(sustained);
vmax_indices = [vmax_starts; vmax_ends];

% Remove standard peaks that fall inside top speed zones to prevent overlapping text
valid_max = true(size(opt_max_idx));
for i = 1:length(opt_max_idx)
    if at_vmax(opt_max_idx(i))
        valid_max(i) = false;
    end
end
opt_max_idx = opt_max_idx(valid_max);
opt_max_v = opt_max_v(valid_max);

% Append the top speed start/end points to the max list for plotting
opt_max_idx = [opt_max_idx; vmax_indices];
opt_max_v = [opt_max_v; v_opt(vmax_indices)];

% Filter out points near the start/finish line to prevent overlapping the trap
margin = 5; 
valid_max_margin = (opt_max_idx > margin) & (opt_max_idx < N - margin);
opt_max_idx = opt_max_idx(valid_max_margin);
opt_max_v = opt_max_v(valid_max_margin);

valid_min_margin = (opt_min_idx > margin) & (opt_min_idx < N - margin);
opt_min_idx = opt_min_idx(valid_min_margin);
opt_min_v = opt_min_v(valid_min_margin);

for i = 1:length(opt_max_idx)
    idx = opt_max_idx(i);
    x0 = x_opt(idx); y0 = y_opt(idx);
    
    % Calculate perpendicular normal vector
    idx_f = min(idx+1, length(x_opt)); idx_b = max(idx-1, 1);
    dx = x_opt(idx_f) - x_opt(idx_b); dy = y_opt(idx_f) - y_opt(idx_b);
    L = sqrt(dx^2 + dy^2); if L==0, L=1; end
    nx = -dy/L; ny = dx/L; % Left of path (Outside)
    ox = 2.625 * nx; oy = 2.625 * ny; 
    
    plot(x0, y0, 'o', 'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
    plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0 0.8 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
    text(x0+ox, y0+oy, sprintf('%.0f', opt_max_v(i)*3.6), ...
        'BackgroundColor', [0.1 0.3 0.1], 'Color', 'w', 'EdgeColor', [0 0.8 0], ...
        'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');
end

for i = 1:length(opt_min_idx)
    idx = opt_min_idx(i);
    x0 = x_opt(idx); y0 = y_opt(idx);
    
    idx_f = min(idx+1, length(x_opt)); idx_b = max(idx-1, 1);
    dx = x_opt(idx_f) - x_opt(idx_b); dy = y_opt(idx_f) - y_opt(idx_b);
    L = sqrt(dx^2 + dy^2); if L==0, L=1; end
    nx = -dy/L; ny = dx/L; 
    ox = 2.625 * nx; oy = 2.625 * ny; 
    
    plot(x0, y0, 'o', 'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
    plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0 0.8 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
    text(x0+ox, y0+oy, sprintf('%.0f', opt_min_v(i)*3.6), ...
        'BackgroundColor', [0.1 0.3 0.1], 'Color', 'w', 'EdgeColor', [0 0.8 0], ...
        'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');
end

% 2. GPS Line Extrema (Red - Pushed INSIDE the track)
v_gps_sm = smoothdata(rc_hover.v, 'gaussian', 20);
[~, gps_max_idx] = findpeaks(v_gps_sm, 'MinPeakProminence', prominence_threshold);
[~, gps_min_idx] = findpeaks(-v_gps_sm, 'MinPeakProminence', prominence_threshold);

for i = 1:length(gps_max_idx)
    idx = gps_max_idx(i);
    x0 = x_rc(idx); y0 = y_rc(idx);
    
    idx_f = min(idx+5, length(x_rc)); idx_b = max(idx-5, 1);
    dx = x_rc(idx_f) - x_rc(idx_b); dy = y_rc(idx_f) - y_rc(idx_b);
    L = sqrt(dx^2 + dy^2); if L==0, L=1; end
    nx = dy/L; ny = -dx/L; % Right of path (Inside)
    ox = 2.625 * nx; oy = 2.625 * ny; 
    
    plot(x0, y0, 'o', 'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
    plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0.8 0 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
    text(x0+ox, y0+oy, sprintf('%.0f', rc_hover.v(idx)*3.6), ...
        'BackgroundColor', [0.3 0.1 0.1], 'Color', 'w', 'EdgeColor', [0.8 0 0], ...
        'LineStyle', '--', 'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');
end

for i = 1:length(gps_min_idx)
    idx = gps_min_idx(i);
    x0 = x_rc(idx); y0 = y_rc(idx);
    
    idx_f = min(idx+5, length(x_rc)); idx_b = max(idx-5, 1);
    dx = x_rc(idx_f) - x_rc(idx_b); dy = y_rc(idx_f) - y_rc(idx_b);
    L = sqrt(dx^2 + dy^2); if L==0, L=1; end
    nx = dy/L; ny = -dx/L; 
    ox = 2.625 * nx; oy = 2.625 * ny;
    
    plot(x0, y0, 'o', 'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
    plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0.8 0 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
    text(x0+ox, y0+oy, sprintf('%.0f', rc_hover.v(idx)*3.6), ...
        'BackgroundColor', [0.3 0.1 0.1], 'Color', 'w', 'EdgeColor', [0.8 0 0], ...
        'LineStyle', '--', 'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');
end

% 3. Trap Speeds (Start/Finish Line & GPS Extents)
% Optimal Trap
x0 = x_opt(1); y0 = y_opt(1);
ox = 0; oy = 2.25; % Push rigidly Up
plot(x0, y0, 'o', 'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0 0.8 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
text(x0+ox, y0+oy, sprintf('%.0f', v_opt(1)*3.6), ...
    'BackgroundColor', [0.1 0.3 0.1], 'Color', 'w', 'EdgeColor', [0 0.8 0], ...
    'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');

% GPS First Data Point (45 degrees forward and inward)
x0 = x_rc(1); y0 = y_rc(1);
idx_f = min(6, length(x_rc)); 
dx = x_rc(idx_f) - x_rc(1); dy = y_rc(idx_f) - y_rc(1);
L = sqrt(dx^2 + dy^2); if L==0, L=1; end
tx = dx/L; ty = dy/L;  % Lap direction
nx = dy/L; ny = -dx/L; % Inward direction
ox = 2.625 * (tx + nx) / sqrt(2); 
oy = 2.625 * (ty + ny) / sqrt(2);

plot(x0, y0, 'o', 'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0.8 0 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
text(x0+ox, y0+oy, sprintf('%.0f', rc_hover.v(1)*3.6), ...
    'BackgroundColor', [0.3 0.1 0.1], 'Color', 'w', 'EdgeColor', [0.8 0 0], 'LineStyle', '--', ...
    'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');

% GPS Last Data Point (45 degrees backward and inward)
x0 = x_rc(end); y0 = y_rc(end);
idx_b = max(length(x_rc)-5, 1);
dx = x_rc(end) - x_rc(idx_b); dy = y_rc(end) - y_rc(idx_b);
L = sqrt(dx^2 + dy^2); if L==0, L=1; end
tx = dx/L; ty = dy/L;  % Lap direction
nx = dy/L; ny = -dx/L; % Inward direction
ox = 2.625 * (-tx + nx) / sqrt(2); 
oy = 2.625 * (-ty + ny) / sqrt(2);

plot(x0, y0, 'o', 'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'HandleVisibility', 'off');
plot([x0, x0+ox], [y0, y0+oy], '--', 'Color', [0.8 0 0], 'LineWidth', 0.5, 'HandleVisibility', 'off');
text(x0+ox, y0+oy, sprintf('%.0f', rc_hover.v(end)*3.6), ...
    'BackgroundColor', [0.3 0.1 0.1], 'Color', 'w', 'EdgeColor', [0.8 0 0], 'LineStyle', '--', ...
    'FontSize', 8, 'Margin', 2, 'HorizontalAlignment', 'center');

% ----------------------------------------



%% 10. Cursor, Comparison box, Title

% Disable default data cursor mode
datacursormode(gcf, 'off');

% Create fixed HUD box in the bottom-left corner
text(0.02, 0.02, 'Hover over track to load telemetry comparison...', ...
    'Units', 'normalized', ...
    'VerticalAlignment', 'bottom', ...
    'BackgroundColor', [0.15 0.15 0.15], ...
    'Color', [0.9 0.9 0.9], ...
    'EdgeColor', [0.5 0.5 0.5], ...
    'Margin', 5, ...
    'Interpreter', 'tex', ...
    'Tag', 'HUD_Text');

% Bind the motion tracker to update the HUD continuously
set(gcf, 'WindowButtonMotionFcn', @hud_update_callback);

% Update figure title with comparative data and track name
gps_lap_time = t_rc(end);
gps_distance = s_rc_raw(end);

title(sprintf('[%s] Optimal Line (%.2fs | %.1fm) vs GPS (%.2fs | %.1fm)', ...
    track_name, true_lap_time, total_distance, gps_lap_time, gps_distance), ...
    'Interpreter', 'none');




%% 11. Background HUD Update Function
function hud_update_callback(fig, ~)
    ax = findobj(fig, 'Type', 'Axes');
    if isempty(ax), return; end
    ax = ax(1);

    hud = findobj(ax, 'Tag', 'HUD_Text');
    if isempty(hud), return; end

    cp = ax.CurrentPoint;
    cx = cp(1,1);
    cy = cp(1,2);

    opt_patch = findobj(ax, 'Type', 'Patch', 'DisplayName', 'Optimal Line');
    gps_patch = findobj(ax, 'Type', 'Patch', 'DisplayName', 'GPS');

    if isempty(opt_patch) || isempty(gps_patch), return; end

    opt_data = opt_patch.UserData;
    gps_data = gps_patch.UserData;

    % Calculate cursor distance to both lines
    dist_opt = sqrt((opt_patch.XData - cx).^2 + (opt_patch.YData - cy).^2);
    [min_d_opt, idx_opt] = min(dist_opt);

    dist_gps = sqrt((gps_patch.XData - cx).^2 + (gps_patch.YData - cy).^2);
    [min_d_gps, idx_gps] = min(dist_gps);

    % If cursor is far away from the track, clear the data
    if min_d_opt > 3 && min_d_gps > 3
        hud.String = 'Hover over track to load telemetry comparison...';
        return;
    end

    % Standardize Deltas (GPS - Optimal)
    delta_t = gps_data.t(idx_gps) - opt_data.t(idx_opt);
    delta_s = gps_data.s(idx_gps) - opt_data.s(idx_opt);
    g = 9.81; 

    txt = {
        '--- GPS ---';
        sprintf('Elapsed:  %6.2f s', gps_data.t(idx_gps));
        sprintf('Time Delta:%+6.2f s', delta_t);
        sprintf('Dist Delta:%+6.1f m', delta_s);
        sprintf('Velocity: %6.1f km/h', gps_data.v(idx_gps) * 3.6);
        sprintf('A_{long}: %6.2f G', gps_data.ax(idx_gps) / g);
        sprintf('A_{lat}:  %6.2f G', gps_data.ay(idx_gps) / g);
        sprintf('J_{long}: %6.2f G/s', gps_data.jx(idx_gps) / g);
        sprintf('J_{lat}:  %6.2f G/s', gps_data.jy(idx_gps) / g);
        ' ';
        '--- OPTIMAL LINE ---';
        sprintf('Elapsed:  %6.2f s', opt_data.t(idx_opt));
        sprintf('Velocity: %6.1f km/h', opt_data.v(idx_opt) * 3.6);
        sprintf('Radius:   %6.1f m', 1/abs(opt_data.kappa(idx_opt)));
        sprintf('A_{long}: %6.2f G', opt_data.ax(idx_opt) / g);
        sprintf('A_{lat}:  %6.2f G', opt_data.ay(idx_opt) / g);
        sprintf('J_{long}: %6.2f G/s', opt_data.jx(idx_opt) / g);
        sprintf('J_{lat}:  %6.2f G/s', opt_data.jy(idx_opt) / g)
    };

    hud.String = txt;
end



%% 12. Track Selection Helper

% Track Definition
% Format: [X, Y, Keepout, Direction (1=CW, -1=CCW)]
% P1 and P2 define the start/finish gate (X=0 cross-section)

function pts = track_selection(track_name)
    % 1. Load from CSV if extension is provided
    if endsWith(track_name, '.csv', 'IgnoreCase', true)
        if isfile(track_name)
            pts = readmatrix(track_name);
            return;
        else
            error('Track file "%s" not found in the current directory.', track_name);
        end
    end

    % 2. Legacy hardcoded tracks
    switch track_name
        case 'T race CW'
            pts = [
                  0.0,  30.0, 0.4, -1;
                  0.0,  20.0, 0.4,  1;
                 10.0,  20.0, 0.4,  1;
                  5.0,  14.0, 0.4, -1;
                  0.0, -20.0, 0.4,  1;
                 -5.0,  14.0, 0.4, -1;
                -10.0,  20.0, 0.4,  1
                ];
    
        case 'T race CCW'
            pts = [
                  0.0,  30.0, 0.4,  1;
                  0.0,  20.0, 0.4, -1;
                -10.0,  20.0, 0.4, -1;
                 -5.0,  14.0, 0.4,  1;
                  0.0, -20.0, 0.4, -1;
                  5.0,  14.0, 0.4,  1;
                 10.0,  20.0, 0.4, -1
                ];
                
        otherwise
            error('Track "%s" not found. Provide a valid .csv filename or hardcoded option.', track_name);
    end
end