% Racing Line and Lap Time Optimization using Frenet-Serret Frame
clear; clc; close all;

%% 1. Parameters and Constraints
g = 9.81; % m/s^2

% Vehicle Limits
A_lat = 1.5 * g;
A_long_fwd = 0.6 * g;
A_long_brake = 0.6 * g;
J_lat = 4.0 * g;
J_long = 1.5 * g;
R_min = 2.0;

% Track Definition
% Format: [X, Y, Keepout, Direction (1=CW, -1=CCW)]
% P1 and P2 define the start/finish gate (X=0 cross-section)

t_race_cw = [
    0.0,   30.0,  0.4, -1;  % P1 Finish Top
    0.0,   20.0,  0.4,  1;  % P2 Finish Bottom
    10.0,   20.0,  0.4,  1;  % P3
    5.0,   14.0,  0.4, -1;  % P4
    0.0,  -20.0,  0.4,  1;  % P5
    -5.0,   14.0,  0.4, -1;  % P6
    -10.0,   20.0,  0.4,  1   % P7
    ];

track_pts = t_race_cw;
nodes = 100;

% Telemetry Rotation
theta = 3.665; % radians
% Telemetry Translation
translate_x = 0;
translate_y = -20.5;
% Telemetry Trimming
rc_start_idx = 178;
rc_end_idx = 725;

%% 2. Generate Reference Centerline with Periodic Boundaries
gate_mid = (track_pts(1, 1:2) + track_pts(2, 1:2)) / 2;
pts_base = [gate_mid; track_pts(3:end, 1:2)];
num_base_pts = size(pts_base, 1);

% Replicate base track 3 times to ensure complete periodic symmetry
pts_ext = repmat(pts_base, 3, 1);
pts_ext = [pts_ext; pts_base(1, :)]; % Close trailing loop

dx_ext = diff(pts_ext(:,1));
dy_ext = diff(pts_ext(:,2));
d_chord_ext = sqrt(dx_ext.^2 + dy_ext.^2);
s_ext = [0; cumsum(d_chord_ext)];

% Center lap starts at the gate of Lap 2 and ends at the gate of Lap 3
s_start = s_ext(num_base_pts + 1);
s_end   = s_ext(2 * num_base_pts + 1);

N_ext = 600; 
s_interp_ext = linspace(s_ext(1), s_ext(end), N_ext);
ref_x_raw = makima(s_ext, pts_ext(:,1), s_interp_ext)';
ref_y_raw = makima(s_ext, pts_ext(:,2), s_interp_ext)';

ref_x_smooth = smoothdata(ref_x_raw, 'gaussian', 15);
ref_y_smooth = smoothdata(ref_y_raw, 'gaussian', 15);

dx_ext_dense = gradient(ref_x_smooth);
dy_ext_dense = gradient(ref_y_smooth);
psi_ext_dense = unwrap(atan2(dy_ext_dense, dx_ext_dense));

N = nodes;
s_lap = linspace(s_start, s_end, N);
ref_path.x = interp1(s_interp_ext, ref_x_smooth, s_lap)';
ref_path.y = interp1(s_interp_ext, ref_y_smooth, s_lap)';
ref_path.psi = interp1(s_interp_ext, psi_ext_dense, s_lap)';

% Ensure exact machine-precision closure
ref_path.x(end)   = ref_path.x(1);
ref_path.y(end)   = ref_path.y(1);
ref_path.psi(end) = ref_path.psi(1);

%% 3. State Vector Initialization (CasADi)
import casadi.*
opti = casadi.Opti();

% Declare optimization variables
n  = opti.variable(N, 1);       % Lateral deviation (m)
v  = opti.variable(N, 1);       % Velocity (m/s)
dt = opti.variable(N-1, 1);     % Time step (s)

%% 4. Bounds and Track Limits
lb_n = -8 * ones(N, 1);
ub_n =  8 * ones(N, 1);

for i = 1:size(track_pts, 1)
    Px = track_pts(i,1);
    Py = track_pts(i,2);
    keepout = track_pts(i,3);
    is_CW = (track_pts(i,4) == 1);
    
    dist = sqrt((ref_path.x - Px).^2 + (ref_path.y - Py).^2);
    [~, idx] = min(dist);
    
    nx = -sin(ref_path.psi(idx));
    ny =  cos(ref_path.psi(idx));
    n_apex = (Px - ref_path.x(idx)) * nx + (Py - ref_path.y(idx)) * ny;
    
    chord_margin = 0.2; 
    
    % Periodic index neighbors
    prev_idx = idx - 1;
    if prev_idx < 1, prev_idx = N - 1; end
    next_idx = idx + 1;
    if next_idx > N, next_idx = 2; end
    
    if is_CW
        lb_n(idx)      = max(lb_n(idx), n_apex + keepout + chord_margin);
        lb_n(prev_idx) = max(lb_n(prev_idx), n_apex + keepout);
        lb_n(next_idx) = max(lb_n(next_idx), n_apex + keepout);
        if idx == 1, lb_n(N) = lb_n(1); end
        if idx == N, lb_n(1) = lb_n(N); end
    else
        ub_n(idx)      = min(ub_n(idx), n_apex - keepout - chord_margin);
        ub_n(prev_idx) = min(ub_n(prev_idx), n_apex - keepout);
        ub_n(next_idx) = min(ub_n(next_idx), n_apex - keepout);
        if idx == 1, ub_n(N) = ub_n(1); end
        if idx == N, ub_n(1) = ub_n(N); end
    end
end

opti.subject_to(lb_n <= n <= ub_n);
opti.subject_to(v >= 1.0);
opti.subject_to(dt >= 0.01);

opti.set_initial(n, zeros(N, 1));
opti.set_initial(v, 10 * ones(N, 1));
opti.set_initial(dt, 0.2 * ones(N-1, 1));

%% 5. Objective & Nonlinear Constraints (CasADi)
W_smooth = 0.05 * (N / 100); 
W_vel_smooth = 0.005 * (N / 100); 

costFunc = sum(dt) ...
    + W_smooth * sumsqr(diff([n; n(1)])) ...
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

opti.subject_to(a_x_nodes.^2 + a_y_nodes.^2 <= A_lat^2);
opti.subject_to(-A_long_brake <= a_x_nodes <= A_long_fwd);

% 6. Periodic Jerk Constraints
dt_nodes = 0.5 * ([dt(end); dt(1:end-1)] + dt);
j_x = diff([a_x_nodes; a_x_nodes(1)]) ./ dt_nodes;
j_y = diff([a_y_nodes; a_y_nodes(1)]) ./ dt_nodes;

opti.subject_to(-J_long <= j_x <= J_long);
opti.subject_to(-J_lat <= j_y <= J_lat);

% Solver Options & Execution
p_opts = struct('expand', true);
s_opts = struct('max_iter', 2000, 'tol', 1e-6);
opti.solver('ipopt', p_opts, s_opts);

disp('Executing CasADi/IPOPT solver...');
sol = opti.solve();

true_lap_time = sum(sol.value(dt));
fprintf('True Lap Time: %.3f s\n', true_lap_time);











%% 6. Extract, Calculate Kinematics, and Plot Results
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
N_dense = rc_end_idx - rc_start_idx + 1;
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

figure('Position', [200, 200, 700, 800]);
plot(ref_path.x, ref_path.y, 'k--', 'DisplayName', 'Centerline'); hold on;

% Plot using the dense arrays
p_opt = plot(x_plot_dense, y_plot_dense, 'b-', 'LineWidth', 2, 'DisplayName', 'Optimal Line', 'UserData', hover_data);
scatter(track_pts(:,1), track_pts(:,2), 50, 'r', 'filled', 'DisplayName', 'Track Apexes');
lgd = legend;
lgd.Position(1) = lgd.Position(1) - 0.05; 
lgd.Position(2) = lgd.Position(2) - 0.4; 
axis equal; grid on;

% Force axis ticks to 2m intervals
xticks(-18:2:18);
yticks(-30:2:30);

true_lap_time = sum(dt_opt);
total_distance = sum(sqrt(diff(x_plot).^2 + diff(y_plot).^2));
title(sprintf('Optimal Racing Line (Lap Time: %.3f s | Distance: %.2f m)', true_lap_time, total_distance));
xlabel('X (m)'); ylabel('Y (m)');

% Calculate the actual minimum radius used on the optimal trajectory
R_actual_min = min(1 ./ abs(kappa));

% Construct the parameter string
param_str = sprintf('Vehicle Limits:\nA_{lat}: %.1f G\nA_{long, fwd}: %.1f G\nA_{long, brake}: %.1f G\nJ_{lat}: %.1f G/s\nJ_{long}: %.1f G/s\nR_{actual}: %.2f m', ...
    A_lat/g, A_long_fwd/g, A_long_brake/g, J_lat/g, J_long/g, R_actual_min);

annotation('textbox', [0.7, 0.55, 0.2, 0.2], 'String', param_str, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', [0.15 0.15 0.15], ... 
    'Color', [0.9 0.9 0.9], ...              
    'EdgeColor', [0.5 0.5 0.5], ...          
    'Interpreter', 'tex');

% Enable interactive data cursor
dcm = datacursormode(gcf);
dcm.Enable = 'on';
dcm.UpdateFcn = @hover_callback;

%% Constraint Function
function [c, ceq] = vehicle_constraints(X, N, A_lat, A_long_fwd, A_long_brake, J_lat, J_long, R_min, ref_path)
    n  = X(1:N);
    v  = X(N+1:2*N);
    dt = X(2*N+1:end); % Length N-1
    
    c = [];
    ceq = [];
    
    % 1. Cartesian Coordinates
    x = ref_path.x - n .* sin(ref_path.psi);
    y = ref_path.y + n .* cos(ref_path.psi);
    
    dx = diff(x); % Length N-1
    dy = diff(y);
    ds = sqrt(dx.^2 + dy.^2); 
    
    % 2. Kinematics Equality (Distance vs Velocity)
    v_avg = 0.5 * (v(1:N-1) + v(2:N));
    ceq = [ceq; ds - v_avg .* dt];
    
    % 3. Flying Start (Periodic Boundary Conditions)
    % C0 Continuity: Match lateral deviation and speed exactly
    ceq = [ceq; n(1) - n(N)];
    ceq = [ceq; v(1) - v(N)];
    
    % C1 Continuity: Match the rates of change in the Frenet frame.
    % This allows the Cartesian path to follow the curvature smoothly.
    dn_start = (n(2) - n(1)) / dt(1);
    dn_end   = (n(N) - n(N-1)) / dt(end);
    ceq = [ceq; dn_start - dn_end];
    
    dv_start = (v(2) - v(1)) / dt(1);
    dv_end   = (v(N) - v(N-1)) / dt(end);
    ceq = [ceq; dv_start - dv_end];
    
    % 4. Curvature Calculation
    heading = atan2(dy, dx); % Length N-1
    dHeading = diff(heading); % Length N-2
    dHeading(dHeading > pi) = dHeading(dHeading > pi) - 2*pi;
    dHeading(dHeading < -pi) = dHeading(dHeading < -pi) + 2*pi;
    
    kappa = dHeading ./ ds(1:N-2); 
    
    % 5. Turning Radius Limit
    c = [c; abs(kappa) - (1/R_min)];
    
    % 6. Acceleration Constraints
    a_x_seg = diff(v) ./ dt; % Length N-1
    a_x_nodes = 0.5 * (a_x_seg(1:end-1) + a_x_seg(2:end)); % Align to interior nodes (N-2)
    a_y_nodes = (v(2:N-1).^2) .* kappa; % Align to interior nodes (N-2)
    
    % Chopped Traction Circle
    c = [c; (a_x_nodes.^2 + a_y_nodes.^2) - A_lat^2];
    c = [c; a_x_nodes - A_long_fwd];        
    c = [c; -a_x_nodes - A_long_brake];     
    
    % 7. Jerk Constraints
    dt_nodes = 0.5 * (dt(1:end-1) + dt(2:end)); % Length N-2
    j_x = diff(a_x_nodes) ./ dt_nodes(1:end-1); % Length N-3
    j_y = diff(a_y_nodes) ./ dt_nodes(1:end-1); 
    
    c = [c; abs(j_x) - J_long];
    c = [c; abs(j_y) - J_lat];
end

%% Hover Callback Function
function txt = hover_callback(~, event_obj)
    pos = event_obj.Position;
    target_line = event_obj.Target;
    target_name = target_line.DisplayName;
    target_data = target_line.UserData;
    target_idx = event_obj.DataIndex;
    
    if isempty(target_data)
        txt = {sprintf('X: %.2f', pos(1)); sprintf('Y: %.2f', pos(2))};
        return;
    end
    
    ax = target_line.Parent;
    if strcmp(target_name, 'Optimal Line')
        other_name = 'RaceChrono Telemetry';
    else
        other_name = 'Optimal Line';
    end
    other_line = findobj(ax, 'Type', 'Line', 'DisplayName', other_name);
    
    g = 9.81; 
    
    has_other = ~isempty(other_line) && ~isempty(other_line.UserData);
    if has_other
        other_data = other_line.UserData;
        other_x = other_line.XData;
        other_y = other_line.YData;
        
        dists = sqrt((other_x - pos(1)).^2 + (other_y - pos(2)).^2);
        [~, other_idx] = min(dists);
        
        % Delta Time: (Telemetry - Optimal)
        if strcmp(target_name, 'RaceChrono Telemetry')
            delta_t = target_data.t(target_idx) - other_data.t(other_idx);
        else
            delta_t = other_data.t(other_idx) - target_data.t(target_idx);
        end
    end
    
    % 1. Build Target Line Data
    txt = {
        sprintf('--- %s ---', upper(target_name));
        sprintf('Elapsed:  %6.2f s', target_data.t(target_idx))
    };
    
    if has_other
        txt = [txt; {sprintf('Delta:   %+6.2f s', delta_t)}];
    end
    
    txt = [txt; {sprintf('Velocity: %6.1f km/h', target_data.v(target_idx) * 3.6)}];
    
    if strcmp(target_name, 'Optimal Line')
        txt = [txt; {sprintf('Radius:   %6.1f m', 1/abs(target_data.kappa(target_idx)))}];
    end
    
    txt = [txt; {
        sprintf('A_{long}: %6.2f G', target_data.ax(target_idx) / g);
        sprintf('A_{lat}:  %6.2f G', target_data.ay(target_idx) / g);
        sprintf('J_{long}: %6.2f G/s', target_data.jx(target_idx) / g);
        sprintf('J_{lat}:  %6.2f G/s', target_data.jy(target_idx) / g)
    }];
    
    % 2. Build Nearest Line Data
    if has_other
        txt = [txt; {
            ' ';
            sprintf('--- %s ---', upper(other_name));
            sprintf('Elapsed:  %6.2f s', other_data.t(other_idx));
            sprintf('Delta:   %+6.2f s', delta_t); 
            sprintf('Velocity: %6.1f km/h', other_data.v(other_idx) * 3.6)
        }];
        
        if strcmp(other_name, 'Optimal Line')
            txt = [txt; {sprintf('Radius:   %6.1f m', 1/abs(other_data.kappa(other_idx)))}];
        end
        
        txt = [txt; {
            sprintf('A_{long}: %6.2f G', other_data.ax(other_idx) / g);
            sprintf('A_{lat}:  %6.2f G', other_data.ay(other_idx) / g);
            sprintf('J_{long}: %6.2f G/s', other_data.jx(other_idx) / g);
            sprintf('J_{lat}:  %6.2f G/s', other_data.jy(other_idx) / g)
        }];
    end
end

%% Racebox data
% 1. Import Data
rc_data = readtable('racechrono_export.csv');
rc_data = rmmissing(rc_data);
lat = rc_data.latitude;
lon = rc_data.longitude;
v_rc = rc_data.speed;

% Extract accelerations and convert from G to m/s^2 for the struct
ax_rc_raw = rc_data.longitudinal_acc * 9.81;
ay_rc_raw = rc_data.lateral_acc * 9.81;

% 2. Equirectangular Projection to Local Cartesian (Meters)
R_earth = 6371000; % m
lat0 = mean(lat);  
lon0 = mean(lon);  

lat_rad = deg2rad(lat);
lon_rad = deg2rad(lon);
lat0_rad = deg2rad(lat0);
lon0_rad = deg2rad(lon0);
x_rc_raw = R_earth * cos(lat0_rad) .* (lon_rad - lon0_rad);
y_rc_raw = R_earth * (lat_rad - lat0_rad);

% 3. Trim Session Data
x_rc_raw = x_rc_raw(rc_start_idx:rc_end_idx);
y_rc_raw = y_rc_raw(rc_start_idx:rc_end_idx);
v_rc = v_rc(rc_start_idx:rc_end_idx);
ax_rc = ax_rc_raw(rc_start_idx:rc_end_idx);
ay_rc = ay_rc_raw(rc_start_idx:rc_end_idx);

% 4. Rotation (applied to raw projected coordinates)
R_mat = [cos(theta), -sin(theta); sin(theta), cos(theta)];
coords_rot = R_mat * [x_rc_raw, y_rc_raw]';
x_rot = coords_rot(1,:)';
y_rot = coords_rot(2,:)';

% 5. Translation (Anchoring P3 after rotation)
[~, idx_P3_rc] = min(y_rot);
dx = x_rot(idx_P3_rc) - translate_x;
dy = y_rot(idx_P3_rc) - translate_y;
x_rc = x_rot - dx;
y_rc = y_rot - dy;

% 6. Calculate Telemetry Kinematics for Hover Display
dx_rc = diff(x_rc);
dy_rc = diff(y_rc);
ds_rc = sqrt(dx_rc.^2 + dy_rc.^2);

% Derive time steps spatially 
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

% 7. Interpolation and Plotting
s_rc_raw = [0; cumsum(ds_rc)];
[s_rc_clean, unique_idx] = unique(s_rc_raw);
v_rc_clean = v_rc(unique_idx);
v_rc_interp = interp1(s_rc_clean, v_rc_clean, s_lap, 'linear', 'extrap');

plot(x_rc, y_rc, 'm-', 'LineWidth', 1.5, 'DisplayName', 'RaceChrono Telemetry', 'UserData', rc_hover);