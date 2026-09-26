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

% Import Data
rc_data = readtable('racechrono_export_1196.csv');
% Telemetry Rotation
theta = 3.697; % radians %3.665 for 12.09
% Telemetry Translation
translate_x = 0;
translate_y = -20.5;
% Telemetry Trimming
rc_start_idx = 193;
rc_end_idx = 734;

% Solver Nodes (recommended ~1 node/m)
nodes = 100;

% Track Definition
track_name = 'T race CW';
track_pts = track_selection(track_name);

% Options:
% T race CW
% T race CCW


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




%% 6. Racebox data
% 1. Import Data
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


%% 8.Plot Results
% Create or update Figure 1, force docking, and clear previous run data

figure(1);
set(gcf, 'WindowStyle', 'docked');
clf; 


% Setup Legend
h_center = plot(ref_path.x, ref_path.y, 'k--', 'DisplayName', 'Centerline'); hold on;
h_opt_dummy = plot(NaN, NaN, 'w--', 'LineWidth', 2);
h_apex = scatter(track_pts(:,1), track_pts(:,2), 50, 'm', 'filled', 'DisplayName', 'Track Apexes');
h_gps_dummy = plot(NaN, NaN, 'w-', 'LineWidth', 2);


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

xticks(-40:2:40);
yticks(-30:2:30);

true_lap_time = sum(dt_opt);
total_distance = sum(sqrt(diff(x_plot).^2 + diff(y_plot).^2));
xlabel('X (m)'); ylabel('Y (m)');

% Calculate the actual minimum radius used on the optimal trajectory
R_actual_min = min(1 ./ abs(kappa));

% Construct the parameter string
param_str = sprintf('Vehicle Limits:\nA_{lat}: %.1f G\nA_{long, fwd}: %.1f G\nA_{long, brake}: %.1f G\nJ_{lat}: %.1f G/s\nJ_{long}: %.1f G/s\nR_{min, actual}: %.2f m', ...
    A_lat/g, A_long_fwd/g, A_long_brake/g, J_lat/g, J_long/g, R_actual_min);

% Create the text box using data units and attach the drag callback
text(-36, 22, param_str, 'Units', 'data', ...
    'BackgroundColor', [0.15 0.15 0.15], ... 
    'Color', [0.9 0.9 0.9], ...              
    'EdgeColor', [0.5 0.5 0.5], ...          
    'Margin', 5, ...
    'Interpreter', 'tex');

% Enable interactive data cursor
dcm = datacursormode(gcf);
dcm.Enable = 'on';
dcm.UpdateFcn = @hover_callback;



s_rc_raw = [0; cumsum(ds_rc)];
[s_rc_clean, unique_idx] = unique(s_rc_raw);
v_rc_clean = v_rc(unique_idx);
v_rc_interp = interp1(s_rc_clean, v_rc_clean, s_lap, 'linear', 'extrap');

% Calculate longitudinal G-force for the color mapping
G_long_rc = rc_hover.ax / 9.81;

% Create a continuous colored dashed line for telemetry
p_gps = patch([x_rc; NaN], [y_rc; NaN], [G_long_rc; NaN], ...
    'FaceColor', 'none', ...
    'EdgeColor', 'interp', ...
    'LineWidth', 2, ...
    'DisplayName', 'GPS', ...
    'UserData', rc_hover);


% Generate the corrected legend using explicit handles and dummy lines
lgd = legend;
lgd.Position(1) = lgd.Position(1) - 0.02; 
lgd.Position(2) = lgd.Position(2) - 0.45; 
lgd = legend([h_center, h_opt_dummy, h_apex, h_gps_dummy], ...
    {'Centerline', 'Optimal Line', 'Track Apexes', 'GPS'}, ...
    'Location', 'southeast');


% Disable default data cursor mode
datacursormode(gcf, 'off');

% Create fixed HUD box in the bottom-left corner
text(0.02, 0.02, 'Hover over track to load telemetry...', ...
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




%% Background HUD Update Function
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
        hud.String = 'Hover over track to load telemetry...';
        return;
    end

    % Standardize Delta as (GPS Time - Optimal Time)
    delta_t = gps_data.t(idx_gps) - opt_data.t(idx_opt);
    g = 9.81; 

    txt = {
        '--- GPS ---';
        sprintf('Elapsed:  %6.2f s', gps_data.t(idx_gps));
        sprintf('Delta:   %+6.2f s', delta_t);
        sprintf('Velocity: %6.1f km/h', gps_data.v(idx_gps) * 3.6);
        sprintf('A_{long}: %6.2f G', gps_data.ax(idx_gps) / g);
        sprintf('A_{lat}:  %6.2f G', gps_data.ay(idx_gps) / g);
        sprintf('J_{long}: %6.2f G/s', gps_data.jx(idx_gps) / g);
        sprintf('J_{lat}:  %6.2f G/s', gps_data.jy(idx_gps) / g);
        ' ';
        '--- OPTIMAL LINE ---';
        sprintf('Elapsed:  %6.2f s', opt_data.t(idx_opt));
        sprintf('Delta:   %+6.2f s', -delta_t);
        sprintf('Velocity: %6.1f km/h', opt_data.v(idx_opt) * 3.6);
        sprintf('Radius:   %6.1f m', 1/abs(opt_data.kappa(idx_opt)));
        sprintf('A_{long}: %6.2f G', opt_data.ax(idx_opt) / g);
        sprintf('A_{lat}:  %6.2f G', opt_data.ay(idx_opt) / g);
        sprintf('J_{long}: %6.2f G/s', opt_data.jx(idx_opt) / g);
        sprintf('J_{lat}:  %6.2f G/s', opt_data.jy(idx_opt) / g)
    };

    hud.String = txt;
end



%% Track Selection Helper

% Track Definition
% Format: [X, Y, Keepout, Direction (1=CW, -1=CCW)]
% P1 and P2 define the start/finish gate (X=0 cross-section)

function pts = track_selection(track_name)
    switch track_name
        case 'T race CW'
            pts = [
                  0.0,  30.0, 0.4, -1; % F1 Finish Top
                  0.0,  20.0, 0.4,  1; % F2 Finish Bottom
                 10.0,  20.0, 0.4,  1; % T1
                  5.0,  14.0, 0.4, -1; % T2
                  0.0, -20.0, 0.4,  1; % T3
                 -5.0,  14.0, 0.4, -1; % T4
                -10.0,  20.0, 0.4,  1  % T5
                ];
    
        case 'T race CCW'
            pts = [
                  0.0,  30.0, 0.4,  1; % F1 Finish Top
                  0.0,  20.0, 0.4, -1; % F2 Finish Bottom
                -10.0,  20.0, 0.4, -1; % T1
                 -5.0,  14.0, 0.4,  1; % T2
                  0.0, -20.0, 0.4, -1; % T3
                  5.0,  14.0, 0.4,  1; % T4
                 10.0,  20.0, 0.4, -1  % T5
                ];
    
        otherwise
            error('Track "%s" not found. Check track_selection options.', track_name);
    end
end