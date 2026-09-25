% Racing Line and Lap Time Optimization using Frenet-Serret Frame
clear; clc; close all;

%% 1. Parameters and Constraints
g = 9.81; % m/s^2

% Vehicle Limits
A_lat = 1.5 * g;
A_long_fwd = 0.6 * g;
A_long_brake = 0.6 * g;
J_lat = 5.0 * g;
J_long = 2.0 * g;
R_min = 2.0;

% Track Definition
% Added P6 to force a sweeping turn and prevent Frenet frame collapse
% Format: [X, Y, Keepout, Direction (1=CW, -1=CCW)]
track_pts = [
    10.5,  20.0, 0,  1;  % P1
     4.6,  14.4, 0, -1;  % P2
     0.0, -20.5, 0,  1;  % P3
    -4.6,  14.4, 0, -1;  % P4
   -10.5,  20.0, 0,  1;  % P5
     0.0,  25.0, 0, -1   % P6 (Outer boundary rounding the top)
];

%% 2. Generate Reference Centerline with Periodic Boundaries
pts_base = track_pts(:, 1:2);

% Wrap points to create a periodic boundary for the spline
% Pads the array with the previous 2 and next 2 points to ensure smooth tangents
pts_ext = [pts_base(end-1:end, :); pts_base; pts_base(1:2, :)];

% Calculate cumulative chord distance
dx_ext = diff(pts_ext(:,1));
dy_ext = diff(pts_ext(:,2));
d_chord_ext = sqrt(dx_ext.^2 + dy_ext.^2);
s_ext = [0; cumsum(d_chord_ext)];

% Shift the lap boundary to the straight section between P6 and P1. 
% This prevents periodic boundary constraints from conflicting with apex bounds.
% P6 is at indices 2 and 8, P1 is at indices 3 and 9.
s_start = (s_ext(2) + s_ext(3)) / 2;
s_end   = (s_ext(8) + s_ext(9)) / 2;

% Create a dense interpolation over the ENTIRE extended array
N_ext = 300; 
s_interp_ext = linspace(s_ext(1), s_ext(end), N_ext);

ref_x_raw = makima(s_ext, pts_ext(:,1), s_interp_ext)';
ref_y_raw = makima(s_ext, pts_ext(:,2), s_interp_ext)';

% Apply the smoothing filter to the extended array to eliminate boundary effects
ref_x_smooth = smoothdata(ref_x_raw, 'gaussian', 15);
ref_y_smooth = smoothdata(ref_y_raw, 'gaussian', 15);

% Extract exactly one lap and resample to the requested N nodes
N = 100;
s_lap = linspace(s_start, s_end, N);
ref_path.x = interp1(s_interp_ext, ref_x_smooth, s_lap)';
ref_path.y = interp1(s_interp_ext, ref_y_smooth, s_lap)';

% Calculate heading of reference line
dx_ref = gradient(ref_path.x);
dy_ref = gradient(ref_path.y);
ref_path.psi = atan2(dy_ref, dx_ref);

%% 3. State Vector Initialization
% X = [n_1..n_N, v_1..v_N, dt_1..dt_{N-1}]
n_init = zeros(N, 1);          % Lateral deviation (m)
v_init = 10 * ones(N, 1);      % Velocity (m/s)
dt_init = 0.2 * ones(N-1, 1);  % Time step (s)

X0 = [n_init; v_init; dt_init];

%% 4. Bounds and Track Limits
lb = -inf(size(X0));
ub =  inf(size(X0));

% General lateral track limits
lb(1:N) = -15;
ub(1:N) =  15;

% Enforce CW / CCW keepout constraints exactly in Cartesian space mapping
for i = 1:size(track_pts, 1)
    Px = track_pts(i,1);
    Py = track_pts(i,2);
    keepout = track_pts(i,3);
    is_CW = (track_pts(i,4) == 1);

    % Find nearest node to the point on the smoothed reference line
    dist = sqrt((ref_path.x - Px).^2 + (ref_path.y - Py).^2);
    [~, idx] = min(dist);

    % Map true Cartesian apex to the smoothed Frenet frame
    % Normal vector points left of the path: [-sin(psi), cos(psi)]
    nx = -sin(ref_path.psi(idx));
    ny =  cos(ref_path.psi(idx));
    
    n_apex = (Px - ref_path.x(idx)) * nx + (Py - ref_path.y(idx)) * ny;

    % Margin to prevent discrete straight-line segments from cutting the corner
    chord_margin = 0.2; 

    if is_CW
        % CW turn (inner right apex). Vehicle must stay left.
        lb(idx) = n_apex + keepout + chord_margin;
        
        % Constrain adjacent nodes to reinforce the boundary against chord clipping
        if idx > 1, lb(idx-1) = n_apex + keepout; end
        if idx < N, lb(idx+1) = n_apex + keepout; end
    else
        % CCW turn (inner left apex). Vehicle must stay right.
        ub(idx) = n_apex - keepout - chord_margin;
        
        if idx > 1, ub(idx-1) = n_apex - keepout; end
        if idx < N, ub(idx+1) = n_apex - keepout; end
    end
end

% Velocity limits (v > 0)
lb(N+1 : 2*N) = 1.0; 

% Time delta limits (dt > 0)
lb(2*N+1 : end) = 0.01;

%% 5. Optimization
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'sqp', ...
    'MaxFunctionEvaluations', 2e5, ...
    'MaxIterations', 2000, ...
    'StepTolerance', 1e-6);

% Regularization weights
W_smooth = 0.05; 
W_vel_smooth = 0.02; % Tune between 0.01 and 0.1 to suppress longitudinal chatter

% Objective: Minimize lap time + spatial smoothing + velocity smoothing
% diff([v; v(1)]) ensures the periodic boundary is also penalized for sudden speed jumps
costFunc = @(X) sum(X(2*N+1 : end)) ...
    + W_smooth * sum(diff([X(1:N); X(1)]).^2) ...
    + W_vel_smooth * sum(diff([X(N+1:2*N); X(N+1)]).^2);

% Nonlinear constraints
nonlincon = @(X) vehicle_constraints(X, N, A_lat, A_long_fwd, A_long_brake, J_lat, J_long, R_min, ref_path);

disp('Executing fmincon solver...');
[X_opt, fval, exitflag, output] = fmincon(costFunc, X0, [], [], [], [], lb, ub, nonlincon, options);

% Recalculate true lap time without the artificial penalties for the output display
true_lap_time = sum(X_opt(2*N+1 : end));
fprintf('True Lap Time: %.3f s\n', true_lap_time);

%% 6. Extract, Calculate Kinematics, and Plot Results
n_opt  = X_opt(1:N);
v_opt  = X_opt(N+1:2*N);
dt_opt = X_opt(2*N+1:end);

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

% Pad with NaN to match the N+1 plot nodes.
% (Boundary derivatives are not strictly constrained in the interior formulation)
hover_data.v  = [v_opt; v_opt(1)];
hover_data.ax = [NaN; a_x_nodes; NaN; NaN];
hover_data.ay = [NaN; a_y_nodes; NaN; NaN];
hover_data.jx = [NaN; NaN; j_x; NaN; NaN];
hover_data.jy = [NaN; NaN; j_y; NaN; NaN];

figure;
plot(ref_path.x, ref_path.y, 'k--', 'DisplayName', 'Centerline'); hold on;
% Store hover_data in the UserData property of the optimal line
p_opt = plot(x_plot, y_plot, 'b-', 'LineWidth', 2, 'DisplayName', 'Optimal Line', 'UserData', hover_data);
scatter(track_pts(:,1), track_pts(:,2), 50, 'r', 'filled', 'DisplayName', 'Track Apexes');

legend; axis equal; grid on;
true_lap_time = sum(dt_opt);
total_distance = sum(sqrt(diff(x_plot).^2 + diff(y_plot).^2));
title(sprintf('Optimal Racing Line (Lap Time: %.3f s | Distance: %.2f m)', true_lap_time, total_distance));
xlabel('X (m)'); ylabel('Y (m)');

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
    data = event_obj.Target.UserData;
    
    if isempty(data)
        txt = {sprintf('X: %.2f', event_obj.Position(1)), ...
               sprintf('Y: %.2f', event_obj.Position(2))};
        return;
    end
    
    idx = event_obj.DataIndex;
    g = 9.81; 
    
    txt = {
        sprintf('Velocity: %.1f km/h', data.v(idx) * 3.6),
        sprintf('A_{long}: %.2f G', data.ax(idx) / g),
        sprintf('A_{lat}: %.2f G', data.ay(idx) / g),
        sprintf('J_{long}: %.2f G/s', data.jx(idx) / g),
        sprintf('J_{lat}: %.2f G/s', data.jy(idx) / g)
    };
end