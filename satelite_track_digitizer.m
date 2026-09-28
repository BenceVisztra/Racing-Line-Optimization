% Two-Boundary Track Digitizer
img = imread('satelite_kistarcsa.png');
ruler_on_map = 50.05;
filename = 'track_kistarcsa.csv';


figure('Name', 'Boundary Digitizer', 'Position', [100, 100, 1200, 800]);
imshow(img); hold on;

% 1. Scale
title('1/3: Draw a line over the scale defining ruler (Double-click last point)');
h_scale = drawpolyline('Color', 'y', 'LineWidth', 2);
pixel_dist = norm(h_scale.Position(2,:) - h_scale.Position(1,:));
scale = ruler_on_map / pixel_dist; 

% 2. Left Boundary
title('2/3: Draw the LEFT/INNER boundary (Double-click last point)');
h_left = drawpolyline('Color', 'b', 'LineWidth', 2);

% 3. Right Boundary
title('3/3: Draw the RIGHT/OUTER boundary (Double-click last point)');
h_right = drawpolyline('Color', 'r', 'LineWidth', 2);

title('Processing centerline and widths...');
drawnow;

% Get raw coordinates
lx_raw = h_left.Position(:,1);
ly_raw = h_left.Position(:,2);
rx_raw = h_right.Position(:,1);
ry_raw = h_right.Position(:,2);

% Calculate cumulative distance
dist_l_raw = [0; cumsum(sqrt(diff(lx_raw).^2 + diff(ly_raw).^2))];
dist_r_raw = [0; cumsum(sqrt(diff(rx_raw).^2 + diff(ry_raw).^2))];

% Filter unique distances
[dist_l, idx_l] = unique(dist_l_raw, 'stable');
[dist_r, idx_r] = unique(dist_r_raw, 'stable');
lx_clean = lx_raw(idx_l); ly_clean = ly_raw(idx_l);
rx_clean = rx_raw(idx_r); ry_clean = ry_raw(idx_r);

% Densify both boundaries
N_dense = 2000;
s_l = linspace(0, dist_l(end), N_dense);
s_r = linspace(0, dist_r(end), N_dense);
lx_dense = interp1(dist_l, lx_clean, s_l, 'makima');
ly_dense = interp1(dist_l, ly_clean, s_l, 'makima');
rx_dense = interp1(dist_r, rx_clean, s_r, 'makima');
ry_dense = interp1(dist_r, ry_clean, s_r, 'makima');

% Geometrically synchronize boundaries
cx_raw = zeros(1, N_dense);
cy_raw = zeros(1, N_dense);
for i = 1:N_dense
    dists_sq = (rx_dense - lx_dense(i)).^2 + (ry_dense - ly_dense(i)).^2;
    [~, min_idx] = min(dists_sq);
    cx_raw(i) = (lx_dense(i) + rx_dense(min_idx)) / 2;
    cy_raw(i) = (ly_dense(i) + ry_dense(min_idx)) / 2;
end

% Resample centerline to ensure even node spacing
dist_c_raw = [0, cumsum(sqrt(diff(cx_raw).^2 + diff(cy_raw).^2))];
[dist_c, idx_c] = unique(dist_c_raw, 'stable');
cx_uniq = cx_raw(idx_c);
cy_uniq = cy_raw(idx_c);

N_final = 500;
s_c = linspace(0, dist_c(end), N_final);
cx_interp = interp1(dist_c, cx_uniq, s_c, 'makima');
cy_interp = interp1(dist_c, cy_uniq, s_c, 'makima');

% Calculate exact track widths geometrically from the new centerline
W_left = zeros(1, N_final);
W_right = zeros(1, N_final);
for i = 1:N_final
    d_left = sqrt((lx_dense - cx_interp(i)).^2 + (ly_dense - cy_interp(i)).^2);
    d_right = sqrt((rx_dense - cx_interp(i)).^2 + (ry_dense - cy_interp(i)).^2);
    W_left(i) = min(d_left) * scale;
    W_right(i) = min(d_right) * scale;
end

% Normalize to origin, scale to meters, invert Y for Cartesian
cx_m = (cx_interp - cx_interp(1)) * scale;
cy_m = -(cy_interp - cy_interp(1)) * scale;

% Plot single centerline to verify visually
plot(cx_interp, cy_interp, 'w--', 'LineWidth', 2);

% Print output array
fprintf('Paste this into track_selection:\n');
fprintf('pts = [\n');
for i = 1:length(cx_m)
    fprintf('    %.2f, %.2f, %.2f, %.2f;\n', cx_m(i), cy_m(i), W_left(i), W_right(i));
end
fprintf('];\n');
title('Done! Centerline verified (white dashed). Output printed to Command Window.');

% Combine into an Nx4 matrix and save to CSV
track_matrix = [cx_m(:), cy_m(:), W_left(:), W_right(:)];
writematrix(track_matrix, filename);

title(sprintf('Done! Track saved to %s', filename));