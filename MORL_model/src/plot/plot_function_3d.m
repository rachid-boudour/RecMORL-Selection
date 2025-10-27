function [membership, member_value] = find_pareto_frontier(input)
    out = [];
    data = unique(input, 'rows');
    for i = 1:size(data, 1)
        c_data = repmat(data(i, :), size(data, 1), 1);
        t_data = data;
        t_data(i, :) = -Inf(1, size(data, 2));  % Change to negative infinity
        larger_idx = c_data <= t_data;  % Change comparison to <=
        idx = sum(larger_idx, 2) == size(data, 2);
        if ~nnz(idx)
            out(end+1, :) = data(i, :);
        end
    end
    membership = ismember(input, out, 'rows');
    member_value = out;
end

function plot_3d(data, data_found)
    % Extract Cloud IDs and Composition Scores from the data
    cloud_ids = data(:, 1:3);
    composition_scores = data(:, 4:end);

    % Find Pareto points
    [membership, member_value] = find_pareto_frontier(composition_scores);
    pareto_indices = membership;
    pareto_points = composition_scores(pareto_indices, :);
    pareto_cloud_ids = cloud_ids(pareto_indices, :);

    % Find common points between Pareto points and data_found
    common_indices = ismember(pareto_points, data_found, 'rows');
    common_points = pareto_points(common_indices, :);

    % Open a new figure
    figure;

    % Plot all data points in blue (dots)
    scatter3(composition_scores(:, 1), composition_scores(:, 2), composition_scores(:, 3), ...
             'Marker', '.', 'MarkerEdgeColor', 'b');
    hold on;

    % Plot Pareto points in red
    scatter3(pareto_points(:, 1), pareto_points(:, 2), pareto_points(:, 3), 'r', 'filled');
    
    % Plot data_found points in purple
    scatter3(data_found(:, 1), data_found(:, 2), data_found(:, 3), 'm', 'filled');

    % Plot common points in green
    scatter3(common_points(:, 1), common_points(:, 2), common_points(:, 3), 'g', 'filled');

    % Add labels and data cursor
    hDcm = datacursormode(gcf);
    set(hDcm, 'UpdateFcn', @(obj, event_obj) custom_datacursor(obj, event_obj, pareto_cloud_ids, composition_scores));

    % Customize plot
    xlabel('Energy');
    ylabel('Response time');
    zlabel('Nb clouds');
    title('True/Found Pareto Points');
    grid on;

    % Set legend
    legend({'All Data', 'Pareto Points', 'Data Found', 'Common Points'}, ...
           'Location', 'northwest', 'FontSize', 8); % Adjust FontSize as needed

    hold off;

    function output_txt = custom_datacursor(obj, event_obj, cloud_ids, composition_scores)
        pos = get(event_obj, 'Position');  % Get data cursor position
        idx = get(event_obj, 'DataIndex');  % Get data cursor index

        % Check if index is within range
        if idx < 1 || idx > size(cloud_ids, 1)
            disp('Index out of range');
            output_txt = {};
            return;
        end

        % Get Cloud IDs corresponding to the index
        cloud_id_str = sprintf('Cloud IDs: %d, %d, %d\n', cloud_ids(idx, :));

        % Create strings for composition scores
        composition_score_str = sprintf('X: %.4f, Y: %.4f, Z: %.4f', pos(1), pos(2), pos(3));

        % Display position and label
        output_txt = {cloud_id_str, composition_score_str};
    end
end

% Load data from Excel files (replace file names with your actual file names)
data = xlsread('labled_composition_scores.xlsx');
data_found = xlsread('found_pareto.xlsx');

% Plot 3D figure
plot_3d(data, data_found);
