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

function plot_all_2d(data, data_found, objective_names)
    % Extract Composition Scores from the data
    composition_scores = data(:, 4:end);
    composition_scores_found = data_found;
    
    % Find Pareto points
    [~, pareto_points] = find_pareto_frontier(composition_scores);

    % Generate all possible combinations of 2 objectives
    combinations = nchoosek(1:size(objective_names, 2), 2);

    % Plot each combination in a separate figure
    for i = 1:size(combinations, 1)
        obj1 = combinations(i, 1);
        obj2 = combinations(i, 2);
        
        % Open a new figure
        figure;
        
        % Plot all data points in blue
        scatter(composition_scores(:, obj1), composition_scores(:, obj2), 'Marker', '.', 'MarkerEdgeColor', 'b');
        hold on;
        
        % Plot found data points in purple
        scatter(composition_scores_found(:, obj1), composition_scores_found(:, obj2), 'm', 'filled');
        %hold on;

        % Plot Pareto points in red
        scatter(pareto_points(:, obj1), pareto_points(:, obj2), 'r', 'filled');
        
        % Find common points between found data and Pareto front
        common_points = intersect(composition_scores_found(:, [obj1 obj2]), pareto_points(:, [obj1 obj2]), 'rows');
        % Plot common points in green
        scatter(common_points(:, 1), common_points(:, 2), 'g', 'filled');
        
        % Customize plot
        xlabel(objective_names{obj1});
        ylabel(objective_names{obj2});
        title(sprintf('2D Plot: %s vs %s', objective_names{obj1}, objective_names{obj2}));
        % Set legend
        legend({'All Data', 'Pareto Points', 'Data Found', 'Common Points'}, ...
           'Location', 'northwest', 'FontSize', 8); % Adjust FontSize as needed
        grid on;
        hold off;
    end
end

% Load data from Excel file (replace 'your_excel_file.xlsx' with your actual file name)
data = xlsread('labled_composition_scores.xlsx');
data_found = xlsread('found_pareto.xlsx');

% Specify the names of objectives
objective_names = {'Energy', 'Response Time', 'Nb Clouds'};

% Plot all possible 2D graphs
plot_all_2d(data, data_found, objective_names);
