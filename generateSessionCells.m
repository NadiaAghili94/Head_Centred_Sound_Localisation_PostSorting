function Sessions = generateSessionCells(csvFilePath,bad_block_path)
% generateSessionCells - Reads a CSV with 'block' and 'trace_shape'
% and returns a cell array where each cell contains the frame indices
% for one block/session.

    % Read the CSV file
    % Read the main data table
    data = readtable(csvFilePath, 'PreserveVariableNames', true);
    if ~isempty(bad_block_path) && isfile(bad_block_path)
    % Read the table containing bad blocks
    data_badblock = readtable(bad_block_path);
    % Extract the bad block identifiers
    bad_block = data_badblock.block;
    % Find rows in 'data' that have blocks listed in 'bad_block'
    rowsToRemove = ismember(data.block, bad_block);
    % Remove those rows and store the result in a new variable
    cleaned_data = data(~rowsToRemove, :);
    else
        cleaned_data = data;
    end

    % Extract trace shapes
    traceShapes = cleaned_data.trace_shape;
    numBlocks = height(cleaned_data);

    % Initialize
    Sessions = cell(numBlocks, 1);
    startIndex = 1;

    % Generate session ranges
    for i = 1:numBlocks
        count = traceShapes(i);
        Sessions{i} = startIndex : (startIndex + count - 1);
        startIndex = startIndex + count;
    end
end


