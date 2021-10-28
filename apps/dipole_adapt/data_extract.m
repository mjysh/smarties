function data = data_extract(dir)
raw = readmatrix(dir);

% nEp = length(raw);
% epiLen = raw(:,4);
% reward = raw(:,5);

data = struct('n_episode',length(raw),'length',raw(:,4),'reward',raw(:,5));
end