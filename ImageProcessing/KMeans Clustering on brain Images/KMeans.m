% Do not change the function name. 
% you have to submit this function
function segmentedImage = KMeans(featureImageIn, numberofClusters, clusterCentersIn)

% Get the dimensions of the input feature image
[M, N, noF] = size(featureImageIn);
% some initialization
% if no clusterCentersIn or it is empty, randomize the clusterCentersIn
% and run kmeans several times and keep the best one
if (nargin == 3) && (~isempty(clusterCentersIn))
    NofRadomization = 1;
else
    NofRadomization = 5;    % Should be greater than one
end


segmentedImage = zeros(M, N); %initialize. This will be the output

BestDfit = 1e10;  % Just a big number!

% run KMeans NofRadomization times
for KMeanNo = 1 : NofRadomization

    % randomize if clusterCentersIn was empty
    if NofRadomization>1
        clusterCentersIn = rand(numberofClusters, noF); %randomize initialization
    end

% ----------------------------------- % 
% -You have to write your code here-- %
% ----------------------------------- % 