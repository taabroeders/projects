%======================================================================
%            State Dynamics based on Edge Time Series
%======================================================================

%@author: Tommy Broeders
%@email:  t.broeders@amsterdamumc.nl
%updated: 18 04 2024 
%status: done 
%to-do: -

%Review History
%Reviewed by Giuseppe Pontillo

% Description:
% - This code performs k-means clustering on edge time series data to deteremine state dynamics. 
%
% - Prerequisites: The fMRI data needs to be pre-processed
% - Input: Pre-processed fMRI data and a file indicating which regions to include (to account for distortion)
% - Output: State sequence and transition parameters
%-----------------------------------------------------------------------

%% --------------------------------------------------------------------
% Initiate the script
%----------------------------------------------------------------------
clearvars                                                     % Clear all pre-existing variables from the workspace
maxNumCompThreads(20);                                        % Limit the number of threads
cd /XXX                                                       % Set working directory
filelocation='/XXX/*/fmri_timeseries.txt';                    % Location of fMRI timeseries
included_regions=logical(readmatrix('included_regions.txt')); % Load textfile that indicates which regions to include (1=include)
TSlength=200;                                                 % Number of volumes in the fMRI data
distancefunc='cityblock';                                     % Set distance function
addpath('/XXX/GroupICAT/icatb/icatb_helper_functions');       % Used to determine te optimal number of clusters
klist=2:7;                                                    % Search for the optimal K amongst these values (note: k > 1)
output_file='StateDynamics.mat';

%% --------------------------------------------------------------------
% Load timeseries and calculate connectivity
%----------------------------------------------------------------------
disp("Loading files...")
alltimeseries=dir(filelocation);     % List all the locations of the timeseries files
numsubs=length(alltimeseries);       % Count the number of subjects
numregs=sum(included_regions);       % Count the number of included regions
nedges=numregs*(numregs-1)/2;        % calculate number of edges
[u,v]=find(triu(ones(numregs),1));   % Get the indices of unique edges (upper triangle)

part=0;                                     % Initiate iteration variable
reshapedcon=zeros(TSlength*numsubs,nedges); % Initiate connectivity output variable
for TSfile=alltimeseries'                     % Loop over all time series files
    part=part+1;                                        % Update iteration variable
    disp(['  Processing participant: ',num2str(part)])  % Display which participant is being processed
    timeseries=readmatrix(sprintf('%s',[TSfile.folder...% Load the actual time series data
        filesep TSfile.name]));
    timeseries=timeseries(:,included_regions);          % Select only the included regions
    timeseries=zscore(timeseries);                      % Convert the time series of each region to z-scores
    partconnect=timeseries(:,u).*timeseries(:,v);       % Perform point-wise multiplication to compute co-fluctuations
    row_indices=(part-1)*TSlength+1:part*TSlength;      % Determine row indices for reshaped connectivity variable
    reshapedcon(row_indices,:)=partconnect;             % Put connectivity values in the reshaped connectivity variable
                                                        %   rows: timepoints for all participants
                                                        %   columns: connections between all regions
end

%% --------------------------------------------------------------------
% Perform K-means clustering
%----------------------------------------------------------------------
disp('Running K-means clustering for all Ks...')
cluster=cell(length(klist),1);      % Initiate cluster output variable
centroids=cell(length(klist),1);    % Initiate centroid output variable
sumd=cell(length(klist),1);         % Initiate sumd output variable
Dist=cell(length(klist),1);         % Initiate Dist output variable
for ii=klist-1                      % Loop over all values in the klist
    disp(['    ',num2str(ii),'-means:']);                     % Display the K that will be used in the iteration
    rng=22;
    [cluster{ii}, centroids{ii},sumd{ii},Dist{ii}]=kmeans(... % Perform K-means clustering
        reshapedcon,...                                       % Input the reshaped connectivity variable
        ii+1,...
        'MaxIter',150,...                                     % Set the maximum number of iterations
        'Display','final',...                                 % Display the results of the final iteration
        'replicates',5,...                                    % Run K-means clustering 5 times to avoid local minima
        'Distance',distancefunc);                             % Use the preselected distance function
end

%% --------------------------------------------------------------------
% Determine optimal number of clusters
%----------------------------------------------------------------------
disp('Determining the optimal number of clusters...')

R=zeros(1,length(klist));  % Initiate the dispersion ratio output variable
for k2=1:length(klist)     % Loop over all values in the klist
    [~, R(k2)]=cluster_goodness(Dist{k2}, cluster{k2});     % Determine cluster fit using the dispersion ratio
end
[OptimalK, yfit]=fit_L_tocurve_area(klist,R);  % Fit an L-curve to determine the optimal value for K
K_i=find(klist==OptimalK);                     % Set another K variable for indexing

save(output_file,'-v7.3')

%% --------------------------------------------------------------------
% Calculate state transition parameters
% ---------------------------------------------------------------------
disp('Computing state parameters...')
aIND=reshape(cluster{K_i},...   % Put the final clustering in a matrix 0(timepoints X subjects) 
    TSlength, numsubs);
aFR=zeros(numsubs, klist(K_i)); % Initiate the fractional occupancy variable
aTM=zeros(numsubs,...           % Initiate the transition matrix variable
    OptimalK, OptimalK);
aMDT=zeros(numsubs, klist(K_i));% Initiate the mean dwell time variable
aNT=zeros(numsubs, 1);          % Initiate the total transitions variable
for part=1:numsubs                % Loop over all subjects
    disp(['  Processing participant: ',num2str(part)])         % Display which participant is being processed
    [aFR(part,:),aTM(part,:,:),aMDT(part, :),aNT(part)]=...
        icatb_dfnc_statevector_stats(aIND(:,part), OptimalK);  % Compute the state parameters per participant
end

% Exclude values (e.g. dwell-time and fraction of transitions) if a state is not occupied
aMDT(aFR==0)=NaN;                       % Exclude dwell time values if the occupancy is equal to zero
noFR=repmat(aFR==0,[1 1 OptimalK]);     % Find all transitions that initiate from a state that was not occupied
aTM2=aTM;aTM2(noFR==1)=NaN;             % Set all these transitions to NaN

% To extract all transitions from and to a particular state, you can use the following code:
%   TfromS1=squeeze(aTM2(:,1,:)); TtoS1=squeeze(aTM2(:,:,1));
%   TfromS2=squeeze(aTM2(:,2,:)); TtoS2=squeeze(aTM2(:,:,2));
%   etc.

save(output_file,'-v7.3')

%% --------------------------------------------------------------------
%                       References
%----------------------------------------------------------------------
% https://github.com/trendscenter/gift (GroupICATv4.0b)
