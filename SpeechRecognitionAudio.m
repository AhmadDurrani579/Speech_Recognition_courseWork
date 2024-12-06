close all; clear all; clc;

% sp16a_w07_hod.mp3
% Define training folder (organized by subfolders for each class)
trainFolder = 'train/';
% Define parameters for MFCC extraction
frameDuration = 0.025;  % 30 ms frame length
hopDuration = 0.015;    % 10 ms hop size
numCoefficients = 13;  % Number of MFCC coefficients (including zeroth)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aggregate MFCCs for All Training Files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Aggregating MFCCs from training data...');
subFolders = dir(trainFolder);

% Initialize variables for global mean and variance calculation
allMFCCs = [];
vocabulary = {};  % Initialize vocabulary array

% Loop through subfolders to process each class
for k = 1:length(subFolders)
    % Check if the current item is a directory and not '.' or '..'
    if subFolders(k).isdir && ~ismember(subFolders(k).name, {'.', '..'})
        % Get the class name (subfolder name)
        className = subFolders(k).name;
        
        % Add the class name to the vocabulary list
        vocabulary{end+1} = className;  % Add class name to vocabulary
        
        % Retrieve all audio files with '.mp3' extension in the current class folder
        audioFiles = dir(fullfile(trainFolder, className, '*.mp3'));
        
        % Display the class being processed and the number of audio files found
        disp(['Processing class: ', className, ' (', num2str(length(audioFiles)), ' files)']);
        
        % Loop through each audio file to process and extract features
        for i = 1:length(audioFiles)
            % Construct the full file path for the current audio file
            filePath = fullfile(audioFiles(i).folder, audioFiles(i).name);
            
            % Read the audio file and get its sampling frequency
            [audio, fs] = audioread(filePath);
            
            % Extract Mel-Frequency Cepstral Coefficients (MFCCs) from the audio
            mfcc = extractMFCCs(audio, fs);
            
            % Append the extracted MFCCs to the collection for global statistics
            if ~isempty(mfcc)
                allMFCCs = [allMFCCs; mfcc];  % Aggregate MFCC features
            end
        end
    end
end

% Calculate global mean and variance for the extracted MFCC features
disp('Calculating global mean and variance...');
if ~isempty(allMFCCs)
    % Compute the mean of the MFCCs across all audio files
    globalMean = mean(allMFCCs, 1);
    
    % Compute the variance of the MFCCs across all audio files
    globalVariance = var(allMFCCs, 0, 1);
    
    % Uncomment the following line to ensure variance is not extremely small,
    % which could cause numerical issues in further processing
    % globalVariance = max(globalVariance, 1e-6);  % Avoid very small variances
else
    % Display an error if no MFCCs were extracted, indicating a problem
    error('No MFCCs were extracted from the training files.');
end

disp(['Global mean size: ', num2str(size(globalMean))]);
disp(['Global variance size: ', num2str(size(globalVariance))]);

% Plot marginal distributions of each MFCC feature
figure;  % Create a new figure for the plots
x = linspace(-10, 10, 1000); % Define the range of x-axis values for plotting the Gaussian PDF
hold on; % Hold the plot so multiple curves can be overlaid

% Loop through each MFCC feature to calculate and plot its Gaussian PDF
for i = 1:numCoefficients
    % Extract the mean and standard deviation for the current MFCC feature
    mu = globalMean(i); % Mean for the current MFCC coefficient
    sigma = sqrt(globalVariance(i)); % Standard deviation for the current coefficient
    
    % Compute the Gaussian Probability Density Function (PDF)
    y = normpdf(x, mu, sigma); % Gaussian PDF using the mean and standard deviation
    
    % Plot the Gaussian distribution with a label for the legend
    plot(x, y, 'DisplayName', ['MFCC ', num2str(i), ...
        ' (mean=', num2str(mu, '%.2f'), ', var=', num2str(globalVariance(i), '%.2f'), ')']);
end

hold off; % Release the hold so further plots do not overlay
xlabel('Feature Value'); % Label for the x-axis
ylabel('Probability Density'); % Label for the y-axis
title('Marginal Distributions of 13 MFCC Features'); % Title of the plot
legend show; % Show the legend to identify each feature's distribution
grid on; % Add a grid for better visualization



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize HMMs for Each Class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Initializing HMMs for each class...');
numStates = 8;  % Number of states per class
HMM = struct();

for word = 1:length(vocabulary)
    % Set a scaling factor to introduce flexibility in the transition matrix
    scalingFactor = 0.05; % Adjust as needed to balance strictness and flexibility

    % Initialize the transition matrix with a strict left-to-right topology
    % Main diagonal (self-transitions) initialized to 0.8
    % Upper diagonal (next state transitions) initialized to 0.2
    A = diag(0.8 * ones(1, numStates)) + diag(0.2 * ones(1, numStates-1), 1);
    A(end, end) = 1;  % Make the last state an absorbing state (cannot transition forward)

    % Apply the scaling factor to the transition matrix for flexibility
    A = (1 - scalingFactor) * A + scalingFactor * eye(numStates); % Blend with identity matrix
    A = A ./ sum(A, 2); % Normalize rows to ensure valid probabilities
    HMM(word).A = A; % Store the transition matrix in the HMM structure

    % Initialize the mean vectors for each state
    % Start with the global mean and add controlled noise for state-dependency
    HMM(word).means = repmat(globalMean, numStates, 1) + ...
                      randn(numStates, numCoefficients) * 0.01;

    % Initialize the covariance matrices for each state
    % Diagonal matrices based on the global variance with small random noise added
    globalVarianceWithNoise = globalVariance .* (1 + rand(size(globalVariance)) * 0.01);  % Add small noise
    HMM(word).covariances = repmat(diag(globalVarianceWithNoise), 1, 1, numStates);

    % Assign the word name to the HMM for clarity and identification
    HMM(word).name = vocabulary{word};
end

disp('HMMs initialized successfully.');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train HMMs with Baum-Welch Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%maxEpochs = 12; % Set maximum training epochs

% Define the number of training epochs
maxEpochs = 1;  % Set epochs to meet coursework requirements

% Preallocate space for error rates and log-likelihood tracking
errorRates = NaN(1, maxEpochs);  % Preallocate with NaN values
logLikelihoodsForWord = cell(length(vocabulary), maxEpochs); % Preallocate cell array for log-likelihoods

% Training Loop for Baum-Welch Algorithm
for epoch = 1:maxEpochs
    disp(['Starting epoch ', num2str(epoch)]);  % Display current epoch

    % Initialize the total log-likelihood for this epoch
    totalLogLikelihood = 0;

    % Preallocate updates for all HMMs for this epoch
    wordUpdates = cell(1, length(vocabulary));

    % Loop through each word/class in the vocabulary
    for word = 1:length(vocabulary)
        % Load training data for the current word
        wordTrainingData = loadTrainingDataForWord(vocabulary{word}, trainFolder);
        logLikelihoodsForWord{word, epoch} = totalLogLikelihood; % Track log-likelihood for this word

        % Initialize accumulators for HMM parameter updates
        localUpdatedMeans = zeros(size(HMM(word).means));
        localUpdatedCovariances = zeros(size(HMM(word).covariances));
        localUpdatedTransitions = zeros(size(HMM(word).A));

        % Loop through each training sequence for the current word
        for sequenceIdx = 1:length(wordTrainingData)
            % Get the observation sequence for the current training sample
            observationSequence = wordTrainingData{sequenceIdx};

            % Run the Forward-Backward algorithm
            [alpha, beta, gamma, xi] = forwardBackward(HMM(word), observationSequence);

            % Calculate the log-likelihood for the current sequence
            logLikelihood = logsumexp(alpha(:, end)); % Combine forward probabilities
            totalLogLikelihood = totalLogLikelihood + logLikelihood;

            % Calculate expected parameter updates using gamma and xi
            [newMeans, newCovariances, newTransitions] = calculateExpectedUpdates(gamma, xi, observationSequence);

            % Accumulate updates for all training sequences for this word
            localUpdatedMeans = localUpdatedMeans + newMeans;
            localUpdatedCovariances = localUpdatedCovariances + newCovariances;
            localUpdatedTransitions = localUpdatedTransitions + newTransitions;
        end

        % Normalize updates by dividing by the number of sequences
        numSequences = length(wordTrainingData);
        wordUpdates{word}.means = localUpdatedMeans / numSequences;
        wordUpdates{word}.covariances = localUpdatedCovariances / numSequences;
        wordUpdates{word}.transitions = normalizeTransitions(localUpdatedTransitions); % Ensure valid transition probabilities
    end

    % Apply accumulated updates to HMM parameters for all words
    for word = 1:length(vocabulary)
        HMM(word).means = wordUpdates{word}.means;
        HMM(word).covariances = wordUpdates{word}.covariances;
        HMM(word).A = wordUpdates{word}.transitions;
    end

    % Evaluate the model on the training set to calculate the error rate
    disp('Evaluating the model...');
    predictedLabels = {}; % Store predicted labels
    trueLabels = {}; % Store true labels

    % Loop through each word to predict its labels
    for word = 1:length(vocabulary)
        % Get all audio files for the current word
        audioFiles = dir(fullfile(trainFolder, vocabulary{word}, '*.mp3'));
        for i = 1:length(audioFiles)
            % Get the full path for the current audio file
            testAudioPath = fullfile(audioFiles(i).folder, audioFiles(i).name);
            
            % Recognize the word using the trained HMMs
            predictedWord = recognizeWord(testAudioPath, HMM, vocabulary);
            
            % Store true and predicted labels
            trueLabels{end+1} = vocabulary{word};
            predictedLabels{end+1} = predictedWord;
        end
    end

    % Compute confusion matrix and error rate
    trueLabels = categorical(trueLabels); % Convert to categorical for compatibility
    predictedLabels = categorical(predictedLabels);
    confMat = confusionmat(trueLabels, predictedLabels); % Calculate confusion matrix
    accuracy = sum(diag(confMat)) / sum(confMat(:)); % Calculate accuracy
    errorRate = 1 - accuracy; % Calculate error rate
    errorRates(epoch) = errorRate; % Store the error rate for this epoch

    % Display metrics for the current epoch
    disp(['Epoch ', num2str(epoch), ' Total Log-Likelihood: ', num2str(totalLogLikelihood)]);
    disp(['Epoch ', num2str(epoch), ' completed. Error rate: ', num2str(errorRate)]);
end

% Plot error rates
figure;
plot(1:maxEpochs, errorRates, '-o');
xlabel('Epoch');
ylabel('Error Rate');
title('Error Rate During Training');
grid on;

% Number of words in the vocabulary
numWords = length(vocabulary);

% Calculate grid size for subplots
numCols = ceil(sqrt(numWords)); % Number of columns
numRows = ceil(numWords / numCols); % Number of rows

% Create a figure for the grid
figure;

% Iterate over each word
for wordIdx = 1:numWords
    % Extract log-likelihoods for the current word
    logLikelihoods = cell2mat(logLikelihoodsForWord(wordIdx, :));
    
    % Calculate total improvement
    improvement = logLikelihoods(end) - logLikelihoods(1);
    
    % Create a subplot for the current word
    subplot(numRows, numCols, wordIdx);
    plot(1:maxEpochs, logLikelihoods, '-o', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Log-Likelihood');
    title(vocabulary{wordIdx}, 'Interpreter', 'none');
    grid on;
    
    % Add the improvement as text on the plot
    text(0.1, 0.9, sprintf('Improvement: %.2f', improvement), ...
        'Units', 'normalized', 'FontSize', 8, 'Color', 'k');
end

% Set the overall figure title
sgtitle('Log-Likelihood During Training for All Words');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate and Plot the Confusion Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize arrays for true labels and predicted labels
trueLabels = {};          % Cell array for true labels
predictedLabels = {};     % Cell array for predicted labels

% Define the test folder path
testFolder = '/Users/ahmad/Desktop/Speech_Recognition/Speech_Recognition_courseWork/train';
subFolders = dir(testFolder);

% Loop through each subfolder
for k = 1:length(subFolders)
    if subFolders(k).isdir && ~ismember(subFolders(k).name, {'.', '..'})
        % Get the current word (subfolder name) as the true label
        trueLabel = subFolders(k).name;
        
        % Get all audio files in the current subfolder
        audioFiles = dir(fullfile(testFolder, trueLabel, '*.mp3'));
        numTestFiles = min(30, length(audioFiles)); % Limit to 30 files or fewer

        % Loop through each audio file in the subfolder
        for i = 1:numTestFiles
            % Get the test audio file path
            testAudioPath = fullfile(testFolder, trueLabel, audioFiles(i).name);

            % Run the word recognition process on the test file
            predictedWord = recognizeWord(testAudioPath, HMM, vocabulary);
            
            % Append the true and predicted labels as strings
            trueLabels{end+1} = trueLabel;
            predictedLabels{end+1} = predictedWord;
        end
    end
end

% Ensure labels are categorical
trueLabels = categorical(trueLabels);
predictedLabels = categorical(predictedLabels);

% Generate the confusion matrix
confMat = confusionmat(trueLabels, predictedLabels);

% Display the confusion matrix in the console
disp('Confusion Matrix:');
disp(confMat);

% Plot the confusion matrix as a heatmap
figure;
cm = confusionchart(confMat, categories(trueLabels), ... % Use categories from trueLabels
    'Title', 'Customized Confusion Matrix', ...
    'XLabel', 'Predicted Labels', ...
    'YLabel', 'True Labels');

% Customize appearance
cm.FontSize = 14;
cm.FontName = 'Calibri';
colormap(cm.Parent, 'spring'); % Correctly set colormap using Parent property
cm.GridVisible = 'on';
cm.CellLabelFormat = '%0.1f%%'; % Show percentages with one decimal place

% Change diagonal and off-diagonal cell colors
cm.DiagonalColor = [0 0.7 0.3];
cm.OffDiagonalColor = [0.9 0.1 0.1];

% Normalize rows and columns
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'absolute';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function for extracting MFCCs from audio
function mfcc = extractMFCCs(audio, fs)
    % Set parameters
    frameDuration = 0.03;  % 30 ms frame length
    hopDuration = 0.01;    % 10 ms hop size
    numCoefficients = 13;  % Include the zeroth coefficient
    NFFT = 1024;            % FFT length

    % Add preprocessing
    audio = audio - mean(audio);  % Remove DC offset
    audio = audio / max(abs(audio)); % Normalize
    
    % Add pre-emphasis
    preemph = [1 -0.97];
    audio = filter(preemph, 1, audio);

    % Create audio feature extractor
    extractor = audioFeatureExtractor('SampleRate', fs, ...
        'Window', hamming(round(frameDuration * fs), 'periodic'), ...
        'OverlapLength', round(hopDuration * fs), ...
        'FFTLength', NFFT, ...
        'mfcc', true);

    % Extract features
    features = extract(extractor, audio);
    mfcc = features(:, 1:numCoefficients);  % Include the zeroth coefficient

    % Add mean and variance normalization
    mfcc = mfcc - mean(mfcc);
    mfcc = mfcc ./ (std(mfcc) + eps);
end


% Function for loading training data for a specific word
function wordTrainingData = loadTrainingDataForWord(word, trainFolder)
    wordTrainingData = {};  % Initialize empty cell array for sequences

    % Get all MP3 files for the given word
    wordFolder = fullfile(trainFolder, word);
    audioFiles = dir(fullfile(wordFolder, '*.mp3'));

    for i = 1:length(audioFiles)
        % Load the audio file for the current training example
        audioPath = fullfile(audioFiles(i).folder, audioFiles(i).name);
        [audio, fs] = audioread(audioPath);

        % Extract MFCCs for the current file
        mfcc = extractMFCCs(audio, fs);

        % Check if MFCCs are valid (non-empty)
        if ~isempty(mfcc)
            wordTrainingData{end + 1} = mfcc;  % Add MFCC sequence to cell array
        else
            disp(['Warning: MFCCs are empty for file ', audioPath]);
        end
    end
end

% Function to normalize transition matrix
function normalizedA = normalizeTransitions(A)
    rowSums = sum(A, 2);
    normalizedA = A ./ rowSums;  % Normalize each row to sum to 1
end

% Forward-Backward Algorithm
% Forward-Backward Algorithm with Gaussian Modeling Logic
function [alpha, beta, gamma, xi] = forwardBackward(HMM, observationSequence)
    epsilon = 1e-6; % Regularization constant for numerical stability
    numStates = size(HMM.A, 1); % Number of states
    numObservations = size(observationSequence, 1); % Number of observations

    % Initialize forward (alpha) and backward (beta) variables
    alpha = -Inf(numStates, numObservations); % Log-space forward probabilities
    beta = -Inf(numStates, numObservations);  % Log-space backward probabilities

    % =============================
    % Forward Recursion (Alpha)
    % =============================
    % Initialization step (t = 1)
    for state = 1:numStates
        % Regularize covariance matrix
        covMatrix = HMM.covariances(:, :, state) + epsilon * eye(size(HMM.covariances, 1));
        
        % Compute the Gaussian probability density for the first observation
        probDensity = gaussianProbability(observationSequence(1, :)', HMM.means(state, :)', covMatrix);
        alpha(state, 1) = log(max(probDensity, epsilon)) + log(1 / numStates); % Avoid log(0)
    end

    % Recursion step (t = 2 to T)
    for t = 2:numObservations
        for state = 1:numStates
            % Regularize covariance matrix
            covMatrix = HMM.covariances(:, :, state) + epsilon * eye(size(HMM.covariances, 1));
            
            % Compute the Gaussian probability density for the current observation
            probDensity = gaussianProbability(observationSequence(t, :)', HMM.means(state, :)', covMatrix);

            % Update alpha using previous states and transition probabilities
            alpha(state, t) = log(max(probDensity, epsilon)) + ...
                              logsumexp(alpha(:, t-1) + log(max(HMM.A(:, state), epsilon))); % Avoid log(0)
        end
    end

    % =============================
    % Backward Recursion (Beta)
    % =============================
    % Initialization step (t = T)
    beta(:, end) = 0; % Log(1) = 0 for backward initialization

    % Recursion step (t = T-1 to 1)
    for t = numObservations-1:-1:1
        for state = 1:numStates
            % Regularize covariance matrix
            covMatrix = HMM.covariances(:, :, state) + epsilon * eye(size(HMM.covariances, 1));
            
            % Compute the Gaussian probability density for the next observation
            probDensity = gaussianProbability(observationSequence(t + 1, :)', HMM.means(state, :)', covMatrix);

            % Update beta using next states and transition probabilities
            beta(state, t) = logsumexp(log(max(HMM.A(state, :), epsilon)) + ...
                                       log(max(probDensity, epsilon)) + beta(:, t+1)');
        end
    end

    % =============================
    % Compute Gamma (State Occupancy Probabilities)
    % =============================
    gamma = exp(alpha + beta - logsumexp(alpha(:, end))); % Normalize gamma

    % =============================
    % Compute Xi (State Transition Probabilities)
    % =============================
    xi = zeros(numStates, numStates, numObservations-1); % Preallocate xi
    for t = 1:numObservations-1
        for i = 1:numStates
            for j = 1:numStates
                % Regularize covariance matrix
                covMatrix = HMM.covariances(:, :, j) + epsilon * eye(size(HMM.covariances, 1));
                
                % Compute the Gaussian probability density for the next observation
                probDensity = gaussianProbability(observationSequence(t + 1, :)', HMM.means(j, :)', covMatrix);

                % Update xi using alpha, transition probabilities, and beta
                xi(i, j, t) = exp(alpha(i, t) + log(max(HMM.A(i, j), epsilon)) + ...
                                log(max(probDensity, epsilon)) + beta(j, t+1));
            end
        end
        xi(:, :, t) = xi(:, :, t) / sum(xi(:, :, t), 'all'); % Normalize xi
    end
end


function probDensity = probabilityDensity(observation, meanVector, covMatrix)
    % Replace existing computation with a call to gaussianProbability
    probDensity = gaussianProbability(observation(:), meanVector(:), covMatrix);
end

function logLikelihood = viterbi(observations, HMM)
    % VITERBI Implements the Viterbi algorithm with Gaussian modeling for HMM sequence likelihood estimation.
    %
    % Inputs:
    %   observations - Matrix of MFCC features for the test audio (numFrames x numCoefficients).
    %   HMM - Hidden Markov Model structure for a single word.
    %
    % Outputs:
    %   logLikelihood - Log-likelihood of the observation sequence given the HMM.

    epsilon = 1e-6;  % Small value for numerical stability
    numStates = size(HMM.A, 1);  % Number of states in the HMM
    numObservations = size(observations, 1);  % Number of frames in the observation sequence

    % Initialize log-probability matrices
    V = -Inf(numStates, numObservations);  % Log-probability matrix
    backtrack = zeros(numStates, numObservations);  % To store back-pointers

    % =============================
    % Initialization: Compute initial state probabilities
    % =============================
    for state = 1:numStates
        % Regularize covariance matrix
        covMatrix = HMM.covariances(:, :, state) + epsilon * eye(size(HMM.covariances, 1));
        
        % Compute Gaussian probability density for the first observation
        probDensity = gaussianProbability(observations(1, :)', HMM.means(state, :)', covMatrix);

        % Compute log probability for the first observation in the state
        V(state, 1) = log(max(probDensity, epsilon)) + log(1 / numStates); % Avoid log(0)
    end

    % =============================
    % Recursion: Compute probabilities for remaining time steps
    % =============================
    for t = 2:numObservations
        for state = 1:numStates
            % Regularize covariance matrix
            covMatrix = HMM.covariances(:, :, state) + epsilon * eye(size(HMM.covariances, 1));
            
            % Compute Gaussian probability density for the current observation
            probDensity = gaussianProbability(observations(t, :)', HMM.means(state, :)', covMatrix);

            % Find the maximum probability path leading to this state
            [maxVal, prevState] = max(V(:, t-1) + log(max(HMM.A(:, state), epsilon))); % Avoid log(0)
            V(state, t) = log(max(probDensity, epsilon)) + maxVal;
            backtrack(state, t) = prevState;
        end
    end

    % =============================
    % Termination: Compute the final log-likelihood
    % =============================
    [logLikelihood, ~] = max(V(:, end));

    % Handle cases where log-likelihood becomes NaN or Inf
    if isnan(logLikelihood) || isinf(logLikelihood)
        logLikelihood = -1e10; % Assign a large negative value
    end
end
% Log-sum-exp function for numerical stability
function result = logsumexp(x)
    maxVal = max(x);
    result = maxVal + log(sum(exp(x - maxVal)));
end

function [updatedMeans, updatedCovariances, updatedTransitions] = calculateExpectedUpdates(gamma, xi, observationSequence)
    % CALCULATEEXPECTEDUPDATES Updates HMM parameters with Gaussian modeling.

    numStates = size(gamma, 1);  % Number of states
    numObservations = size(observationSequence, 1);  % Number of observations
    numCoefficients = size(observationSequence, 2);  % Number of MFCC coefficients

    % Initialize accumulators
    updatedMeans = zeros(numStates, numCoefficients);
    updatedCovariances = zeros(numCoefficients, numCoefficients, numStates);
    updatedTransitions = zeros(numStates, numStates);

    % =============================
    % Update means
    % =============================
    for i = 1:numStates
        gammaSum = sum(gamma(i, :));  % Sum of responsibilities for state i
        if gammaSum > 0
            updatedMeans(i, :) = sum(gamma(i, :)' .* observationSequence, 1) / gammaSum;
        end
    end

    % =============================
    % Update covariances
    % =============================
    for i = 1:numStates
        gammaSum = sum(gamma(i, :));  % Sum of responsibilities for state i
        if gammaSum > 0
            diff = observationSequence - updatedMeans(i, :);  % Difference from mean
            weightedDiff = diff .* sqrt(gamma(i, :)');  % Weight differences by gamma
            updatedCovariances(:, :, i) = (weightedDiff' * weightedDiff) / gammaSum;
        end
        % Add a variance floor for numerical stability
        updatedCovariances(:, :, i) = updatedCovariances(:, :, i) + 1e-6 * eye(numCoefficients);
    end

    % =============================
    % Update transition probabilities
    % =============================
    for i = 1:numStates
        gammaSum = sum(gamma(i, 1:end-1));  % Sum of responsibilities for transitions
        if gammaSum > 0
            updatedTransitions(i, :) = sum(xi(i, :, :), 3) / gammaSum;  % Normalize by gamma
        end
    end

    % Ensure transition probabilities are normalized
    updatedTransitions = updatedTransitions ./ sum(updatedTransitions, 2);
end

function predictedWord = recognizeWord(testAudioPath, HMM, vocabulary)
    % RECOGNIZEWORD Predicts the word for a test audio file using trained HMMs.
    %
    % Inputs:
    %   testAudioPath - Path to the test audio file.
    %   HMM - Trained Hidden Markov Models (structure array).
    %   vocabulary - Cell array of word labels (e.g., {'heed', 'hid', ...}).
    %
    % Outputs:
    %   predictedWord - The recognized word (string).
    
    % Load the audio file and extract MFCC features
    [audio, fs] = audioread(testAudioPath);
    mfcc = extractMFCCs(audio, fs);  % Extract MFCCs for the test audio file
    
    % Initialize variables to track the best word prediction
    maxLikelihood = -Inf;
    predictedWord = '';
    
    % Iterate through each HMM in the vocabulary
    for wordIdx = 1:length(HMM)
        % Compute the log-likelihood of the MFCC sequence under the current HMM
        logLikelihood = viterbi(mfcc, HMM(wordIdx));
        
        % Debugging: Display the log-likelihood for each word
        % disp(['Log likelihood for ', vocabulary{wordIdx}, ': ', num2str(logLikelihood)]);
        
        % Update the predicted word if this HMM gives a higher log-likelihood
        if logLikelihood > maxLikelihood
            maxLikelihood = logLikelihood;
            predictedWord = vocabulary{wordIdx};
        end
    end
end

function prob = gaussianProbability(x, meanVec, covMatrix)
    d = length(x);
    diff = x - meanVec;

    % Add regularization
    epsilon = 1e-6;
    covMatrix = covMatrix + epsilon * eye(size(covMatrix, 1)); % Add regularization
    detCov = det(covMatrix);
    if detCov <= 0
        detCov = eps; % Avoid invalid determinant
    end
    invCov = inv(covMatrix);

    % Compute Gaussian probability
    prefactor = 1 / ((2 * pi)^(d / 2) * sqrt(detCov));
    exponent = -0.5 * diff' * invCov * diff;
    prob = max(prefactor * exp(exponent), eps); % Ensure no zero probabilities
end
