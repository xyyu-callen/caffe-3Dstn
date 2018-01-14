clear;
clc;
close all;
ST_FCN_RTS = 'ST-FCN-R.log';
ST_CNN_RTS = 'ST-CNN-R.log';
FCN_RTS = 'FCN-R.log';
CNN_RTS = 'CNN-R.log';
% display value in solver.prototxt
train_interval = 100;
% test_interval value in solver.prototxt
test_interval = 1000;
ST_FCN_RTS_O = 'ST-FCN-RTS-O.log'
ST_FCN_RTS_ = 'ST-FCN-RTS.log'



[~, string_output] = dos(['cat ', ST_FCN_RTS_O, ' | grep ''Train net output #0'' | awk ''{print $11}''']);
st_fcn_train_loss_O = str2num(string_output);
n = 1:length(st_fcn_train_loss_O);
st_fcn_idx_train_O = (n - 1) * train_interval;

[~, string_output] = dos(['cat ', ST_FCN_RTS_O, ' | grep ''Test net output #1'' | awk ''{print $11}''']);
st_fcn_test_loss_O = str2num(string_output);
m = 1:length(st_fcn_test_loss_O);
st_fcn_idx_test_O = (m - 1) * test_interval;

[~, string_output] = dos(['cat ', ST_FCN_RTS_, ' | grep ''Train net output #0'' | awk ''{print $11}''']);
st_fcn_train_loss_ = str2num(string_output);
n = 1:length(st_fcn_train_loss_);
st_fcn_idx_train_ = (n - 1) * train_interval;

[~, string_output] = dos(['cat ', ST_FCN_RTS_, ' | grep ''Test net output #1'' | awk ''{print $11}''']);
st_fcn_test_loss_ = str2num(string_output);
m = 1:length(st_fcn_test_loss_);
st_fcn_idx_test_ = (m - 1) * test_interval;

figure;
plot(st_fcn_idx_train_O, st_fcn_train_loss_O);
hold on;
plot(st_fcn_idx_train_, st_fcn_train_loss_);
hold on;
plot(st_fcn_idx_test_O, st_fcn_test_loss_O);
hold on;
plot(st_fcn_idx_test_, st_fcn_test_loss_);

grid on;
legend('Train Loss ST-FCN-O', 'Train Loss ST-FCN', 'Test Loss ST-FCN-O', 'Test Loss ST-FCN');
xlabel('iterations');
ylabel('loss');
title('Loss Curve on Mnist RTS ');


%{
[~, string_output] = dos(['cat ', ST_FCN_RTS, ' | grep ''Train net output #0'' | awk ''{print $11}''']);
st_fcn_train_loss = str2num(string_output);
n = 1:length(st_fcn_train_loss);
st_fcn_idx_train = (n - 1) * train_interval;

[~, string_output] = dos(['cat ', ST_FCN_RTS, ' | grep ''Test net output #1'' | awk ''{print $11}''']);
st_fcn_test_loss = str2num(string_output);
m = 1:length(st_fcn_test_loss);
st_fcn_idx_test = (m - 1) * test_interval;

[~, string_output] = dos(['cat ', ST_CNN_RTS, ' | grep ''Train net output #0'' | awk ''{print $11}''']);
st_cnn_train_loss = str2num(string_output);
n = 1:length(st_cnn_train_loss);
st_cnn_idx_train = (n - 1) * train_interval;

[~, string_output] = dos(['cat ', ST_CNN_RTS, ' | grep ''Test net output #1'' | awk ''{print $11}''']);
st_cnn_test_loss = str2num(string_output);
m = 1:length(st_cnn_test_loss);
st_cnn_idx_test = (m - 1) * test_interval;

[~, string_output] = dos(['cat ', FCN_RTS, ' | grep ''Train net output #0'' | awk ''{print $11}''']);
fcn_train_loss = str2num(string_output);
n = 1:length(fcn_train_loss);
fcn_idx_train = (n - 1) * train_interval;

[~, string_output] = dos(['cat ', FCN_RTS, ' | grep ''Test net output #1'' | awk ''{print $11}''']);
fcn_test_loss = str2num(string_output);
m = 1:length(fcn_test_loss);
fcn_idx_test = (m - 1) * test_interval;

[~, string_output] = dos(['cat ', CNN_RTS, ' | grep ''Train net output #0'' | awk ''{print $11}''']);
cnn_train_loss = str2num(string_output);
n = 1:length(cnn_train_loss);
cnn_idx_train = (n - 1) * train_interval;

[~, string_output] = dos(['cat ', CNN_RTS, ' | grep ''Test net output #1'' | awk ''{print $11}''']);
cnn_test_loss = str2num(string_output);
m = 1:length(cnn_test_loss);
cnn_idx_test = (m - 1) * test_interval;


figure;
% plot(idx_test, test_loss, 'g', 'LineWidth', 0.5);

%{
plot(st_fcn_idx_train, st_fcn_train_loss);
hold on;
plot(st_cnn_idx_train, st_cnn_train_loss);
hold on;
plot(fcn_idx_train, fcn_train_loss);
hold on;
plot(cnn_idx_train, cnn_train_loss);
%}


plot(st_fcn_idx_test, st_fcn_test_loss);
hold on;
plot(st_cnn_idx_test, st_cnn_test_loss);
hold on;
plot(fcn_idx_test, fcn_test_loss);
hold on;
plot(cnn_idx_test, cnn_test_loss);


grid on;
%legend('Train Loss ST-FCN', 'Train Loss ST-CNN', 'Train Loss FCN', 'Train Loss CNN');
legend('Test Loss ST-FCN', 'Test Loss ST-CNN', 'Test Loss FCN', 'Test Loss CNN');
xlabel('iterations');
ylabel('loss');
title('Test Loss Curve on Mnist R ');
%title('Train & Test Loss Curve on RTS');
%}

