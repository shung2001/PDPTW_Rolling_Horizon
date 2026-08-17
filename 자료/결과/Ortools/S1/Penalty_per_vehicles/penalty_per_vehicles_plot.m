%% penalty_per_vehicles_plot.m
% x축: 차량 수(num_vehicles)
% y축: 총 penalty(total_penalty)
% 최소 penalty에 대해 y = 최소값 수평선을 표시

clear;
clc;
close all;

%% 경로 설정
folderPath = "C:\Users\c\Desktop\대학생활\학연생\새로운_경로방식\PDPTW_Rolling_Horizon\자료\결과\Ortools\Penalty_per_vehicles";

% CSV 파일 경로
fileName = fullfile(folderPath, "penalty_per_vehicles.csv");

%% CSV 불러오기
data = readtable(fileName);

% 차량 수 기준 정렬
data = sortrows(data, 'num_vehicles');

x = data.num_vehicles;
y = data.total_penalty;

%% 최소 Penalty 계산
[minPenalty, minIdx] = min(y);
minVehicle = x(minIdx);

%% 그래프 생성
figure;

plot(x, y, '-o', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 5);

hold on;

%% 최소 Penalty 수평선
yline(minPenalty, '--', ...
    sprintf('y = %s', commaFormat(minPenalty)), ...
    'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'left');

%% 최소값 위치 표시
scatter(minVehicle, minPenalty, 70, 'filled');

text(minVehicle, minPenalty, ...
    sprintf('  Minimum: (%d, %s)', ...
    minVehicle, commaFormat(minPenalty)), ...
    'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'left');

%% 그래프 설정
xlabel('Number of Vehicles');
ylabel('Penalty');
title('Penalty per Number of Vehicles');

grid on;
box on;

% x축 범위
xlim([min(x), max(x)]);

% y축 지수표기 제거 및 천 단위 쉼표
ax = gca;
ax.YAxis.Exponent = 0;
ytickformat('%,.0f');

%% 범례
legend( ...
    'Total Penalty', ...
    sprintf('y = %s', commaFormat(minPenalty)), ...
    'Minimum Penalty', ...
    'Location', 'best');

hold off;

%% 결과 출력
fprintf('Minimum penalty = %s\n', commaFormat(minPenalty));
fprintf('Vehicle count   = %d\n', minVehicle);

%% PNG 저장
outputFile = fullfile(folderPath, "penalty_per_vehicles.png");
exportgraphics(gcf, outputFile, 'Resolution', 300);

fprintf('Graph saved to: %s\n', outputFile);

%% =========================================================
% 천 단위 쉼표 함수
%% =========================================================
function str = commaFormat(value)
    str = sprintf('%.0f', value);
    str = regexprep(str, '\d(?=(\d{3})+$)', '$&,');
end