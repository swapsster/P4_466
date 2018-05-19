clear;clc;close all

% Features exstracted from training data
features = readtable('features.dat');

% Official classification
official = readtable('Official.xlsx');

c = categorical(official{:,2});

notch_index = logical(official{:, 4});

skin_index = logical(official{:, 7});
meat_index = ~skin_index;



%% Mean colour
skin_s_value = features{skin_index, 3};
skin_v_value = features{skin_index, 4};

meat_s_value = features{meat_index, 3};
meat_v_value = features{meat_index, 4};


figure
set(gcf, 'Position', [0, 0, 700, 500])
%dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
%x = features{:, 3}; y = features{:, 4};
%text(x+dx, y+dy, cellstr(c), 'FontSize',7);
hold on

scatter(skin_s_value, skin_v_value)
scatter(meat_s_value, meat_v_value)
refline([-10 1545])
title('S,V channel values from Skin and Meat side')
xlabel('Satuation')
ylabel('Value')
legend('Skin side', 'Meat side', 'Location','northwest')



%% Notches
% Save all features where notches was set to 1
notches = features{notch_index, 6};
not_notches = features{~notch_index, 6};

figure
set(gcf, 'Position', [700 0 900 400])
b = bar(c, features{:,6});
hold on
b.FaceColor = 'flat';
for i=1:length(notch_index)
    if(notch_index(i))
        b.CData(i,:) = [0.8510 0.3294 0.1020];
    end
end
y_max = min(notches);
xlim = get(gca, 'xlim');
plot(xlim,[y_max y_max])
title('Comparison of notch area')
ylabel('Notch area [pixels^2]')
legend('Fillets without notches')
