clear;clc;close all

% Load features exstracted from training data
features = readtable('features.dat');

% Load Official classification
official = readtable('Official.xlsx');

% Save fish names
c = categorical(official{:,2});

% Make index vectors of official data
notch_index = logical(official{:, 4});

skin_index = logical(official{:, 7});
meat_index = ~skin_index;

deform_index = logical(official{:, 6});

excessive_skin_index = logical(official{:, 5});



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
refline([-10 1545]) % (y = m*x + b) (10*x + y >= 1545) -> meat side

text(132, 250, 'y = -10x+1545', 'FontSize',11);
%title('S and V channel values from the skin and meat sides')
xlabel('Satuation')
ylabel('Value')
legend('Skin side', 'Meat side', 'Limit boundary', 'Location','northeast')



%% Notches
% Save all features where notches was set to 1
notches = features{notch_index, 6};
%not_notches = features{~notch_index, 6};
max_notch_area = min(notches);

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
xlim = get(gca, 'xlim');
plot(xlim,[max_notch_area max_notch_area])
title('Comparison of notch area')
ylabel('Notch area [pixels^2]')
legend('Fillets without notches')

%% Convexity

good_convexity = features{~deform_index, 7};
bad_convexity = features{deform_index, 7};
min_convexity = max(bad_convexity);

figure
set(gcf, 'Position', [700 400 900 400])
b = bar(c, features{:,7});
hold on
b.FaceColor = 'flat';
for i=1:length(deform_index)
    if(deform_index(i))
        b.CData(i,:) = [0.8510 0.3294 0.1020]; % red
    elseif(features{:,7}(i) < min_convexity)
        b.CData(i,:) = [0.8000 0.8000 0.8000]; % grey
    end
end

xlim = get(gca, 'xlim');
plot(xlim,[min_convexity min_convexity])
title('Comparison of convexity')
ylabel('Convexity')
legend('Fillets with normal shape','Location','northoutside')

%% Excessive skin

good_amount_skin = features{~excessive_skin_index, 8};
bad_amount_skin = features{excessive_skin_index, 8};
max_skin_area = min(bad_amount_skin);

figure
set(gcf, 'Position', [0 400 900 400])
b = bar(c, features{:,8});
hold on
b.FaceColor = 'flat';
for i=1:length(excessive_skin_index)
    if(excessive_skin_index(i))
        b.CData(i,:) = [0.8510 0.3294 0.1020]; % red
    end
end

xlim = get(gca, 'xlim');
plot(xlim,[max_skin_area max_skin_area])
title('Comparison of Excessive skin area')
ylabel('Excessive skin area [pixels^2]')
legend('Fillets without excessive skin', 'Skin area limit','Location','northoutside')

%% Save limits

fprintf('double max_notch_area = %.1f;\ndouble min_convexity = %.5f;\nint max_skin_area = %d;\n',max_notch_area,min_convexity,max_skin_area);
