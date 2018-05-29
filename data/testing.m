clear;clc;close all

% Load features exstracted from training data
% //bad meat = 1; bad skin =2; good meat 3; good skin=4; 
predicted = readtable('classification.dat');

% Load Official classification
official = readtable('Official - Testing.csv');

% Load unit test results
unit_test = readtable('unittest.dat');

% Save fish names
c = categorical(official{:,2});

% Make index vectors of official data
bloodstain_index = logical(official{:, 3});
notch_index = logical(official{:, 4});
excessive_skin_index = logical(official{:, 5});
deform_index = logical(official{:, 6});
skin_index = logical(official{:, 7});



%% General requirement test
good_fillet_index = ~bloodstain_index & ~notch_index & ~excessive_skin_index & ~deform_index;
good_fillet_predicted = predicted{:,2} == 3 | predicted{:,2} == 4;

Train = double([~good_fillet_index.'; good_fillet_index.']);
Predicted = double([~good_fillet_predicted.'; good_fillet_predicted.']);

plotconfusion(Predicted,Train)

title('')
xlabel('Predicted class', 'FontWeight','bold','fontsize',14)
ylabel('Actual class', 'FontWeight','bold','fontsize',14)
set(gca,'xticklabel',{'Bad fillet' 'Good fillet' ''},'fontsize',12)
set(gca,'yticklabel',{'Bad fillet' 'Good fillet' ''},'fontsize',12)
set(findobj(gca,'type','text'),'fontsize',14)        
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[0.8,0.8,0.8])


%% Unit test skin_meat
predicted_skin_side = unit_test{:,2};

Train = double([skin_index.'; ~skin_index.']);
Predicted = double([predicted_skin_side.'; ~predicted_skin_side.']);

plotconfusion(Predicted,Train)
title('')
xlabel('Predicted class', 'FontWeight','bold')
ylabel('Actual class', 'FontWeight','bold','fontsize',14)
set(gca,'xticklabel',{'Skin side' 'Meat side' ''},'fontsize',12)
set(gca,'yticklabel',{'Skin side' 'Meat side' ''},'fontsize',12)
set(findobj(gca,'type','text'),'fontsize',14)        
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[0.8,0.8,0.8])

%% Unit test bloodstain
predicted_bloodstain = unit_test{:,3};

Train = double([bloodstain_index.'; ~bloodstain_index.']);
Predicted = double([predicted_bloodstain.'; ~predicted_bloodstain.']);

plotconfusion(Predicted,Train)
title('')
xlabel('Predicted class', 'FontWeight','bold')
ylabel('Actual class', 'FontWeight','bold')
set(gca,'xticklabel',{'Contains bloodstain' 'No bloodstain' ''},'fontsize',12)
set(gca,'yticklabel',{'Contains bloodstain' 'No bloodstain' ''},'fontsize',12)
set(findobj(gca,'type','text'),'fontsize',14)        
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[0.8,0.8,0.8])

%% Unit test Notch
predicted_notch = unit_test{:,4};

Train = double([notch_index.'; ~notch_index.']);
Predicted = double([predicted_notch.'; ~predicted_notch.']);

plotconfusion(Predicted,Train)
title('')
xlabel('Predicted class', 'FontWeight','bold')
ylabel('Actual class', 'FontWeight','bold')
set(gca,'xticklabel',{'Contains notch' 'No notch' ''},'fontsize',12)
set(gca,'yticklabel',{'Contains notch' 'No notch' ''},'fontsize',12)
set(findobj(gca,'type','text'),'fontsize',14)        
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[0.8,0.8,0.8])

%% Unit test Deform
predicted_deform = unit_test{:,5};

Train = double([deform_index.'; ~deform_index.']);
Predicted = double([predicted_deform.'; ~predicted_deform.']);

plotconfusion(Predicted,Train)
title('')
xlabel('Predicted class', 'FontWeight','bold')
ylabel('Actual class', 'FontWeight','bold')
set(gca,'xticklabel',{'Deformity' 'Not deformity' ''},'fontsize',12)
set(gca,'yticklabel',{'Deformity' 'Not deformity' ''},'fontsize',12)
set(findobj(gca,'type','text'),'fontsize',14)        
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[0.8,0.8,0.8])

%% Unit test Excessive skin
predicted_excessive_skin = unit_test{:,6};

Train = double([excessive_skin_index.'; ~excessive_skin_index.']);
Predicted = double([predicted_excessive_skin.'; ~predicted_excessive_skin.']);

plotconfusion(Predicted,Train)
%title('Detection of excessive skin')
xlabel('Predicted class', 'FontWeight','bold')
ylabel('Actual class', 'FontWeight','bold')
set(gca,'xticklabel',{'Excessive skin' 'No skin' ''},'fontsize',12)
set(gca,'yticklabel',{'Excessive skin' 'No skin' ''},'fontsize',12)
set(findobj(gca,'type','text'),'fontsize',14)        
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[0.8,0.8,0.8])

