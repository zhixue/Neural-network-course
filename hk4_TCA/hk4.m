% load data
load('EEG_X.mat','X')
load('EEG_Y.mat','Y')

x = [];
y = [];

nsubject = 8;
%cross_idx = randi(10,1);
cross_idx = 4;
masked = false(3394*nsubject,1);
%domainFt = false(3394*nsubject,310);
for subject = 1:nsubject
    if subject == cross_idx
        masked(1+(subject-1)*3394:subject*3394) = 0;
   
    else
        masked(1+(subject-1)*3394:subject*3394) = 1;
  
    end
    x = [x;X{subject}];
    y = [y;Y{subject}];         
end    


cvObj.NumTestSets = 1;



%% two discrete domains dataset

cvObj.training = masked;
cvObj.test = ~cvObj.training;


% TCA
param = []; param.kerName = 'lin';param.bSstca = 0;
param.mu = 1;param.m = 10;param.gamma = .1;param.lambda = 0;
[xproj,transMdl] = ftTrans_tca(x,masked,y(masked),masked,param);

save('all8_X_D310.mat','xproj');

param.mu = 1;param.m = 2;param.gamma = .1;param.lambda = 0;
[xproj,transMdl] = ftTrans_tca(x,masked,y(masked),masked,param);
save('all8_X_D2.mat','xproj');



