clear all;close all;clc

%%% -- shazri 2017---- %%%

load mnist_uint8;

Input = double(reshape(train_x',28,28,60000))/255;
Input2 = double(reshape(test_x',28,28,10000))/255;
Expected = double(train_y');
test_y = double(test_y');
    
%Number of iteration/epoch
T = 600

%Number of hidden nodes
Hidden = 300

% [DataLen,SampleSize]=size (Input);
% [DataLen2,SampleSize2]=size (Input2);
% [OutLen , nocares]=size(Expected);

%%% --- Tests ---%%%
X=Input(:,:,1);
wname = 'sym4';
[CA,CH,CV,CD] = dwt2(X,wname,'mode','per');
tmp=horzcat(CA,CH,CV,CD);
sample_in = reshape(tmp,[],1);
sample_in = downsample(sample_in,3,2);
%%% --- Tests ---%%%


[DataLen,no_cares]=size(sample_in);
% DataLen
% pause

[DataLenP,DataLen_,SampleSize]=size (Input);
[DataLen2,DataLen2_,SampleSize2]=size (Input2);
[OutLen , nocares]=size(Expected);

Weight1t2 = 0.01*(2.*rand(DataLen , Hidden)-1);
Weight2t3 = 0.01*(2.*rand(Hidden+1 , OutLen)-1);

%Sigmoid Gradient
sigmoidgrad = 3 ;

%Learning Rate
lr = 0.0625 ;
%lr = 0.125 ;

Out1=ones(1,Hidden+1);
for Epoch = 2:T
    
    tempErr = 0 ;
    RandVect=randperm(SampleSize);
    for SampleChooser = 1: SampleSize
        S=RandVect(:,SampleChooser);
        
        %Input matrix is Input(row,column)
        
        %%%---Convolution Portion----%%%%%%%
        wname = 'sym4';
        [CA,CH,CV,CD] = dwt2(Input(:,:,S),wname,'mode','per');
        tmp=horzcat(CA,CH,CV,CD);
        sample_in = reshape(tmp,[],1);
        
        %%%---Down Sampling Portion----%%%%%%%
        sample_in = downsample(sample_in,3,2);
        
        
        %%%---2 Layers Fully Connected MLP----%%%%%%%
        Out1(1,2:Hidden+1) = sample_in'*Weight1t2 ;
        SigOut1 = 1./(1+exp(-sigmoidgrad*Out1));
        Out2 = SigOut1*Weight2t3 ;
        SigOut2 = exp(Out2) ./ sum(exp(Out2)) ;

        %find partial error against weight at out using cross entropy
        Err = (Expected(: , S)' - SigOut2) ;
        for i = 1:Hidden+1
            Part2t3(i,:) = Err.*SigOut1(1 , i) ;
        end
        tempErr = tempErr + sum(abs(Err)');
        %Based on sum of error definition
        ErrHidden = SigOut1.*(1-SigOut1).*(Err*Weight2t3');

        for i = 1:DataLen
            Part1t2(i,:) = ErrHidden.*sample_in(i) ;
        end

        Weight1t2 = Weight1t2 + lr*Part1t2(: , 2:Hidden+1) ;
        Weight2t3 = Weight2t3 + lr*Part2t3 ;
        SigOut2;
        Expected(:,S)';
    end
    disp('CumulativeError')
    tempErr
    


%     Epoch = Epoch-1;

end

                

