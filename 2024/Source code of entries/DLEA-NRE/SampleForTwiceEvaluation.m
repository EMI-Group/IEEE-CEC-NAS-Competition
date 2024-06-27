function [samples,maxIter] = SampleForTwiceEvaluation(maxFEs,N,k)
    samples = zeros(k,1);
    maxIter = ceil(maxFEs / N);
%     bottom = 1.05 * (100 / maxIter) ;
    bottom = 1.1;
%     bottom = 1.05;
    maxIndex = bottom^maxIter;
    interval = floor(maxIndex / k);
    for i = 1 : k
%         samples(i) = floor(log(interval * i) / log(bottom));
        samples(i) = ceil(log(interval * i) / log(bottom));
    end
end