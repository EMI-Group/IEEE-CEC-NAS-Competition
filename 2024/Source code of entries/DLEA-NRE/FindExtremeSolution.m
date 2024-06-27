function [Z, Extreme] = FindExtremeSolution(pop)
    % calculate extreme solution
    [n,m] = size(pop);
    Extreme = zeros(1,m);
    w       = zeros(m)+1e-6+eye(m);
    for i = 1 : m
        [~,Extreme(i)] = min(max(pop./repmat(w(i,:),n,1),[],2));
    end
    Z = pop(Extreme,:);
end