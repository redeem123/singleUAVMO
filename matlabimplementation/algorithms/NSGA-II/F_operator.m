function newpop = F_operator(population,MatingPool,Boundary,model)
% This function generates a new population by genetic operators
    
    Norig = size(MatingPool,1);
    if Norig == 0
        newpop = population([]);
        return;
    end
    if mod(Norig,2) == 1
        % Keep crossover pairing valid for odd selection counts.
        MatingPool(end+1,1) = MatingPool(randi(Norig));
    end
    N = size(MatingPool,1);
    D = 3;
%-----------------------------------------------------------------------------------------
% Parameters setting
    ProC = 1;       % The probability of crossover
    ProM = 1/D;     % The probability of mutation
    DisC = 20;   	% The parameter of crossover
    DisM = 20;   	% The parameter of mutation
%-----------------------------------------------------------------------------------------
% Simulated binary crossover
    % Vectorized parent extraction
    nPoints = size(population(1).rnvec, 1);
    all_rnvec = cat(3, population(MatingPool).rnvec); % [nPoints x D x N]
    
    Parent1 = reshape(permute(all_rnvec(:,:,1:N/2), [1, 3, 2]), [], D);
    Parent2 = reshape(permute(all_rnvec(:,:,N/2+1:end), [1, 3, 2]), [], D);
    
    beta    = zeros(N/2*nPoints,D);
    miu     = rand(N/2*nPoints,D);
    beta(miu<=0.5) = (2*miu(miu<=0.5)).^(1/(DisC+1));
    beta(miu>0.5)  = (2-2*miu(miu>0.5)).^(-1/(DisC+1));
    beta = beta.*(-1).^randi([0,1],N/2*nPoints,D);
    beta(rand(N/2*nPoints,D)<0.5) = 1;
    beta(repmat(rand(N/2*nPoints,1)>ProC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
%-----------------------------------------------------------------------------------------
% Polynomial mutation
    if rand<1 %Using the DTLZ mutation strategy
        MaxValue = repmat(Boundary(1,:),N*nPoints,1);
        MinValue = repmat(Boundary(2,:),N*nPoints,1);
        k    = rand(N*nPoints,D);
        miu  = rand(N*nPoints,D);
        Temp = k<=ProM & miu<0.5;
        Offspring(Temp) = Offspring(Temp)+(MaxValue(Temp)-MinValue(Temp)).*((2.*miu(Temp)+(1-2.*miu(Temp)).*(1-(Offspring(Temp)-MinValue(Temp))./(MaxValue(Temp)-MinValue(Temp))).^(DisM+1)).^(1/(DisM+1))-1);
        Temp = k<=ProM & miu>=0.5; 
        Offspring(Temp) = Offspring(Temp)+(MaxValue(Temp)-MinValue(Temp)).*(1-(2.*(1-miu(Temp))+2.*(miu(Temp)-0.5).*(1-(MaxValue(Temp)-Offspring(Temp))./(MaxValue(Temp)-MinValue(Temp))).^(DisM+1)).^(1/(DisM+1)));
    
        % Vectorized boundary enforcement
        Offspring(:,1) = max(Boundary(2,1), min(Boundary(1,1), Offspring(:,1)));
        Offspring(:,2) = max(Boundary(2,2), min(Boundary(1,2), Offspring(:,2)));
        Offspring(:,3) = max(Boundary(2,3), min(Boundary(1,3), Offspring(:,3)));

        newpop = repmat(population(1), 1, Norig);
        for pos = 1:Norig
            newpop(pos) = population(MatingPool(pos));
            newpop(pos).rnvec = Offspring((pos-1)*nPoints+1:pos*nPoints,:);
            % Sort control points to maintain a forward-moving path
            newpop(pos).rnvec = sortrows(newpop(pos).rnvec, 1);
            newpop(pos).path = newpop(pos).rnvec;
            newpop(pos) = adjust_constraint_turning_angle(newpop(pos),model);            
            newpop(pos) = evaluate(newpop(pos), model);
        end
    else %Using no mutation strategy, just crossover
        % Vectorized boundary enforcement
        Offspring(:,1) = max(Boundary(2,1), min(Boundary(1,1), Offspring(:,1)));
        Offspring(:,2) = max(Boundary(2,2), min(Boundary(1,2), Offspring(:,2)));
        Offspring(:,3) = max(Boundary(2,3), min(Boundary(1,3), Offspring(:,3)));

        newpop = repmat(population(1), 1, Norig);
        for pos = 1:Norig
            newpop(pos) = population(MatingPool(pos));
            newpop(pos).rnvec = Offspring((pos-1)*nPoints+1:pos*nPoints,:);
            newpop(pos).rnvec = sortrows(newpop(pos).rnvec, 1);
            newpop(pos).path = newpop(pos).rnvec;
            newpop(pos) = adjust_constraint_turning_angle(newpop(pos),model);
            newpop(pos) = evaluate(newpop(pos), model);
        end
    end


end
