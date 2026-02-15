clc; close; clear all;
initTime = 0;
save('InitTimeNMOPSO6.mat','initTime');
for iter = 1:1
    disp(['The number of turn ' num2str(iter)]);
    NMOPSO;
end

function NMOPSO()
    %% Problem Definition
    model = CreateModel(); % Create search map and parameters
    model_name = 6;
    
    nVar=model.n;       % Number of Decision Variables = searching dimension of PSO = number of path nodes
    
    VarSize=[1 nVar];   % Size of Decision Variables Matrix
    
    % Lower and upper Bounds of particles (Variables)
    VarMin.x=model.xmin;           
    VarMax.x=model.xmax;           
    VarMin.y=model.ymin;           
    VarMax.y=model.ymax;           
    VarMin.z=model.zmin;           
    VarMax.z=model.zmax;                 
    
    VarMax.r=3*norm(model.start-model.end)/nVar;  % r is distance
    VarMin.r=VarMax.r/9;
    
    % Inclination (elevation)
    AngleRange = pi/4; % Limit the angle range for better solutions
    VarMin.psi=-AngleRange;            
    VarMax.psi=AngleRange;          
    
    % Azimuth 
    VarMin.phi=-AngleRange;            
    VarMax.phi=AngleRange;          
    
    % Lower and upper Bounds of 
    alpha=0.5;
    VelMax.r=alpha*(VarMax.r-VarMin.r);    
    VelMin.r=-VelMax.r;                    
    VelMax.psi=alpha*(VarMax.psi-VarMin.psi);    
    VelMin.psi=-VelMax.psi;                    
    VelMax.phi=alpha*(VarMax.phi-VarMin.phi);    
    VelMin.phi=-VelMax.phi;   
    
    CostFunction=@(x) MyCost(x,model,VarMin);    % Cost Function
    
    %% PSO Parameters
    
    dummy_output = CostFunction(struct('x', ones(1, model.n), 'y', ones(1, model.n), 'z', ones(1, model.n)));
    nObj = numel(dummy_output);                   % Determine the number of objectives
    
    MaxIt = 500;          % Maximum Number of Iterations
    
    nPop = 100;           % Population Size (Swarm Size)
            
    nRep = 50;            % Repository Size
    
    w = 1;                % Inertia Weight
    wdamp = 0.98;         % Inertia Weight Damping Ratio
    c1 = 1.5;             % Personal Learning Coefficient
    c2 = 1.5;             % Global Learning Coefficient
    
    nGrid = 5;            % Number of Grids per Dimension
    alpha = 0.1;          % Inflation Rate
    
    beta = 2;             % Leader Selection Pressure
    gamma = 2;            % Deletion Selection Pressure
    
    mu = 0.5;             % Mutation Rate
    delta = 20;           % delta = num(rep)/10
    
    %% Initialization
    % Create Empty Particle Structure
    empty_particle.Position=[];
    empty_particle.Velocity=[];
    empty_particle.Cost=[];
    empty_particle.Best.Position=[];
    empty_particle.Best.Cost=[];
    empty_particle.IsDominated = [];
    empty_particle.GridIndex = [];
    empty_particle.GridSubIndex = [];
    
    % Initialize Global Best
    GlobalBest.Cost=Inf(nObj,1); % Minimization problem
    
    % Create an empty Particles Matrix, each particle is a solution (searching path)
    particle=repmat(empty_particle,nPop,1);
    
    isInit = false;
    % tic;
    while (~isInit)
        disp('Initialising...');
        for i=1:nPop
    
            % Initialize Position
            particle(i).Position=CreateRandomSolution(VarSize,VarMin,VarMax);
    
            % Initialize Velocity
            particle(i).Velocity.r=zeros(VarSize);
            particle(i).Velocity.psi=zeros(VarSize);
            particle(i).Velocity.phi=zeros(VarSize);
    
            % Evaluation
            particle(i).Cost= CostFunction(SphericalToCart2(particle(i).Position,model));
    
            % Update Personal Best
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
    
            % Update Global Best
            if Dominates(particle(i).Best.Cost,GlobalBest.Cost)
                GlobalBest=particle(i).Best;
                isInit = true;
            end
        end
    end
    
    % Array to Hold Best Cost Values at Each Iteration
    BestCost=zeros(MaxIt,nObj);
    
    % Determine Domination
    particle = DetermineDomination(particle);
    
    rep = particle(~[particle.IsDominated]); % the un-dominated 
    % rep.GridSubIndex = [];
    % rep.GridIndex = [];
    
    Grid = CreateGrid(rep, nGrid, alpha);
    
    for i = 1:numel(rep)
        rep(i) = FindGridIndex(rep(i), Grid);
    end
    
    %% PSO Main Loop
    
    for it=1:(MaxIt)
    
        % Update Best Cost Ever Found
        BestCost(it,:)=GlobalBest.Cost;
        
        for i=1:nPop   
            
            % select leader = update global best
            GlobalBest = SelectLeader(rep, beta);
            
            
            % ----------------------r Part--------------------------        
            % Update Velocity
            particle(i).Velocity.r = w*particle(i).Velocity.r ...
                + c1*rand(VarSize).*(particle(i).Best.Position.r-particle(i).Position.r) ...
                + c2*rand(VarSize).*(GlobalBest.Position.r-particle(i).Position.r);
    
            % Update Velocity Bounds
            particle(i).Velocity.r = max(particle(i).Velocity.r,VelMin.r);
            particle(i).Velocity.r = min(particle(i).Velocity.r,VelMax.r);
    
            % Update Position
            particle(i).Position.r = particle(i).Position.r + particle(i).Velocity.r;
    
            % Velocity Mirroring
            % If a particle moves out of the range, it will moves backward next
            % time
            OutOfTheRange=(particle(i).Position.r<VarMin.r | particle(i).Position.r>VarMax.r);
            particle(i).Velocity.r(OutOfTheRange)=-particle(i).Velocity.r(OutOfTheRange);
    
            % Update Position Bounds
            particle(i).Position.r = max(particle(i).Position.r,VarMin.r);
            particle(i).Position.r = min(particle(i).Position.r,VarMax.r);
    
            
            % -------------------psi Part----------------------
    
            % Update Velocity
            particle(i).Velocity.psi = w*particle(i).Velocity.psi ...
                + c1*rand(VarSize).*(particle(i).Best.Position.psi-particle(i).Position.psi) ...
                + c2*rand(VarSize).*(GlobalBest.Position.psi-particle(i).Position.psi);
    
            % Update Velocity Bounds
            particle(i).Velocity.psi = max(particle(i).Velocity.psi,VelMin.psi);
            particle(i).Velocity.psi = min(particle(i).Velocity.psi,VelMax.psi);
    
            % Update Position
            particle(i).Position.psi = particle(i).Position.psi + particle(i).Velocity.psi;
    
            % Velocity Mirroring
            OutOfTheRange=(particle(i).Position.psi<VarMin.psi | particle(i).Position.psi>VarMax.psi);
            particle(i).Velocity.psi(OutOfTheRange)=-particle(i).Velocity.psi(OutOfTheRange);
    
            % Update Position Bounds
            particle(i).Position.psi = max(particle(i).Position.psi,VarMin.psi);
            particle(i).Position.psi = min(particle(i).Position.psi,VarMax.psi);
    
            
            % -----------------------Phi part----------------------------------
            % Update Velocity
            particle(i).Velocity.phi = w*particle(i).Velocity.phi ...
                + c1*rand(VarSize).*(particle(i).Best.Position.phi-particle(i).Position.phi) ...
                + c2*rand(VarSize).*(GlobalBest.Position.phi-particle(i).Position.phi);
    
            % Update Velocity Bounds
            particle(i).Velocity.phi = max(particle(i).Velocity.phi,VelMin.phi);
            particle(i).Velocity.phi = min(particle(i).Velocity.phi,VelMax.phi);
    
            % Update Position
            particle(i).Position.phi = particle(i).Position.phi + particle(i).Velocity.phi;
    
            % Velocity Mirroring
            OutOfTheRange=(particle(i).Position.phi<VarMin.phi | particle(i).Position.phi>VarMax.phi);
            particle(i).Velocity.phi(OutOfTheRange)=-particle(i).Velocity.phi(OutOfTheRange);
    
            % Update Position Bounds
            particle(i).Position.phi = max(particle(i).Position.phi,VarMin.phi);
            particle(i).Position.phi = min(particle(i).Position.phi,VarMax.phi);
    
                    
            %----------------------- Evaluation------------------------
            particle(i).Cost=CostFunction(SphericalToCart2(particle(i).Position,model));
            
            %------ Apply mutation-------
            pm = (1-(it-1)/(MaxIt-1))^(1/mu);
            if rand<pm
                NewSol.Position = Mutate(particle(i),rep,delta,VarMax,VarMin);
                NewSol.Cost = CostFunction(SphericalToCart2(NewSol.Position,model));
                if Dominates(NewSol, particle(i))
                    particle(i).Position = NewSol.Position;
                    particle(i).Cost = NewSol.Cost;
    
                elseif Dominates(particle(i),NewSol)
                    %do nothing
    
                else
                    if rand < 0.5
                        particle(i).Position = NewSol.Position;
                        particle(i).Cost = NewSol.Cost;
                    end
                end
            end
            % Update Personal Best
                
            if Dominates(particle(i), particle(i).Best)
    
                particle(i).Best.Position=particle(i).Position;
                particle(i).Best.Cost=particle(i).Cost;
                
            elseif Dominates(particle(i).Best, particle(i))
                % Do Nothing
                
            else
                if rand<0.5
                    particle(i).Best.Position = particle(i).Position;
                    particle(i).Best.Cost = particle(i).Cost;
                end
            end
    
        end
        
        % Determine Domination
        particle = DetermineDomination(particle);
    
        % Add Non-Dominated Particles to REPOSITORY
        rep = [rep
             particle(~[particle.IsDominated])]; %#ok
        
        % Determine Domination of New Resository Members
        rep = DetermineDomination(rep);
        
        % Keep only Non-Dminated Members in the Repository
        rep = rep(~[rep.IsDominated]);
        
        % Update Grid
        Grid = CreateGrid(rep, nGrid, alpha);
        
        % Update Grid Indices
        for i = 1:numel(rep)
            rep(i) = FindGridIndex(rep(i), Grid);
        end
        
        % Check if Repository is Full
        if numel(rep)>nRep
            Extra = numel(rep)-nRep;
            for e = 1:Extra
                rep = DeleteOneRepMember(rep, gamma);
            end   
        end
        
        % Inertia Weight Damping
        w=w*wdamp;
    
    %     Show Iteration Information 
        if mod(it, 20) == 0 || it == 1
            fprintf('Iteration %d: Path=%.5f, Threat=%.5f, Altitude=%.5f, Smoothness=%.5f\n', ...
                    it, BestCost(it,1), BestCost(it,2), BestCost(it,3), BestCost(it,4));
        end
    end
    
    GlobalBest = SelectLeader(rep, beta);
    
    %% Plot results
    % Calculate F1-F4 performance metrics
    final_metrics = calculatePerformanceMetrics(rep, nObj);
    
    % Display F1-F4 metrics
    fprintf('\n=== NMOPSO Performance Metrics (F1-F4 Objectives) ===\n');
    fprintf('F1=Path Length, F2=Threat Avoidance, F3=Altitude, F4=Smoothness\n\n');
    fprintf('F1 (Path): Max=%.4f, Min=%.4f, Mean=%.4f, Std=%.4f\n', ...
            final_metrics.F1_max, final_metrics.F1_min, final_metrics.F1_mean, final_metrics.F1_std);
    fprintf('F2 (Threat): Max=%.4f, Min=%.4f, Mean=%.4f, Std=%.4f\n', ...
            final_metrics.F2_max, final_metrics.F2_min, final_metrics.F2_mean, final_metrics.F2_std);
    fprintf('F3 (Altitude): Max=%.4f, Min=%.4f, Mean=%.4f, Std=%.4f\n', ...
            final_metrics.F3_max, final_metrics.F3_min, final_metrics.F3_mean, final_metrics.F3_std);
    fprintf('F4 (Smooth): Max=%.4f, Min=%.4f, Mean=%.4f, Std=%.4f\n\n', ...
            final_metrics.F4_max, final_metrics.F4_min, final_metrics.F4_mean, final_metrics.F4_std);
    
    % Best solution
    BestPosition = SphericalToCart2(GlobalBest.Position,model);
    smooth = 1;
    PlotSolution(BestPosition,model,smooth);
end

function metrics = calculatePerformanceMetrics(archive, nObj)
    if isempty(archive)
        % Initialize all F1-F4 statistics to default values
        metrics.F1_max = 0; metrics.F1_min = 0; metrics.F1_mean = 0; metrics.F1_std = 0;
        metrics.F2_max = 0; metrics.F2_min = 0; metrics.F2_mean = 0; metrics.F2_std = 0;
        metrics.F3_max = 0; metrics.F3_min = 0; metrics.F3_mean = 0; metrics.F3_std = 0;
        metrics.F4_max = 0; metrics.F4_min = 0; metrics.F4_mean = 0; metrics.F4_std = 0;
        return;
    end
    
    % Extract cost values for all solutions in archive
    % Each archive(i).Cost is a column vector [J1; J2; J3; J4], so transpose and concatenate
    costs = horzcat(archive.Cost)'; % Each row is a solution, each column is an objective
    
    % Calculate statistics for F1 (Path Length)
    F1_values = costs(:, 1);
    metrics.F1_max = max(F1_values);
    metrics.F1_min = min(F1_values);
    metrics.F1_mean = mean(F1_values);
    metrics.F1_std = std(F1_values);
    
    % Calculate statistics for F2 (Threat Avoidance)
    F2_values = costs(:, 2);
    metrics.F2_max = max(F2_values);
    metrics.F2_min = min(F2_values);
    metrics.F2_mean = mean(F2_values);
    metrics.F2_std = std(F2_values);
    
    % Calculate statistics for F3 (Altitude)
    F3_values = costs(:, 3);
    metrics.F3_max = max(F3_values);
    metrics.F3_min = min(F3_values);
    metrics.F3_mean = mean(F3_values);
    metrics.F3_std = std(F3_values);
    
    % Calculate statistics for F4 (Smoothness)
    F4_values = costs(:, 4);
    metrics.F4_max = max(F4_values);
    metrics.F4_min = min(F4_values);
    metrics.F4_mean = mean(F4_values);
    metrics.F4_std = std(F4_values);
end   
    
function A = TransfomationMatrix(r,phi,psi)

Rot_z = [ cos(phi), -sin(phi), 0, 0;...
          sin(phi),  cos(phi), 0, 0;...
                 0,         0, 1, 0;...
                 0,         0, 0, 1];
             
Rot_y = [ cos(-psi), 0, sin(-psi), 0;...
                  0, 1,         0, 0;...
         -sin(-psi), 0, cos(-psi), 0;...
                  0, 0,         0, 1];
 
Trans_x = [1 0 0 r;...
           0 1 0 0;...
           0 0 1 0;...
           0 0 0 1];

A = Rot_z*Rot_y*Trans_x;

end
    
    % the positive direction is counterclockwise


function position = SphericalToCart2(solution,model)
    %% solution of sphere space
    r = solution.r;
    phi = solution.phi;
    psi = solution.psi;
    
    %% find the start matrix
    %start position is 4*4 matrix that including postion and orientation
    xs = model.start(1);
    ys = model.start(2);
    zs = model.start(3);
    %supplement the start orientation
    start = [1, 0, 0, xs;...
             0, 1, 0, ys;...
             0, 0, 1, zs;...
             0, 0, 0,  1];
    dirVector = model.end - model.start;
    
    phistart = atan2(dirVector(2),dirVector(1));
    psistart = atan2(dirVector(3),norm([dirVector(1),dirVector(2)]));
    
    dir = TransfomationMatrix(0,phistart,psistart);
    startPosition = start*dir;
    
    %% find the position of each particle
    % T is the transformation matrix from start position to i position
    T(1).value = TransfomationMatrix(r(1),phi(1),psi(1));
    pos(1).value = startPosition*T(1).value;
    
    x(1) = pos(1).value(1,4);
    x(1) = max(model.xmin,x(1));
    x(1) = min(model.xmax,x(1));
    
    y(1) = pos(1).value(2,4);
    y(1) = max(model.ymin,y(1));
    y(1) = min(model.ymax,y(1));
    
    z(1) = pos(1).value(3,4);
    z(1) = max(model.zmin,z(1));
    z(1) = min(model.zmax,z(1));
    
    for i=2:model.n
       
       T(i).value = T(i-1).value*TransfomationMatrix(r(i),phi(i),psi(i));
       pos(i).value = startPosition*T(i).value;
      
       x(i) = pos(i).value(1,4);
       x(i) = max(model.xmin,x(i));
       x(i) = min(model.xmax,x(i));
       
       y(i) = pos(i).value(2,4);
       y(i) = max(model.ymin,y(i));
       y(i) = min(model.ymax,y(i));
       
       z(i) = pos(i).value(3,4);
       z(i) = max(model.zmin,z(i));
       z(i) = min(model.zmax,z(i));
    
    end
    
    position.x = x;
    position.y = y;
    position.z = z;
end

function leader = SelectLeader(rep, beta)

    % Grid Index of All Repository Members
    GI = [rep.GridIndex];
    
    % Occupied Cells
    OC = unique(GI); % unique() find and sort unique values in an array
    
    % Number of Particles in Occupied Cells
    N = zeros(size(OC));
    for k = 1:numel(OC)
        N(k) = numel(find(GI == OC(k)));
    end
    
    % Selection Probabilities
    P = exp(-beta*N);
    P = P/sum(P);
    
    % Selected Cell Index
    sci = RouletteWheelSelection(P);
    
    % Selected Cell
    sc = OC(sci);
    
    % Selected Cell Members
    SCM = find(GI == sc);
    
    % Selected Member Index
    smi = randi([1 numel(SCM)]);
    
    % Selected Member
    sm = SCM(smi);
    
    % Leader
    leader = rep(sm);

end

function i = RouletteWheelSelection(P)

    r = rand;
    
    C = cumsum(P); % cumulative sum
    
    i = find(r <= C, 1, 'first');

end

%{
 This function will plot:
- model with a terrain map and obstacles
- solutions with different views
%}

function PlotSolution(sol,model,smooth)

    %% Plot 3D view
    figure(1)
    PlotModel(model)
    
    x=sol.x;
    y=sol.y;
    z=sol.z;
    
    % Start location
    xs=model.start(1);
    ys=model.start(2);
    zs=model.start(3);
    
    % Final location
    xf=model.end(1);
    yf=model.end(2);
    zf=model.end(3);
    
    x_all = [xs x xf];
    y_all = [ys y yf];
    z_all = [zs z zf];

    N = size(x_all,2); % real path length
    
   % Path height is relative to the ground height
    for i = 1:N
        z_map = model.H(round(y_all(i)),round(x_all(i)));
        z_all(i) = z_all(i) + z_map;
    end
    
    % given data in a point matrix, xyz, which is 3 x number of points
    xyz = [x_all;y_all;z_all];
    [ndim,npts]=size(xyz);
    xyzp=zeros(size(xyz));
    for k=1:ndim
       xyzp(k,:)=ppval(csaps(1:npts,xyz(k,:),smooth),1:npts);
    end
    plot3(xyzp(1,:),xyzp(2,:),xyzp(3,:),'k','LineWidth',2);

    for i=2:(N-1)
       plot3(x_all(i),y_all(i),z_all(i),'ko','MarkerSize',5,'MarkerFaceColor','y');
    end
    % plot start point
    plot3(x_all(1),y_all(1),z_all(1),'ks','MarkerSize',7,'MarkerFaceColor','k');
    % plot target point
    plot3(x_all(N),y_all(N),z_all(N),'ko','MarkerSize',7,'MarkerFaceColor','k');
    hold off;
    
    %% Plot top view
    figure(3)
    mesh(model.X,model.Y,model.H); % Plot the data
    colormap summer;                    % Default color map.
    set(gca, 'Position', [0 0 1 1]); % Fill the figure window.
    axis equal vis3d on;            % Set aspect ratio and turn off axis.
    shading interp;                  % Interpolate color across faces.
    material dull;                   % Mountains aren't shiny.
    camlight left;                   % Add a light over to the left somewhere.
    lighting gouraud;                % Use decent lighting.
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    hold on
    
    % Threats as cylinders
    threats = model.threats;
    threat_num = size(threats,1);
    
    for i = 1:threat_num
        threat = threats(i,:);
        threat_x = threat(1);
        threat_y = threat(2);
        threat_z = max(max(model.H))+1;  % choose z to be the highest peak
        threat_radius = threat(4);

        for j=1:3 
        % Define circle parameters:
        % Make an array for all the angles:
        theta = linspace(0, 2 * pi, 2000);
        % Create the x and y locations at each angle:
        x = threat_radius * cos(theta) + threat_x;
        y = threat_radius * sin(theta) + threat_y;
        % Need to make a z value for every (x,y) pair:
        z = zeros(1, numel(x)) + threat_z;
        % Do the plot:
        % First plot the center:
        plot3(threat_x, threat_y, threat_z, 'o', 'color', 'red', 'MarkerSize', 3, 'MarkerFaceColor','red');
        % Next plot the circle:
        plot3(x, y, z, '-', 'color', 'red', 'LineWidth', 1);
        
        % Repeat for a smaller radius
        threat_radius = threat_radius - 20;
        end
    end

    % plot path
    plot3(xyzp(1,:),xyzp(2,:),xyzp(3,:),'k','LineWidth',2);

    for i=2:(N-1)
       plot3(x_all(i),y_all(i),z_all(i),'ko','MarkerSize',5,'MarkerFaceColor','y');
    end

    % plot start point
    plot3(x_all(1),y_all(1),z_all(1),'ks','MarkerSize',7,'MarkerFaceColor','k');

    % plot target point
    plot3(x_all(N),y_all(N),z_all(N),'ko','MarkerSize',7,'MarkerFaceColor','k');
    
    % Set top view
    view(0,90)
    hold off;
    
    
    %% Plot side view
    figure(7)
    mesh(model.X,model.Y,model.H); % Plot the data
    colormap summer;                    % Default color map.
    set(gca, 'Position', [0 0 1 1]); % Fill the figure window.
    axis equal vis3d on;            % Set aspect ratio and turn off axis.
    shading interp;                  % Interpolate color across faces.
    material dull;                   % Mountains aren't shiny.
    camlight left;                   % Add a light over to the left somewhere.
    lighting gouraud;                % Use decent lighting.
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    hold on

    % plot path
    plot3(xyzp(1,:),xyzp(2,:),xyzp(3,:),'k','LineWidth',2);

    for i=2:(N-1)
       plot3(x_all(i),y_all(i),z_all(i),'ko','MarkerSize',5,'MarkerFaceColor','y');
    end

    % plot start point
    plot3(x_all(1),y_all(1),z_all(1),'ks','MarkerSize',7,'MarkerFaceColor','k');

    % plot target point
    plot3(x_all(N),y_all(N),z_all(N),'ko','MarkerSize',7,'MarkerFaceColor','k');
    
    view(90,0);
    hold off;

end

% Plot the terrain model and threats
function PlotModel(model)

    mesh(model.X,model.Y,model.H);   % Plot the data
    colormap summer;                 % Default color map.
    set(gca, 'Position', [0 0 1 1]); % Fill the figure window.
    axis equal vis3d on;             % Set aspect ratio and turn off axis.
    shading interp;                  % Interpolate color across faces.
    material dull;                   % Mountains aren't shiny.
    camlight left;                   % Add a light over to the left somewhere.
    lighting gouraud;                % Use decent lighting.
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    hold on

    % Threats as cylinders
    threats = model.threats;
    threat_num = size(threats,1);
    h=250; % Height
    
    for i = 1:threat_num
        threat = threats(i,:);
        threat_x = threat(1);
        threat_y = threat(2);
        threat_z = threat(3);
        threat_radius = threat(4);


        [xc,yc,zc]=cylinder(threat_radius); % create a unit cylinder
        % set the center and height 
        xc=xc+threat_x;  
        yc=yc+threat_y;
        zc=zc*h+threat_z;
        c = mesh(xc,yc,zc); % plot the cylinder 
        set(c,'Edgecolor','none','Facecolor','red','FaceAlpha',.3); % set color and transparency
    end

end

function cost=MyCost(sol,model,varmin)

    J_inf = inf;
    n = model.n; % n is the number of path node, not including start point
    H = model.H; % H is the map
    
    % Input solution
    x=sol.x;
    y=sol.y;
    z=sol.z;
    
    % Start location
    xs=model.start(1);
    ys=model.start(2);
    zs=model.start(3);
    
    % Final location
    xf=model.end(1);
    yf=model.end(2);
    zf=model.end(3);
    
    x_all = [xs x xf];
    y_all = [ys y yf];
    z_all = [zs z zf];
    
    N = size(x_all,2); % Full path length
    
    % Altitude wrt sea level = z_relative + ground_level
    z_abs = zeros(1,N);
    for i = 1:N
        z_abs(i) = z_all(i) + H(round(y_all(i)),round(x_all(i)));
    end
    
    %============================================
    % J1 - Cost for path length 
    Traj = 0;
    % rmax = varmax.r;
    for i = 1:N-1
        diff = [x_all(i+1) - x_all(i);y_all(i+1) - y_all(i);z_abs(i+1) - z_abs(i)];
        if norm(diff) <= varmin.r
            Traj = 0;
            break;
        end
        Traj = Traj + norm(diff);
    end
    PP = norm([xf yf zf]-[xs ys zs]);
    J1 = abs(1 - PP/Traj);

    %==============================================
    % J2 - threats/obstacles Cost   

    % Threats/Obstacles
    threats = model.threats;
    threat_num = size(threats,1);
    
    drone_size = 1;
    danger_dist = 10*drone_size;
    
    J2 = 0;
    n2 = 0;
    for i = 1:threat_num
        threat = threats(i,:);
        threat_x = threat(1);
        threat_y = threat(2);
        threat_radius = threat(4);
        for j = 1:N-1
            % Distance between projected line segment and threat origin
            dist = DistP2S([threat_x threat_y],[x_all(j) y_all(j)],[x_all(j+1) y_all(j+1)]);
            if dist > (threat_radius + drone_size + danger_dist) % No collision
                threat_cost = 0;
                % do nothing
            elseif dist < (threat_radius + drone_size)  % Collision
                threat_cost = J_inf;
            else  % danger
                threat_cost = 1 - (dist-drone_size-threat_radius)/danger_dist;
            end
            n2 = n2+1;
            J2 = J2 + threat_cost;
        end
    end
    J2 = J2/n2;

    %==============================================
    % J3 - Altitude cost
    % Note: In this calculation, z, zmin & zmax are heights with respect to the ground
    zmax = model.zmax;
    zmin = model.zmin;
    J3 = 0;
    n3 = 0;
    for i=1:n        
        if z(i) < 0   % crash into ground
            J3_node = J_inf;
        else
            J3_node = abs(z(i) - (zmax + zmin)/2); 
        end
        J3_node = J3_node/((zmax-zmin)/2);
        n3 = n3+1;
        J3 = J3 + J3_node;
    end
    J3 = J3/n3;

    %==============================================
    % J4 - Smooth cost
    J4 = 0;
    n4 = 0;
    %find the heading angle at i position
    for i = 1:N-2   
        % P(ij)P(i,j+1)        
        for j = i:-1:1
             segment1 = [x_all(j+1); y_all(j+1); z_abs(j+1)] - [x_all(j); y_all(j); z_abs(j)];
             if nnz(segment1) ~= 0 % returns the number of nonzero elements in matrix
                 break; % --> if point(j+1) and point(j) is not coincide  
             end
        end
        
        % P(i,j+1)P(i,j+2)
        for j = i:N-2
            segment2 = [x_all(j+2); y_all(j+2); z_abs(j+2)] - [x_all(j+1); y_all(j+1); z_abs(j+1)];
             if nnz(segment2) ~= 0 
                 break;
             end
        end

        heading_angle = atan2(norm(cross(segment1,segment2)),dot(segment1,segment2));

        heading_angle = abs(heading_angle)/pi;
        n4 = n4+1;
        J4 = J4 + abs(heading_angle);
%         J4 = J4 + abs(climb_angle);
    end
    J4 = J4/n4;

    %============================================
    % Overall cost

    cost = [J1;J2;J3;J4];
end

function xnew = Mutate(x,pm,delta,VarMax,VarMin)

    nVar = numel(x.Position.r);
    pbest = x.Best;
    
    beta = tanh(delta*length(pm)); % alpha/F in reference
        
    xnew.r = x.Position.r + randn(1,nVar).*pbest.Position.r*beta;
    xnew.phi = x.Position.phi + randn(1,nVar).*pbest.Position.phi*beta;
    xnew.psi = x.Position.psi + randn(1,nVar).*pbest.Position.psi*beta;

    xnew.r = max(VarMax.r,xnew.r);
    xnew.r = min(VarMin.r,xnew.r);

    xnew.phi = max(VarMax.phi,xnew.phi);
    xnew.phi = min(VarMin.phi,xnew.phi);

    xnew.psi = max(VarMax.psi,xnew.psi);
    xnew.psi = min(VarMin.psi,xnew.psi);
end


function particle = FindGridIndex(particle, Grid)

    nObj = numel(particle.Cost);
    
    nGrid = numel(Grid(1).LB)-2;
    
    particle.GridSubIndex = zeros(1, nObj);
    idx = zeros(1, nObj);
    
    for j = 1:nObj
        
        if isempty(find(particle.Cost(j)<Grid(j).UB, 1, 'first'))
            idx(j) = nGrid;
        else
            idx(j) = find(particle.Cost(j)<Grid(j).UB, 1, 'first'); 
        end
    end
    particle.GridSubIndex = idx;

    particle.GridIndex = particle.GridSubIndex(1);
    for j = 2:nObj
        particle.GridIndex = particle.GridIndex-1;
        particle.GridIndex = nGrid*particle.GridIndex;
        particle.GridIndex = particle.GridIndex+particle.GridSubIndex(j);
    end
    
end

% Calculate the minimum Distance between a Point to a Segment

function dist = DistP2S(x,a,b)
    d_ab = norm(a-b);
    d_ax = norm(a-x);
    d_bx = norm(b-x);
    if d_ab ~= 0 
        if dot(a-b,x-b)*dot(b-a,x-a)>=0 % x is between a and b
            A = [b-a;x-a];
            dist = abs(det(A))/d_ab; % Formula of point - line distance       
        else
            dist = min(d_ax, d_bx); 
        end
    else % if a and b are identical
        dist = d_ax;
    end
end


function b = Dominates(x,y)

    if isstruct(x)
        x = x.Cost;
    end
    
    if isstruct(y)
        y = y.Cost;
    end

    b = all(x <= y) && any(x < y) && all(x < inf);
%     b = all(x <= y) && any(x < y);

end


function pop = DetermineDomination(pop)

    nPop = numel(pop);
    
    for i = 1:nPop
        pop(i).IsDominated = false;
    end
    
    % for the pop 2:nPop-1
    for i = 1:nPop-1
        
        if any(pop(i).Cost == Inf)
            pop(i).IsDominated = true;
        end
        
        for j = i+1:nPop
            
            if Dominates(pop(i), pop(j))
               pop(j).IsDominated = true;
            end
            
            if Dominates(pop(j), pop(i))
               pop(i).IsDominated = true;
            end
            
        end
        
    end
    % for the end pop
    if any(pop(end).Cost == Inf)
            pop(end).IsDominated = true;
    end

    for j = 1:nPop-1
        
        if Dominates(pop(end), pop(j))
           pop(j).IsDominated = true;
        end
        
        if Dominates(pop(j), pop(end))
           pop(end).IsDominated = true;
           break;
        end 
    end

end
function rep = DeleteOneRepMember(rep, gamma)

    % Grid Index of All Repository Members
    GI = [rep.GridIndex];
    
    % Occupied Cells
    OC = unique(GI);
    
    % Number of Particles in Occupied Cells
    N = zeros(size(OC));
    for k = 1:numel(OC)
        N(k) = numel(find(GI == OC(k)));
    end
    
    % Selection Probabilities
    P = exp(gamma*N);
    P = P/sum(P);
    
    % Selected Cell Index
    sci = RouletteWheelSelection(P);
    
    % Selected Cell
    sc = OC(sci);
    
    % Selected Cell Members
    SCM = find(GI == sc);
    
    % Selected Member Index
    smi = randi([1 numel(SCM)]);
    
    % Selected Member
    sm = SCM(smi);
    
    % Delete Selected Member
    rep(sm) = [];

end
%
% Create random paths (solutions)
% 

function sol=CreateRandomSolution(VarSize,VarMin,VarMax) 
    % Random path nodes
    sol.r=unifrnd(VarMin.r,VarMax.r,VarSize);
    sol.psi=unifrnd(VarMin.psi,VarMax.psi,VarSize);
    sol.phi=unifrnd(VarMin.phi,VarMax.phi,VarSize);
end
%{
This model is generated by:
- Loading terrain map
- Creating threats as cylinders
- Creating start and finish points
- Setting ranges and limits
%}

function model=CreateModel()

    H = imread('ChrismasTerrain2.tif'); % Get elevation data
    H (H < 0) = 0;
    MAPSIZE_X = size(H,2); % x index: columns of H
    MAPSIZE_Y = size(H,1); % y index: rows of H
    [X,Y] = meshgrid(1:MAPSIZE_X,1:MAPSIZE_Y); % Create all (x,y) points to plot

    % Threats as cylinders
    R1=70;  % Radius
    x1 = 200; y1 = 230; z1 = 250; % center

    R2=70;  % Radius
    x2 = 600; y2 = 250; z2 = 250; % center
    
    R3=70;  % Radius
    x3 = 450; y3 = 550; z3 = 250; % center
    
    R4=50;  % Radius
    x4 = 700; y4 = 600; z4 = 250; % center
    
    R5=60;  % Radius
    x5 = 200; y5 = 500; z5 = 250; % center
    
    R6=60;  % Radius
    x6 = 500; y6 = 800; z6 = 250; % center
    
%     R7=60;  % Radius
%     x7 = 200; y7 = 500; z7 = 250; % center
    
    % Map limits
    xmin= 1;
    xmax= MAPSIZE_X;
    
    ymin= 1;
    ymax= MAPSIZE_Y;
    
    zmin = 100;
    zmax = 200;  
 
    % Start and end position
    start_location = [50;50;150];
    end_location = [800;800;170];
    
    % Number of path nodes (not including the start position (start node))
    n=10;
    
    % Incorporate map and searching parameters to a model
    model.start=start_location;
    model.end=end_location;
    model.n=n;
    model.xmin=xmin;
    model.xmax=xmax;
    model.zmin=zmin;
    model.ymin=ymin;
    model.ymax=ymax;
    model.zmax=zmax;
    model.MAPSIZE_X = MAPSIZE_X;
    model.MAPSIZE_Y = MAPSIZE_Y;
    model.X = X;
    model.Y = Y;
    model.H = H;
    model.threats = [x1 y1 z1 R1;x2 y2 z2 R2; x3 y3 z3 R3; x4 y4 z4 R4; x5 y5 z5 R5; ...
                    x6 y6 z6 R6];
    PlotModel(model);
end


function Grid = CreateGrid(pop, nGrid, alpha)

    c = [pop.Cost]; 
    
    cmin = min(c, [], 2); 
    cmax = max(c, [], 2);
    
    dc = cmax-cmin;
    cmin = cmin-alpha*dc;
    cmax = cmax+alpha*dc;
    
    nObj = size(c, 1); 
    
    empty_grid.LB = [];
    empty_grid.UB = [];
    Grid = repmat(empty_grid, nObj, 1);
    
    for j = 1:nObj
        
        cj = linspace(cmin(j), cmax(j), nGrid+1); 
        
        Grid(j).LB = [-inf cj];
        Grid(j).UB = [cj +inf];
        
    end

end