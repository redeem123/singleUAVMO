 classdef Chromosome
    
    properties
        rnvec;
        path;
        objs;
        front;
        vel;
        CD;
        rank;
        cons;
        dominationcount=0;
        dominatedset=[];
        dominatedsetlength=0;
        highBound;
        Qtable;
        currentState;
        reward;
    end
    
    methods

        function object=Chromosome(model)
            dim = model.n;
            bound = model.xmax;
            object.highBound(1) = model.xmin;
            for i = 2 : dim-1
                object.highBound(i) = i*bound/model.n; % Higherbounding of *something* in the population
            end

            object.rnvec(1,:) = model.start; % Setting the starting and ending coordinates for the initialized solution
            object.rnvec(model.n,:) = model.end;
        end

        function object=initialize(object,model)% Custom initialization function based on some constraints
            if model.ymax == 200
                varRange = [-20,20];
            else
                varRange = [-5,5];
            end
            
            % Optimized initialization:
            % Instead of while-loop guessing, use a base diagonal with noise
            % and then ensure it's non-decreasing by sorting if needed, 
            % but a better way is linear interpolation + perturbations.
            n = model.n;
            t = linspace(0, 1, n)';
            
            % Linear interpolation between start and end
            object.rnvec(:, 1) = model.start(1) + t * (model.end(1) - model.start(1));
            object.rnvec(:, 2) = model.start(2) + t * (model.end(2) - model.start(2));
            
            % Add noise to intermediate points
            noise = (rand(n-2, 2) - 0.5) * (varRange(2) - varRange(1));
            object.rnvec(2:end-1, 1:2) = object.rnvec(2:end-1, 1:2) + noise;
            
            % Boundary enforcement
            object.rnvec(:, 1) = max(model.xmin, min(model.xmax, object.rnvec(:, 1)));
            object.rnvec(:, 2) = max(model.ymin, min(model.ymax, object.rnvec(:, 2)));
            
            % Use control points directly as the path
            object.path = object.rnvec;
            
            % Adjust for constraints
            object = adjust_constraint_turning_angle(object,model);
        end
        
        function object = custom_cv(object,model)
            const_viol = 0;
            for k = 2 : size(object.path,1)-1 
                if k>3
                    L1 = sqrt((object.path(k,1)-object.path(k-1,1))^2+(object.path(k,2)-object.path(k-1,2))^2);
                    L2 = sqrt((object.path(k-1,1)-object.path(k-2,1))^2+(object.path(k-1,2)-object.path(k-2,2))^2);
                    L3 = sqrt((object.path(k,1)-object.path(k-2,1))^2+(object.path(k,2)-object.path(k-2,2))^2);
                    alpha = acosd((L1^2+L2^2-L3^2)/(2*L1*L2));
                    if alpha < 75%alpha < 120
                        % horizontal turning angle constraint have not satisfied
                        const_viol = const_viol+abs(alpha-75);
                    end
                
                    if object.path(k,1) < model.xmin
                        object.path(k,1) = model.xmin;%object.path(i,1) + model.xmin;
                    end
                    if object.path(k,1) > model.xmax
                        object.path(k,1) = model.xmax;%object.path(i,1) +  model.xmax;
                    end
                    if object.path(k,2) < model.ymin
                        object.path(k,2) = model.ymin;%object.path(i,2) + model.ymin;
                    end
                    if object.path(k,2) > model.ymax
                        object.path(k,2) = model.ymax;%object.path(i,2) + model.ymax;
                    end
                end
                L4 = sqrt((object.path(k,1)-object.path(k-1,1))^2+(object.path(k,2)-object.path(k-1,2))^2); %Euclidean distance between (x2,y2) and (x1,y1) (length of the path)
                beta = atand(abs(object.path(k,3)-object.path(k-1,3))/L4); % Inverse tangent function of the difference of heights of the current point and previous point (checkpoint) divided by the length of the path
                if beta > 60
                    % vertical turning angle constraint have not satisfied
                    const_viol = const_viol + abs(beta-60);
                end
            end
            object.cons = const_viol;
        end
    
        function object = get_cv(object,model)            
            cv = 0;
            object.path(1,3) = model.start(3);%+1500;
            for i = 2 : size(object.path,1)-1                 
                if i>3
                    L1 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2);
                    L2 = sqrt((object.path(i-1,1)-object.path(i-2,1))^2+(object.path(i-1,2)-object.path(i-2,2))^2);
                    L3 = sqrt((object.path(i,1)-object.path(i-2,1))^2+(object.path(i,2)-object.path(i-2,2))^2);
                    alpha = acosd((L1^2+L2^2-L3^2)/(2*L1*L2));
                    if alpha < 75%alpha < 120
                        % horizontal turning angle constraint have not satisfied
                        cv = cv+abs(alpha-75);
                    end
                
                    if object.path(i,1) < model.xmin
                        object.path(i,1) = model.xmin;%object.path(i,1) + model.xmin;
                    end
                    if object.path(i,1) > model.xmax
                        object.path(i,1) = model.xmax;%object.path(i,1) +  model.xmax;
                    end
                    if object.path(i,2) < model.ymin
                        object.path(i,2) = model.ymin;%object.path(i,2) + model.ymin;
                    end
                    if object.path(i,2) > model.ymax
                        object.path(i,2) = model.ymax;%object.path(i,2) + model.ymax;
                    end
                    if object.path(i,3)<model.zmin
                        object.path(i,3) = model.zmin;
                    end
                    if object.path(i,3)>model.zmax
                        object.path(i,3) = model.zmax;
                    end
                end
                L4 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2); %Euclidean distance between (x2,y2) and (x1,y1) (length of the path)
                beta = atand(abs(object.path(i,3)-object.path(i-1,3))/L4); % Inverse tangent function of the difference of heights of the current point and previous point (checkpoint) divided by the length of the path
                if beta > 60
                    % vertical turning angle constraint have not satisfied
                    cv = cv + abs(beta-60);
                end

                if model.xmax == 20
                    v1 = [model.H(floor(object.path(i,2))*10,floor(object.path(i,1))*10);   % H is the base terrain (x,y) values. Gets the altitude at a specific x,y location in the trajectory
                    model.H(floor(object.path(i,2))*10,ceil(object.path(i,1))*10);
                    model.H(ceil(object.path(i,2))*10,floor(object.path(i,1))*10);
                    model.H(ceil(object.path(i,2))*10,ceil(object.path(i,1))*10)];
                else
                    v1 = [model.H(floor(object.path(i,2)),floor(object.path(i,1)));
                    model.H(floor(object.path(i,2)),ceil(object.path(i,1)));
                    model.H(ceil(object.path(i,2)),floor(object.path(i,1)));
                    model.H(ceil(object.path(i,2)),ceil(object.path(i,1)))];
                    % v1 = [model.H(floor(object.path(i,i)),floor(object.path(i,2)));% Modification made because the previous code doesn't make sense. Why assign value for height at y,x to height at x,y
                    % model.H(floor(object.path(i,1)),ceil(object.path(i,2)));
                    % model.H(ceil(object.path(i,1)),floor(object.path(i,2)));
                    % model.H(ceil(object.path(i,i)),ceil(object.path(i,2)))];
                end
                object.path(i,3) = max(v1) + model.safeH;
            end
            object.path(end,3) = model.end(3);
            object.cons = cv;
            %object.path(1,3) = object.path(1,3) + rand*(object.path(2,3)-object.path(1,3));
        end
                

        function [object] = adjust_constraint_turning_angle(object,model) % Adjusts the horizontal and vertical turning angles for the UAV
            
            % Ensure start point clears terrain
            zs = model.H(max(1,floor(model.start(2))), max(1,floor(model.start(1)))) + model.safeH;
            object.path(1,1:3) = [model.start(1), model.start(2), zs];
            if model.ymax == 200
                varRange = [-2,10];
            else
                varRange = [-5,5];
            end
            
            numPoints = size(object.path,1);
            for i = 2 : numPoints-1
                % Horizontal adjustment (limited iterations)
                iter = 0;
                while i > 3 && check_constraint_horizontal_turning_angle(object,i) ~= 0 && iter < 10
                    if rand(1) < 0.5
                        object.path(i,2) = object.path(i-1,2) + rand*(varRange(2)-varRange(1))+varRange(1);
                    else
                        object.path(i,1) = object.path(i-1,1) + rand*(varRange(2)-varRange(1))+varRange(1);                                                                        
                    end
                    object.path(i, 1:2) = max([model.xmin, model.ymin], min([model.xmax, model.ymax], object.path(i, 1:2)));
                    iter = iter + 1;
                end
                
                % Determine terrain height at current point (Max of 4 surrounding pixels for safety)
                x = object.path(i,1);
                y = object.path(i,2);
                x1 = max(1, floor(x)); x2 = min(model.xmax, ceil(x));
                y1 = max(1, floor(y)); y2 = min(model.ymax, ceil(y));
                
                if model.xmax == 20
                    v1 = max([model.H(y1*10, x1*10), model.H(y1*10, x2*10), ...
                              model.H(y2*10, x1*10), model.H(y2*10, x2*10)]);
                else
                    v1 = max([model.H(y1, x1), model.H(y1, x2), ...
                              model.H(y2, x1), model.H(y2, x2)]);
                end
                minH = v1 + model.safeH;
                
                % Vertical adjustment (more deterministic)
                L1 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2);
                maxDeltaZ = L1 * tand(60);
                
                % Apply vertical constraint but NEVER go below terrain height
                % Priority 1: Safety (minH), Priority 2: Smoothness (maxDeltaZ)
                currentZ = minH; 
                if currentZ < object.path(i-1,3) - maxDeltaZ
                    % Descent is too steep, clamp to max descent
                    targetZ = object.path(i-1,3) - maxDeltaZ;
                    % But still must clear terrain
                    object.path(i,3) = max(minH, targetZ);
                elseif currentZ > object.path(i-1,3) + maxDeltaZ
                    % Ascent is too steep, but we MUST climb to clear terrain
                    object.path(i,3) = currentZ; 
                else
                    % Terrain is cleared and pitch is within limits
                    object.path(i,3) = currentZ;
                end
            end
            % Ensure end point clears terrain
            ze = model.H(max(1,floor(model.end(2))), max(1,floor(model.end(1)))) + model.safeH;
            object.path(end,1:3) = [model.end(1), model.end(2), ze];
        end
        

        function [flag] = check_constraint_horizontal_turning_angle(object,i)
            flag = 0;
            L1 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2);
            L2 = sqrt((object.path(i-1,1)-object.path(i-2,1))^2+(object.path(i-1,2)-object.path(i-2,2))^2);
            L3 = sqrt((object.path(i,1)-object.path(i-2,1))^2+(object.path(i,2)-object.path(i-2,2))^2);
            % Avoid precision issues with acos
            cosAlpha = (L1^2+L2^2-L3^2)/(2*L1*L2);
            cosAlpha = max(-1, min(1, cosAlpha));
            alpha = acosd(cosAlpha);
            if alpha < 75
                flag = 1;
            end
        end

        function [flag] = check_constraint_vertical_turning_angle(object,i)
            flag = 0;
            L1 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2); 
            beta = atand(abs(object.path(i,3)-object.path(i-1,3))/L1); 
            if beta > 60
                flag = 1;
            end
        end
        

        function object=evaluate(object, model)
            if nargin < 2
                % Minimum fallback for initialization if model isn't ready
                object.objs = [1e9, 1e9, 1e9, 1e9];
                return;
            end
            
            % Call the master evaluator
            object.objs = evaluate_path(object.path, model);
        end
        
        
    end    
    
end

