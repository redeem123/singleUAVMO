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
                object.highBound(i) = i*bound/model.n;
            end

            object.rnvec(1,:) = model.start;
            object.rnvec(model.n,:) = model.end;
        end

        function object=initialize(object,model)
            if model.ymax == 200
                varRange = [-20,20];
            else
                varRange = [-5,5];
            end
            bound = model.xmax;
            i = 2;
            while i < model.n
                object.rnvec(i,2) = i*bound/model.n + rand(1)*(varRange(2)-varRange(1))+varRange(1);
                object.rnvec(i,1) = i*bound/model.n + rand(1)*(varRange(2)-varRange(1))+varRange(1);
                if ~(object.rnvec(i,1) < object.rnvec(i-1,1) || object.rnvec(i,2) < object.rnvec(i-1,2))
                    i = i + 1;
                end
            end
            
            a = object.rnvec(:,1);
            a(a < model.ymin) = model.xmin;
            a(a > model.ymax) = model.ymax;
            object.rnvec(:,1) = a;
            a = object.rnvec(:,2);
            a(a < model.ymin) = model.xmin;
            a(a > model.ymax) = model.ymax;
            object.rnvec(:,2) = a;
            object.path = testBspline([object.rnvec(:,1)';object.rnvec(:,2)'],model.xmax)';
            
            object = adjust_constraint_turning_angle(object,model);

        end
        
        function [object] = adjust_constraint_turning_angle(object,model)
            
            object.path(1,3) = model.start(3);
            if model.ymax == 200
                varRange = [-2,10];
            else
                varRange = [-5,5];
            end
            for i = 2 : size(object.path,1)-1
                while i > 3 && check_constraint_horizontal_turning_angle(object,i) ~= 0
                    if rand(1) < 0.5
                        object.path(i,2) = object.path(i-1,2) + rand*(varRange(2)-varRange(1))+varRange(1);
                    else
                        object.path(i,1) = object.path(i-1,1) + rand*(varRange(2)-varRange(1))+varRange(1);                                                                        
                    end
                    object.path = Evolve.check_boundary(object.path,i,model);
                end
                if model.xmax == 20
                    v1 = [model.H(floor(object.path(i,2))*10,floor(object.path(i,1))*10);
                    model.H(floor(object.path(i,2))*10,ceil(object.path(i,1))*10);
                    model.H(ceil(object.path(i,2))*10,floor(object.path(i,1))*10);
                    model.H(ceil(object.path(i,2))*10,ceil(object.path(i,1))*10)];
                else
                    v1 = [model.H(floor(object.path(i,2)),floor(object.path(i,1)));
                    model.H(floor(object.path(i,2)),ceil(object.path(i,1)));
                    model.H(ceil(object.path(i,2)),floor(object.path(i,1)));
                    model.H(ceil(object.path(i,2)),ceil(object.path(i,1)))];
                end
                object.path(i,3) = max(v1) + model.safeH;
                while check_constraint_vertical_turning_angle(object,i) ~= 0
                    j = i;
                    while j > 1
                        if object.path(j,3) < object.path(j-1,3)
                            object.path(j,3) = object.path(j,3) + rand*(object.path(j-1,3)-object.path(j,3));
                            if check_constraint_vertical_turning_angle(object,j) == 0
                                break;
                            end
                        else
                            object.path(j-1,3) = object.path(j-1,3) + rand*(object.path(j,3)-object.path(j-1,3));
                            if check_constraint_vertical_turning_angle(object,j) == 0
                                if j > 2 && check_constraint_vertical_turning_angle(object,j-1) == 0
                                    break;
                                else
                                    j = j - 1;
                                end
                            end
                                
                        end
                        
                    end
                end
            end
            object.path(end,3) = model.end(3);
        end
        

        function [flag] = check_constraint_horizontal_turning_angle(object,i)
            flag = 0;
            L1 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2);
            L2 = sqrt((object.path(i-1,1)-object.path(i-2,1))^2+(object.path(i-1,2)-object.path(i-2,2))^2);
            L3 = sqrt((object.path(i,1)-object.path(i-2,1))^2+(object.path(i,2)-object.path(i-2,2))^2);
            alpha = acosd((L1^2+L2^2-L3^2)/(2*L1*L2));
            if alpha < 75%alpha < 120
                % horizontal turning angle constraint have not satisfied
                flag = 1;
            end
        end

        function [flag] = check_constraint_vertical_turning_angle(object,i)
            flag = 0;
            L1 = sqrt((object.path(i,1)-object.path(i-1,1))^2+(object.path(i,2)-object.path(i-1,2))^2);
%             L1 = sqrt((object.rnvec(x2,1)-object.rnvec(x1,1))^2+(object.rnvec(x2,2)-object.rnvec(x1,2))^2);
            beta = atand(abs(object.path(i,3)-object.path(i-1,3))/L1);
            if beta > 60
                % vertical turning angle constraint have not satisfied
                flag = 1;
            end
        end

        function object=evaluate(object)
            object = BridgeEvaluatePathPopulation(object);
        end
        
        
    end    
    
end

