function [model, loss] = trainDQN(ERB, batch_size, maxEpoch, model, gamma, r_real_cur, lr, t, maxIter, target_model)
% EMMOP fair-bridge DQN training shim.
% The official code assumes a large replay buffer. Fast benchmark smoke runs
% can reach training before 128 transitions exist, so cap the batch to the
% available buffer without changing the update equations.

    batch_size = min(batch_size, size(ERB, 1));
    if batch_size <= 0
        loss = 0;
        return;
    end
    sampleIdx = randperm(size(ERB, 1));
    dataset = ERB(sampleIdx(1:batch_size), :);
    train_state = cell2mat(dataset(:, 1)');
    train_action = cell2mat(dataset(:, 2)');
    r_real = cell2mat(dataset(:, 3)');
    train_next_state = cell2mat(dataset(:, 4)');
    for epoch = 1:maxEpoch
        [r_pred, model] = forward(train_state, model);
        r_pred = sum(r_pred .* train_action, 1);
        if t == maxIter
            error = r_real + gamma * max(r_real_cur, [], 1) - r_pred;
        else
            [r_pred_next_best, ~] = max(forward(train_next_state, target_model), [], 1);
            error = r_real + gamma * r_pred_next_best - r_pred;
        end
        [model.W, model.B] = backprop(model, lr, error, batch_size, train_action);
    end
    loss = sum(error.^2) ./ batch_size;
end
