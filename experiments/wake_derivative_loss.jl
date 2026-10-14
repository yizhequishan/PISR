function wake_derivative_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, dx, flag_x = eval_diff_tree_array(tree, dataset.X, options, 1)
    flag_x || return L(Inf)
    _, dy, flag_y = eval_diff_tree_array(tree, dataset.X, options, 2)
    flag_y || return L(Inf)

    # Materialize Python targets once before fitting, then keep the hot loop
    # type-stable. Arithmetic and summation order match the P2 reference loss.
    dx_target = OASIS_DX_TARGET::Vector{Float64}
    dy_target = OASIS_DY_TARGET::Vector{Float64}
    value_scale = L(OASIS_VALUE_SCALE::Float64)
    dx_scale = L(OASIS_DX_SCALE::Float64)
    dy_scale = L(OASIS_DY_SCALE::Float64)
    lambda_grad = L(OASIS_LAMBDA_GRAD::Float64)
    value_loss = L(0.0)
    dx_loss = L(0.0)
    dy_loss = L(0.0)
    @inbounds for i in eachindex(prediction)
        value_err = prediction[i] - dataset.y[i]
        dx_err = dx[i] - L(dx_target[i])
        dy_err = dy[i] - L(dy_target[i])
        value_loss += value_err^2
        dx_loss += dx_err^2
        dy_loss += dy_err^2
    end
    count = L(length(prediction))
    value_loss = (value_loss / count) / value_scale
    dx_loss = (dx_loss / count) / dx_scale
    dy_loss = (dy_loss / count) / dy_scale
    gradient_loss = L(0.5) * (dx_loss + dy_loss)
    return value_loss + lambda_grad * gradient_loss
end
