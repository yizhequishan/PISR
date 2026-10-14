function wake_operator_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, dx, good_x = eval_diff_tree_array(tree, dataset.X, options, 1)
    good_x || return L(Inf)
    _, dy, good_y = eval_diff_tree_array(tree, dataset.X, options, 2)
    good_y || return L(Inf)
    tx = OASIS_DX_TARGET::Vector{Float64}
    ty = OASIS_DY_TARGET::Vector{Float64}
    tk = WAKE_K_TARGET::Vector{Float64}
    sv = L(OASIS_VALUE_SCALE::Float64)
    sx = L(OASIS_DX_SCALE::Float64)
    sy = L(OASIS_DY_SCALE::Float64)
    sk = L(WAKE_K_SCALE::Float64)
    radius = L(WAKE_RADIUS::Float64)
    weight = L(WAKE_K_WEIGHT::Float64)
    lv = L(0.0); lx = L(0.0); ly = L(0.0); lk = L(0.0)
    @inbounds for i in eachindex(prediction)
        lv += (prediction[i] - dataset.y[i])^2
        lx += (dx[i] - L(tx[i]))^2
        ly += (dy[i] - L(ty[i]))^2
        kr = -((dataset.X[1, i] + L(4.0)) * dx[i] + dataset.X[2, i] * dy[i]) / radius
        lk += (kr - L(tk[i]))^2
    end
    n = L(length(prediction))
    return (lv/n)/sv + L(0.5)*((lx/n)/sx + (ly/n)/sy) + weight*(lk/n)/sk
end
