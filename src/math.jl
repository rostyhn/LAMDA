function getIndexFromSQMesh(i::Int64, resolution::Float64)

    #number of points in sq mesh
    #[0:resolution:pi;]
    #push!(phiRange, pi) #ass pi to close the hole at the end introduced by resolution
    thetaRange = [0:resolution:2*pi;] #

    numberY = trunc(Int, pi / resolution) + 2
    numberX = trunc(Int, 2 * pi / resolution) + 1
    number = numberX * numberY
    return trunc(Int, i / number) + 1
end

function signPow(base, exponent)::Float64
    return sign(base) * abs(base)^exponent
end

function qz(phi::Float64, theta::Float64, alpha::Float64, beta::Float64)
    x = signPow(cos(theta), alpha) * signPow(sin(phi), beta)
    y = signPow(sin(theta), alpha) * signPow(sin(phi), beta)
    z = signPow(cos(phi), beta)

    return Point3f(x, y, z)
end

function qx(phi::Float64, theta::Float64, alpha::Float64, beta::Float64)
    x = signPow(cos(phi), beta)
    y = -signPow(sin(theta), alpha) * signPow(sin(phi), beta)
    z = signPow(cos(theta), alpha) * signPow(sin(phi), beta)
    return Point3f(x, y, z)

end

function superquadric(scale::Float64, position::Point3f, principalStretches::Vector{GeometryBasics.Vec{3,Float32}}, K1::Float32, sharpness::Float64, resolution=0.2)
    points = Vector{Point3f}()

    #K2 is the volume perserving fractionalAnisotropy
    #K3 is the mode defining the type of anisotropy: -1 planar to 1 linear

    stretchRatio1 = norm(principalStretches[3])
    stretchRatio2 = norm(principalStretches[2])
    stretchRatio3 = norm(principalStretches[1])

    stretchDirection1 = principalStretches[3] / stretchRatio1
    stretchDirection2 = principalStretches[2] / stretchRatio2
    stretchDirection3 = principalStretches[1] / stretchRatio3


    cl = (stretchRatio1 - stretchRatio2) / (stretchRatio1 + stretchRatio2 + stretchRatio3)   #linear anisotopy
    cp = 2 * (stretchRatio2 - stretchRatio3) / (stretchRatio1 + stretchRatio2 + stretchRatio3) # planar anisotropy

    phiRange = [0:resolution:pi;]  #vertical: south -> north
    push!(phiRange, pi) #ass pi to close the hole at the end introduced by resolution
    thetaRange = [0:resolution:2*pi;] #horizontal: west -> east


    if cl >= cp
        alpha = signPow((1 - cp), sharpness)
        beta = signPow((1 - cl), sharpness)

        for phi in phiRange
            for theta in thetaRange
                push!(points, qx(phi, theta, alpha, beta))
            end
        end
    else
        alpha = (1 - cl)^sharpness
        beta = (1 - cp)^sharpness

        for phi in phiRange
            for theta in thetaRange
                push!(points, qz(phi, theta, alpha, beta))
            end
        end
    end


    scaleMatrix = zeros(3, 3)
    scaleMatrix[1, 1] = stretchRatio1
    scaleMatrix[2, 2] = stretchRatio2
    scaleMatrix[3, 3] = stretchRatio3

    scaleMatrix = scaleMatrix * scale

    rotationMatrix = zeros(3, 3)
    rotationMatrix[:, 1] = stretchDirection1
    rotationMatrix[:, 2] = stretchDirection2
    rotationMatrix[:, 3] = stretchDirection3

    if det(rotationMatrix) < 0
        rotationMatrix[:, 1] = -1 * rotationMatrix[:, 1]
    end

    transform = rotationMatrix * scaleMatrix

    points = Point3f.(Ref(transform) .* points) .+ Ref(position)
    nPhi = length(phiRange)
    nTheta = length(thetaRange)

    indices = Vector{Tuple{UInt32,UInt32,UInt32}}() # triangles over the points

    for y in 1:(nPhi-1)
        for x in 1:nTheta

            #@show "???"
            p11 = x + nTheta * (y - 1)
            p21 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
            p31 = x + nTheta * (y)

            p12 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
            p22 = x < nTheta ? (x + 1) + nTheta * y : 1 + nTheta * y # index 
            p32 = x + nTheta * (y)

            push!(indices, (p11, p31, p21))
            push!(indices, (p32, p22, p12))

        end
    end

    triFaces = TriangleFace.(indices)

    # Create the Mesh
    mesh = GeometryBasics.Mesh(points, triFaces)

    return mesh
end

function angle(a, b)
    return acosd(clamp(a ⋅ b / (norm(a) * norm(b)), -1, 1))
end

function fractionalAnisotropy(ev::Vector{Float64})
    meanEV = (ev[1] + ev[2] + ev[3]) / 3.0
    a =
        sqrt((ev[1] - meanEV)^2 + (ev[2] - meanEV)^2 + (ev[3] - meanEV)^2) /
        sqrt(ev[1]^2 + ev[2]^2 + ev[3]^2)
    return sqrt(3.0 / 2.0) * a
end

function center_atom_positions(p)
    cm = mean(p, dims=1)
    return (p .- cm), cm
end

function com(p, weights)
    # assume weights to be positive
    return sum(p .* weights, dims=1) ./ sum(weights)
end

function pure_align(r1, r2)
    Ra = pinv(r1' * r2) * (r1' * r1)
    U, S, Vh = svd(Ra, full=true)

    Ri = U * Diagonal([1, 1, -1]) * Vh'
    Rb = U * Vh'

    if sum((r1 - r2 * Ri) .^ 2) < sum((r1 - r2 * Rb) .^ 2)
        R = Ri
    else
        R = Rb
    end

    return R, norm(r1 - r2 * R)
end
