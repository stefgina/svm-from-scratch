using Convex, SCS

# Generate data.
n = 2; # dimensionality of data
C = 100; # inverse regularization parameter in the objective
N = 100; # number of positive examples
M = 100; # number of negative examples

using Distributions: MvNormal
# positive data points
pos_data = rand(MvNormal([30,30], 5.0), N);


# negative data points
neg_data = rand(MvNormal([10, 10], 5.0), M)

#making the kernel


function svm(pos_data, neg_data, solver=() -> SCS.Optimizer(verbose=0))
    # Create variables for the separating hyperplane w'*x = b.
    w = Variable(n)
    b = Variable()
    # Form the objective.
    obj = 1/2*sumsquares(w) + C*sum(max(1+b-w'*pos_data, 0)) + C*sum(max(1-b+w'*neg_data, 0))
    # Form and solve problem.
    problem = minimize(obj)
    solve!(problem, solver)
    return evaluate(w), evaluate(b)
end;

function predict(feat)
    classification = sign(dot(feat,w)-b)
    return classification
end

function read_input_vector()
    input = readline()
    x = parse(Float64, input)
    return x
end



w, b = svm(pos_data, neg_data);
# Plot our results.
using Plots

dbx = -10:1:50;
dby = (-w[1] * dbx .+ b)/w[2];

dbyplus = (-w[1] * dbx .+ b .+1)/w[2]
dbyminus = (-w[1] * dbx .+ b .-1)/w[2]


plot(pos_data[1,:], pos_data[2,:], st=:scatter, label="Positive points")
plot!(neg_data[1,:], neg_data[2,:], st=:scatter, label="Negative points")
plot!(dbx, dby, label="Decision Boundary")
plot!(dbx, dbyplus , label="db+")
plot!(dbx, dbyminus , label="db-")

println("Please type your input vector to test the SVM ")
println("Type a x : ")
inx = read_input_vector()

println("type a y : ")
iny = read_input_vector()

features = [inx, iny]
if (predict(features))==-1
    print("Classified as Negative")
else
    print("Classified as Positive")
end



plot!([inx], [iny], st=:scatter, color=:yellow, label = "test_point")
