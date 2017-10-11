# Copyright 2017 Eric S. Tellez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using KernelMethods
import KernelMethods.Scores: accuracy, recall, precision, f1, precision_recall
import KernelMethods.CrossValidation: montecarlo
import KernelMethods.Supervised: NearNeighborClassifier, optimize!
import SimilaritySearch: L2Distance
using Base.Test

@testset "Scores" begin
    @test accuracy([1,1,1,1,1], [1,1,1,1,1]) == 1.0
    @test accuracy([1,1,1,1,1], [0,0,0,0,0]) == 0.0
    @test accuracy([1,1,1,1,0], [0,1,1,1,1]) == 0.6
    @test precision_recall([0,1,1,1,0,1], [0,1,1,1,1,1]) == Dict{Int,Tuple}(0 => (1.0, 0.5, 2), 1 => (0.8, 1.0, 4))
    @test precision([0,1,1,1,0,1], [0,1,1,1,1,1]) == 0.9
    @test recall([0,1,1,1,0,1], [0,1,1,1,1,1]) == 0.75
    @test precision([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:micro) == (1.0 * 2/6 + 0.8 * 4/6) / 2
    @test recall([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:micro) == (0.5 * 2/6 + 1.0 * 4/6) / 2
    @test f1([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:macro) ≈ (2 * 0.5 / 1.5 + 2 * 0.8 / 1.8) / 2
    @test f1([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:micro) ≈ (2/6 * 2 * 0.5 / 1.5 + 4 / 6 * 2 * 0.8 / 1.8) / 2
end

@testset "CrossValidation" begin
    data = collect(1:100)
    function f(train_X, train_y, test_X, test_y)
        @test train_X == train_y
        @test test_X == test_y
        @test length(train_X ∩ test_X) == 0
        @test length(train_X ∪ test_X) >= 99
        1
    end
    @test montecarlo(f, data, data, runs=10) |> sum == 10
end

@testset "KNN" begin
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = basename(url)
    if !isfile(filename)
        download(url)
    end
    data = readcsv(filename)
    X = data[:, 1:4]
    X = [Float64.(X[i, :]) for i in 1:size(X, 1)]
    y = String.(data[:, 5])
    nnc = NearNeighborClassifier(X, y, L2Distance())
    @test optimize!(nnc, accuracy, runs=3, trainratio=0.2, validationratio=0.2)[1][1] > 0.9
    @test optimize!(nnc, accuracy, runs=3, trainratio=0.3, validationratio=0.3)[1][1] > 0.95
    @test optimize!(nnc, accuracy, runs=3, trainratio=0.7, validationratio=0.3)[1][1] > 0.96
end
