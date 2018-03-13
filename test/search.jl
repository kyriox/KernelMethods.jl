# Copyright 2017,2018 Eric S. Tellez
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

using Base.Test

function loadiris()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = basename(url)
    if !isfile(filename)
        download(url, filename)
    end
    data = readcsv(filename)
    X = data[:, 1:4]
    X = [Float64.(X[i, :]) for i in 1:size(X, 1)]
    y = String.(data[:, 5])
    X, y
end

@testset "encode by farthest points" begin
    using KernelMethods.KMap: search_model
    using KernelMethods.Scores: accuracy, recall
    using SimilaritySearch: L2Distance
    X, y = loadiris()
    models = search_model(X, y, folds=5, keep=10, size=32, score=accuracy)
    @show [m[1] for m in models]
end

# @testset "KlusterClassifier" begin
#     X, y = loadiris()
#     kc = KlusterClassifier(X,y)
#     @show kc
#     @test kc[1][2]>0.9
# end